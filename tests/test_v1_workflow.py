from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eeg_asd_agent.adapters import EEGSignalAdapter, SeniorMLPBaselineAdapter
from eeg_asd_agent.planning import LLMNextActionPlanner
from eeg_asd_agent.providers import OfflineReportGenerator, OpenAIResponsesProvider
from eeg_asd_agent.schemas import AssessmentCaseState, InputKind, RequestedGoal
from eeg_asd_agent.tools import ToolPreconditionError
from eeg_asd_agent.tools.core_workflow import build_core_registry
from eeg_asd_agent.workflow import run_workflow

ROOT = Path(__file__).resolve().parents[1]


def _load_case(name: str) -> AssessmentCaseState:
    return AssessmentCaseState.model_validate_json(
        (ROOT / "demo_cases" / name).read_text(encoding="utf-8")
    )


def _actions(output: Path) -> list[str | None]:
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    return [item["tool_name"] for item in plan["decisions"]]


def test_eeg_manifest_chain_is_real_versioned_and_closed() -> None:
    registry = build_core_registry()
    for name in (
        "inspect_eeg",
        "check_eeg_quality",
        "preprocess_eeg",
        "extract_events",
        "create_epochs",
        "extract_eeg_features",
    ):
        manifest = registry.describe(name)
        assert manifest.version == "eeg-v1.0"
        assert manifest.real_or_fake.value == "real"
        assert manifest.produced_outputs
        assert manifest.allowlisted_fields


def test_npz_eeg_adapter_extracts_finite_features(tmp_path: Path) -> None:
    sfreq = 256
    time = np.arange(sfreq * 8) / sfreq
    data = np.stack(
        [
            20e-6 * np.sin(2 * np.pi * 10 * time),
            18e-6 * np.sin(2 * np.pi * 10.5 * time),
            12e-6 * np.sin(2 * np.pi * 6 * time),
            11e-6 * np.sin(2 * np.pi * 6.5 * time),
        ]
    )
    source = tmp_path / "recording.npz"
    np.savez(
        source,
        data=data,
        sfreq=np.asarray([sfreq]),
        channel_names=np.asarray(["F3", "F4", "P3", "P4"]),
        unit=np.asarray(["v"]),
    )
    state = AssessmentCaseState(
        case_id="npz-signal",
        input_type=InputKind.RAW_EEG,
        raw_files=[str(source)],
        modalities=["EEG"],
    )
    adapter = EEGSignalAdapter()
    assert adapter.inspect(state)["source_format"] == "npz"
    assert adapter.quality(state)["quality_status"] == "pass"
    extracted = adapter.extract_features(state)
    assert extracted["feature_count"] >= 14
    assert all(np.isfinite(value) for value in extracted["features"].values())


def test_pretrained_baseline_runs_current_case_without_torch() -> None:
    adapter = SeniorMLPBaselineAdapter()
    prediction = adapter.predict(
        {
            "higuchi_fd": 1.102631575,
            "arousal_mean": 0.1026075,
            "arousal_max": 0.6,
            "arousal_std": 0.260971602,
            "parietal_alpha": 0.001702,
            "beta_gamma_mean": 5.356931975,
        }
    )
    assert prediction["model_version"].startswith("senior-mlp-ability-v1+sha256:")
    assert prediction["scaler_version"].startswith("proxy-sha256:")
    assert prediction["predicted_pattern"] in adapter.output_labels
    assert sum(prediction["class_scores"].values()) == pytest.approx(1.0, abs=2e-6)
    assert prediction["readiness_status"] == "restricted"
    assert "ASD probability" in prediction["score_meaning"]


@pytest.mark.parametrize(
    ("case_name", "prompt", "required", "forbidden", "has_report"),
    [
        (
            "01_raw_eeg_full_assessment.json",
            "分析这个 EEG，完成 ASD 辅助评估，并检索相关知识生成报告",
            {"extract_eeg_features", "run_asd_baseline_model", "build_asd_rag_query", "render_report"},
            {"research_replay_classifier"},
            True,
        ),
        (
            "04_feature_table_full_assessment.json",
            "根据现有特征完成 ASD 辅助评估并生成报告",
            {"check_data_quality", "run_asd_baseline_model", "retrieve_asd_knowledge", "render_report"},
            {"inspect_eeg", "preprocess_eeg", "research_replay_classifier"},
            True,
        ),
        (
            "core_b_existing_model.json",
            "解释已有模型结果，检索相关知识并生成报告",
            {"inspect_existing_model_result", "build_asd_rag_query", "render_report"},
            {"run_asd_baseline_model", "research_replay_classifier", "extract_eeg_features"},
            True,
        ),
        (
            "02_raw_eeg_quality_only.json",
            "只检查这个 EEG 的质量，不运行模型，不检索，不生成报告",
            {"inspect_eeg", "check_eeg_quality"},
            {"run_asd_baseline_model", "retrieve_asd_knowledge", "generate_report_spec"},
            False,
        ),
    ],
)
def test_four_required_e2e_routes(
    tmp_path: Path,
    case_name: str,
    prompt: str,
    required: set[str],
    forbidden: set[str],
    has_report: bool,
) -> None:
    output = tmp_path / case_name.removesuffix(".json")
    artifacts = run_workflow(_load_case(case_name), prompt, output)
    actions = set(_actions(output))
    assert required <= actions
    assert not (forbidden & actions)
    assert ("report_spec" in artifacts) is has_report
    if has_report:
        report = json.loads((output / "report_spec.json").read_text(encoding="utf-8"))
        assert report["human_review_required"] is True
        assert report["knowledge_interpretation"]["evidence_ids"]


def test_quality_observation_blocks_full_workflow(tmp_path: Path) -> None:
    sfreq = 128
    source = tmp_path / "artifact.npz"
    np.savez(
        source,
        data=np.full((4, sfreq * 6), 1.0),
        sfreq=np.asarray([sfreq]),
        channel_names=np.asarray(["F3", "F4", "P3", "P4"]),
        unit=np.asarray(["v"]),
    )
    state = AssessmentCaseState(
        case_id="quality-block",
        input_type=InputKind.RAW_EEG,
        raw_files=[str(source)],
        modalities=["EEG"],
    )
    output = tmp_path / "blocked"
    artifacts = run_workflow(
        state,
        "完成 EEG ASD 辅助评估并生成报告",
        output,
    )
    actions = _actions(output)
    assert actions[:3] == ["inspect_case", "inspect_eeg", "check_eeg_quality"]
    assert actions[3] == "stop_with_insufficient_input"
    assert "run_asd_baseline_model" not in actions
    assert "retrieve_asd_knowledge" not in actions
    assert "report_spec" not in artifacts
    assert state.data_quality == "blocked"


def test_llm_next_action_loop_receives_sanitized_context(tmp_path: Path) -> None:
    class EchoNextActionProvider:
        name = "scripted_provider"

        def __init__(self) -> None:
            self.payloads: list[dict[str, object]] = []
            self.reporter = OfflineReportGenerator()

        def generate_json(self, purpose: str, payload: dict[str, object]) -> str:
            if purpose == "next_action":
                self.payloads.append(payload)
                return json.dumps(payload["fallback_action"], ensure_ascii=False)
            return self.reporter.generate_json(purpose, payload)

    provider = EchoNextActionProvider()
    output = tmp_path / "llm-loop"
    artifacts = run_workflow(
        _load_case("01_raw_eeg_full_assessment.json"),
        "分析 EEG 并生成报告",
        output,
        planner_kind="llm",
        provider=provider,
    )
    assert "planner_payloads" in artifacts
    assert len(provider.payloads) >= 10
    serialized = json.dumps(provider.payloads, ensure_ascii=False)
    assert "synthetic://" not in serialized
    assert "/Users/" not in serialized
    assert "delta_relative_power\"" not in serialized
    assert all(len(payload["allowed_next_actions"]) <= 1 for payload in provider.payloads)


def test_registry_rejects_illegal_arguments() -> None:
    registry = build_core_registry()
    with pytest.raises(ToolPreconditionError):
        registry.execute_outcome(
            "inspect_case",
            _load_case("04_feature_table_full_assessment.json"),
            {"shell": "whoami"},
        )


def test_invalid_llm_action_falls_back_to_legal_deterministic_action() -> None:
    class InvalidProvider:
        name = "invalid_provider"

        def generate_json(self, purpose: str, payload: dict[str, object]) -> str:
            del purpose, payload
            return json.dumps(
                {
                    "tool_name": "shell",
                    "arguments": {"command": "whoami"},
                    "reason": "invalid",
                    "terminal": False,
                    "based_on_observations": [],
                }
            )

    registry = build_core_registry()
    state = _load_case("04_feature_table_full_assessment.json")
    state.user_request = "完成 ASD 辅助评估"
    state.requested_goal = RequestedGoal.FULL_ASSESSMENT
    planner = LLMNextActionPlanner(registry, InvalidProvider())
    decision = planner.decide(state)
    assert decision.tool_name == "inspect_case"
    assert planner.fallback_count == 1


def test_openai_provider_uses_closed_strict_next_action_schema() -> None:
    schema, name = OpenAIResponsesProvider._schema("next_action")
    assert name == "next_action"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    arguments = schema["properties"]["arguments"]
    assert arguments["additionalProperties"] is False
