from __future__ import annotations

import json
from pathlib import Path

import pytest

from eeg_asd_agent.execution import (
    AllowlistProjector,
    IncrementalWorkflowExecutor,
    ObservationNormalizer,
)
from eeg_asd_agent.intent import RuleBasedIntentParser
from eeg_asd_agent.planning import NextActionPlanner
from eeg_asd_agent.reporting import MarkdownRenderer
from eeg_asd_agent.schemas import (
    AssessmentCaseState,
    ImplementationKind,
    Observation,
    RequestedGoal,
    ToolStatus,
)
from eeg_asd_agent.tools import UnknownToolError
from eeg_asd_agent.tools.core_workflow import build_core_registry
from eeg_asd_agent.tracing import TraceLogger
from eeg_asd_agent.workflow import run_workflow

ROOT = Path(__file__).resolve().parents[1]


def _case(name: str) -> AssessmentCaseState:
    return AssessmentCaseState.model_validate_json(
        (ROOT / "demo_cases" / name).read_text(encoding="utf-8")
    )


def _execute(
    state: AssessmentCaseState,
    request: str,
    tmp_path: Path,
    *,
    registry: object | None = None,
) -> AssessmentCaseState:
    active_registry = registry or build_core_registry()
    state.user_request = request
    state.requested_goal = RuleBasedIntentParser().parse(request)
    IncrementalWorkflowExecutor(
        active_registry,  # type: ignore[arg-type]
        ObservationNormalizer(),
        TraceLogger(tmp_path / "trace.jsonl", state.trace_id),
    ).execute(state, NextActionPlanner(active_registry))  # type: ignore[arg-type]
    return state


def _actions(state: AssessmentCaseState) -> list[str | None]:
    return [item.tool_name for item in state.action_history]


def test_feature_input_skips_raw_eeg_tools_and_redecides_from_observation(
    tmp_path: Path,
) -> None:
    state = _execute(
        _case("core_a_feature_assessment.json"),
        "根据现有特征完成 ASD 辅助评估并生成报告。",
        tmp_path,
    )
    actions = _actions(state)
    assert "inspect_eeg" not in actions
    assert "preprocess_eeg" not in actions
    readiness_index = actions.index("check_model_readiness")
    baseline = state.action_history[readiness_index + 1]
    assert baseline.tool_name == "run_asd_baseline_model"
    assert len(baseline.based_on_observations) == 3
    assert "research_replay_classifier" not in actions


def test_existing_model_result_is_not_rerun(tmp_path: Path) -> None:
    state = _execute(
        _case("core_b_existing_model.json"),
        "只解释当前模型结果并检索相关知识。",
        tmp_path,
    )
    assert "inspect_existing_model_result" in _actions(state)
    assert "research_replay_classifier" not in _actions(state)


def test_blocked_readiness_forbids_classifier(tmp_path: Path) -> None:
    state = _execute(
        _case("core_c_readiness_only.json"),
        "根据现有特征完成 ASD 辅助评估并生成报告。",
        tmp_path,
    )
    assert state.model_readiness.value == "blocked"
    assert "research_replay_classifier" not in _actions(state)
    assert "generate_report_spec" in _actions(state)


def test_rag_failure_generates_limited_report(tmp_path: Path) -> None:
    class FailingKnowledge:
        def retrieve(self, query: str, limit: int = 5) -> object:
            del query, limit
            raise RuntimeError("injected RAG outage")

    registry = build_core_registry(knowledge_adapter=FailingKnowledge())
    state = _execute(
        _case("core_a_feature_assessment.json"),
        "根据现有特征完成 ASD 辅助评估并生成报告。",
        tmp_path,
        registry=registry,
    )
    assert "retrieve_asd_knowledge" in state.failed_workflows
    assert state.report_spec is not None
    assert state.report_spec.knowledge_interpretation.status == "unavailable"
    assert "knowledge_interpretation" in state.report_spec.unavailable_sections


def test_readiness_only_stops_before_rag_and_report(tmp_path: Path) -> None:
    state = _execute(
        _case("core_c_readiness_only.json"),
        "只检查数据和模型是否具备执行条件，不生成完整报告。",
        tmp_path,
    )
    assert "retrieve_asd_knowledge" not in _actions(state)
    assert "generate_report_spec" not in _actions(state)
    assert state.report_spec is None


def test_unknown_tool_is_rejected() -> None:
    registry = build_core_registry()
    with pytest.raises(UnknownToolError):
        registry.execute_outcome(
            "shell",
            _case("core_a_feature_assessment.json"),
        )


def test_nested_allowlist_is_deny_by_default() -> None:
    registry = build_core_registry()
    observation = Observation(
        invocation_id="inv",
        tool_name="retrieve_asd_knowledge",
        status=ToolStatus.SUCCESS,
        facts={
            "query": "ASD",
            "mode": "offline_local_index",
            "new_top_level": "hidden",
            "items": [
                {
                    "evidence_id": "e1",
                    "title": "Allowed",
                    "retrieved_chunk": "Allowed source chunk",
                    "summary": "Allowed summary",
                    "source": "source",
                    "relevance": "match",
                    "limitations": ["background only"],
                    "internal_stack": "hidden",
                    "absolute_path": "/private/index.bin",
                }
            ],
        },
        implementation_kind=ImplementationKind.OFFLINE,
    )
    visible = AllowlistProjector(registry).project(
        observation, RequestedGoal.FULL_ASSESSMENT
    )
    rendered = json.dumps(visible.model_dump(mode="json"), ensure_ascii=False)
    assert "new_top_level" not in rendered
    assert "internal_stack" not in rendered
    assert "absolute_path" not in rendered
    assert "Allowed summary" in rendered


def test_invalid_report_spec_cannot_render() -> None:
    registry = build_core_registry()
    outcome = registry.execute_outcome(
        "render_report",
        _case("core_a_feature_assessment.json"),
    )
    assert outcome.result.status is ToolStatus.BLOCKED
    assert "content" not in outcome.result.data


def test_renderer_is_stable_for_same_spec(tmp_path: Path) -> None:
    state = _execute(
        _case("core_a_feature_assessment.json"),
        "根据现有特征完成 ASD 辅助评估并生成报告。",
        tmp_path,
    )
    assert state.report_spec is not None
    renderer = MarkdownRenderer()
    assert renderer.render(state.report_spec) == renderer.render(state.report_spec)


def test_full_incremental_e2e_writes_grounded_artifacts(tmp_path: Path) -> None:
    state = _case("core_a_feature_assessment.json")
    artifacts = run_workflow(
        state,
        "根据现有特征完成 ASD 辅助评估并生成报告。",
        tmp_path / "run",
    )
    assert all(path.exists() for path in artifacts.values())
    payload = json.loads(artifacts["payload"].read_text(encoding="utf-8"))
    assert set(payload) == {
        "case_id",
        "user_request",
        "executed_workflows",
        "visible_results",
        "tool_limitations",
        "missing_information",
        "report_spec_schema",
        "non_diagnostic_rules",
    }
    assert "feature_table" not in payload
    assert "generate_report_spec" not in payload["executed_workflows"]
    assert "render_report" not in payload["executed_workflows"]
    assert state.execution_status.value == "completed"
