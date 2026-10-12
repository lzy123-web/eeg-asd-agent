from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from eeg_asd_agent.adapters import EEGAgentKnowledgeAdapter, EEGSignalAdapter
from eeg_asd_agent.cli import app
from eeg_asd_agent.schemas import AssessmentCaseState, InputKind, RequestedGoal
from eeg_asd_agent.tools.domain_catalog import build_agent_registry
from eeg_asd_agent.workflow import run_workflow


def test_asd_eeg_profile_is_an_enabled_real_tool() -> None:
    manifest = build_agent_registry().describe("build_eeg_asd_profile")
    assert manifest.enabled is True
    assert manifest.real_or_fake.value == "real"
    assert all(
        "research_highlights" in fields
        for fields in manifest.allowlisted_fields.values()
    )


def test_eegagent_prepared_chunks_are_reused_without_faiss(tmp_path: Path) -> None:
    rag_dir = tmp_path / "RAG"
    rag_dir.mkdir()
    with (rag_dir / "chunks.pkl").open("wb") as handle:
        pickle.dump(
            [
                "EEG background activity and spectral power reporting guidance.",
                "Unrelated sleep text.",
            ],
            handle,
        )
    result = EEGAgentKnowledgeAdapter(tmp_path).retrieve(
        "EEG spectral power background activity",
        limit=1,
    )
    assert result.mode == "eegagent_prepared_chunks_bm25"
    assert len(result.items) == 1
    assert result.items[0].source == "EEGAgent:RAG/chunks.pkl"
    assert result.items[0].evidence_id.startswith("EEGAGENT-KB-")


def test_edf_plus_annotations_are_extracted_as_recorded_events() -> None:
    raw = b"+0\x14\x14\x00+1.5\x150\x149\x14\x00+2.0\x150\x142\x14\x00"
    events, labels, source = EEGSignalAdapter._edf_events(raw, [], 100.0, 1000)
    assert source == "edf_annotations"
    assert labels == ("9", "2")
    assert np.array_equal(events, np.asarray([[150, 9], [200, 2]]))


def test_assess_requires_a_real_llm_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eeg_path = tmp_path / "input.edf"
    eeg_path.touch()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    result = CliRunner().invoke(app, ["assess", "--eeg", str(eeg_path)])
    assert result.exit_code != 0
    assert "不会降级到离线模式" in result.output


def test_raw_eeg_generates_non_diagnostic_profile_report(tmp_path: Path) -> None:
    state = AssessmentCaseState(
        case_id="baseline-smoke",
        input_type=InputKind.RAW_EEG,
        raw_files=["synthetic://baseline-smoke"],
        modalities=["EEG"],
        metadata_summary={
            "age": 8,
            "sex": "unknown",
            "sampling_rate_hz": 128,
            "duration_seconds": 10,
            "channel_count": 8,
        },
    )
    artifacts = run_workflow(
        state,
        "分析 EEG，形成 ASD 儿童研究性辅助评估报告",
        tmp_path / "assessment",
    )

    plan = json.loads(artifacts["plan"].read_text(encoding="utf-8"))
    actions = [decision["tool_name"] for decision in plan["decisions"]]
    assert "extract_eeg_features" in actions
    assert "build_eeg_asd_profile" in actions
    assert "generate_report_spec" in actions
    assert "render_report" in actions

    report = json.loads(artifacts["report_spec"].read_text(encoding="utf-8"))
    assert report["report_type"] == "non_diagnostic_research_assessment"
    details = report["data_summary"]["details"]
    assert details["technical_description"]["channel_count"] == 8
    assert "eeg_description" in details
    assert "clinical_correlation" in details
    assert details["recording"]["channel_count"] == 8
    assert details["subject_context"]["age"] == 8
    assert details["research_highlights"]
    assert report["human_review_required"] is True


def test_full_report_request_keeps_summaries_and_evidence_visible(tmp_path: Path) -> None:
    state = AssessmentCaseState(
        case_id="full-context-regression",
        input_type=InputKind.RAW_EEG,
        raw_files=["synthetic://full-context-regression"],
        modalities=["EEG"],
        metadata_summary={"sampling_rate_hz": 128, "duration_seconds": 10},
    )
    request = (
        "分析该儿童的 EEG，并生成 ASD 相关研究性评估报告。"
        "请说明信号质量、频谱、脑区、左右差异、连接性、知识证据与局限。"
    )
    output = tmp_path / "full-context"
    run_workflow(state, request, output)
    assert state.requested_goal is RequestedGoal.FULL_ASSESSMENT
    payload = json.loads((output / "llm_payload.json").read_text(encoding="utf-8"))
    visible = {item["tool_name"]: item["visible_facts"] for item in payload["visible_results"]}
    assert visible["extract_eeg_features"]["assessment_summary"]
    assert visible["build_eeg_asd_profile"]["spectral_summary"]
    assert visible["retrieve_asd_knowledge"]["items"][0]["evidence_id"]
    report = json.loads((output / "report_spec.json").read_text(encoding="utf-8"))
    eeg = report["data_summary"]["details"]["eeg_description"]
    assert all(
        eeg[name]["status"] == "available"
        for name in ("spectral_analysis", "regional_analysis", "laterality", "connectivity")
    )
    assert report["knowledge_interpretation"]["evidence_ids"]
