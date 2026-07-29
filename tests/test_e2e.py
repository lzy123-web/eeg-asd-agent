import json
from pathlib import Path

import pytest

from eeg_asd_agent.schemas import AssessmentCaseState
from eeg_asd_agent.workflow import run_workflow

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("filename", "query"),
    [
        ("core_a_feature_assessment.json", "根据现有特征完成 ASD 辅助评估并生成报告。"),
        ("core_b_existing_model.json", "只解释当前模型结果并检索相关知识。"),
    ],
)
def test_three_input_state_e2e(tmp_path: Path, filename: str, query: str) -> None:
    state = AssessmentCaseState.model_validate_json(
        (ROOT / "demo_cases" / filename).read_text(encoding="utf-8")
    )
    output = tmp_path / filename.removesuffix(".json")
    artifacts = run_workflow(state, query, output)
    required = {
        "plan.json",
        "case_state.json",
        "observations.jsonl",
        "llm_payload.json",
        "report_spec.json",
        "report.md",
        "report.html",
        "trace.jsonl",
    }
    assert required <= {path.name for path in artifacts.values()}
    payload = (output / "llm_payload.json").read_text(encoding="utf-8")
    assert "synthetic://" not in payload
    report = json.loads((output / "report_spec.json").read_text(encoding="utf-8"))
    assert report["report_type"] == "non_diagnostic_research_assessment"
    assert report["human_review_required"] is True


def test_readiness_only_stops_without_report(tmp_path: Path) -> None:
    state = AssessmentCaseState.model_validate_json(
        (ROOT / "demo_cases" / "core_c_readiness_only.json").read_text(encoding="utf-8")
    )
    artifacts = run_workflow(
        state,
        "只检查数据和模型是否具备执行条件，不生成完整报告。",
        tmp_path / "readiness",
    )
    assert "report_spec" not in artifacts
    assert state.completed_workflows == [
        "inspect_case",
        "check_data_quality",
        "check_model_readiness",
    ]


def test_repeated_run_report_content_is_consistent(tmp_path: Path) -> None:
    case_text = (ROOT / "demo_cases" / "core_a_feature_assessment.json").read_text(
        encoding="utf-8"
    )
    first = AssessmentCaseState.model_validate_json(case_text)
    second = AssessmentCaseState.model_validate_json(case_text)
    request = "根据现有特征完成 ASD 辅助评估并生成报告。"
    run_workflow(first, request, tmp_path / "one")
    run_workflow(second, request, tmp_path / "two")
    report_one = json.loads((tmp_path / "one" / "report_spec.json").read_text(encoding="utf-8"))
    report_two = json.loads((tmp_path / "two" / "report_spec.json").read_text(encoding="utf-8"))
    report_one.pop("generated_at")
    report_two.pop("generated_at")
    for report in (report_one, report_two):
        for item in report["provenance"]:
            item.pop("observation_id")
    assert report_one == report_two
