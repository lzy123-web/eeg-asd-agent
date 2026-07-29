import json
from pathlib import Path

import pytest

from eeg_asd_agent.schemas import AssessmentCaseState
from eeg_asd_agent.workflow import run_workflow

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("filename", "query"),
    [
        ("01_raw_eeg_full_assessment.json", "分析 EEG，完成 ASD 辅助评估并检索知识"),
        ("04_feature_table_full_assessment.json", "根据现有特征完成 ASD 辅助评估并生成报告"),
        ("06_existing_medical_report.json", "根据现有报告检索相关知识"),
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
    assert report["evaluation_mode"] == "fixture"


def test_repeated_run_report_content_is_consistent(tmp_path: Path) -> None:
    case_text = (ROOT / "demo_cases" / "02_raw_eeg_quality_only.json").read_text(
        encoding="utf-8"
    )
    first = AssessmentCaseState.model_validate_json(case_text)
    second = AssessmentCaseState.model_validate_json(case_text)
    run_workflow(first, "只检查 EEG 数据质量", tmp_path / "one")
    run_workflow(second, "只检查 EEG 数据质量", tmp_path / "two")
    report_one = json.loads((tmp_path / "one" / "report_spec.json").read_text(encoding="utf-8"))
    report_two = json.loads((tmp_path / "two" / "report_spec.json").read_text(encoding="utf-8"))
    report_one.pop("generated_at")
    report_two.pop("generated_at")
    assert report_one == report_two
