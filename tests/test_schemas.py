import pytest
from pydantic import ValidationError

from eeg_asd_agent.schemas import (
    AssessmentCaseState,
    FinalAssessmentReportSpec,
    InputKind,
    PlanSpec,
    PlanStep,
    RequestedGoal,
    ToolCapabilityManifest,
)


def test_case_schema_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AssessmentCaseState(case_id="c", input_kind=InputKind.RAW_EEG, unknown=True)


def test_manifest_name_is_closed_format() -> None:
    with pytest.raises(ValidationError):
        ToolCapabilityManifest(
            name="Shell Tool",
            category="unsafe",
            implementation_ref="none",
            real_or_fake="unavailable",
        )


def test_plan_rejects_forward_dependency() -> None:
    with pytest.raises(ValidationError):
        PlanSpec(
            case_id="c",
            goal=RequestedGoal.QUALITY_CHECK,
            planner_kind="test",
            steps=[PlanStep(step_id="s1", tool_name="inspect_case", depends_on=["s2"])],
        )


def test_plan_rejects_duplicate_step_id() -> None:
    with pytest.raises(ValidationError):
        PlanSpec(
            case_id="c",
            goal=RequestedGoal.QUALITY_CHECK,
            planner_kind="test",
            steps=[
                PlanStep(step_id="s1", tool_name="inspect_case"),
                PlanStep(step_id="s1", tool_name="check_data_quality"),
            ],
        )


def test_report_schema_rejects_free_text() -> None:
    with pytest.raises(ValidationError):
        FinalAssessmentReportSpec.model_validate({"free_text": "not a report spec"})

