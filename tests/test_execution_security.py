from pathlib import Path

from eeg_asd_agent.execution import (
    AllowlistProjector,
    ContextBuilder,
    ObservationNormalizer,
    WorkflowExecutor,
)
from eeg_asd_agent.planning import DeterministicPlanner
from eeg_asd_agent.schemas import (
    AssessmentCaseState,
    ImplementationKind,
    InputKind,
    ModelReadiness,
    RequestedGoal,
    ToolResult,
    ToolStatus,
)
from eeg_asd_agent.tools.builtin import build_default_registry
from eeg_asd_agent.tracing import TraceLogger


def test_normalizer_removes_sensitive_fields() -> None:
    result = ToolResult(
        tool_name="check_data_quality",
        manifest_version="1.0",
        status=ToolStatus.SUCCESS,
        data={
            "quality_status": "pass",
            "raw_signal": [1, 2],
            "local_path": "/private/person.edf",
            "nested": {"api_key": "secret", "safe": 1},
        },
        implementation_kind=ImplementationKind.REAL,
    )
    facts = ObservationNormalizer().normalize(result).facts
    assert "raw_signal" not in facts
    assert "local_path" not in facts
    assert facts["nested"] == {"safe": 1}


def test_projector_is_deny_by_default() -> None:
    registry = build_default_registry()
    result = ToolResult(
        tool_name="check_data_quality",
        manifest_version="1.0",
        status=ToolStatus.SUCCESS,
        data={"quality_status": "pass", "new_unreviewed_field": "must stay hidden"},
        implementation_kind=ImplementationKind.REAL,
    )
    observation = ObservationNormalizer().normalize(result)
    visible = AllowlistProjector(registry).project(observation, RequestedGoal.FULL_ASSESSMENT)
    assert visible.visible_facts == {"quality_status": "pass"}


def test_report_context_contains_no_raw_paths() -> None:
    registry = build_default_registry()
    state = AssessmentCaseState(
        case_id="secure",
        input_kind=InputKind.RAW_EEG,
        raw_files=["/private/patient.edf"],
    )
    context = ContextBuilder(registry, AllowlistProjector(registry)).report_context(
        state, RequestedGoal.FULL_ASSESSMENT
    )
    rendered = str(context)
    assert "patient.edf" not in rendered
    assert "/private" not in rendered


def test_executor_skips_classifier_when_readiness_blocked(tmp_path: Path) -> None:
    registry = build_default_registry()
    state = AssessmentCaseState(
        case_id="blocked",
        input_kind=InputKind.FEATURE_TABLE,
        feature_table={"f1": 1.0},
    )
    plan = DeterministicPlanner(registry).plan(state, RequestedGoal.FULL_ASSESSMENT)
    trace = TraceLogger(tmp_path / "trace.jsonl", state.trace_id)
    outcome = WorkflowExecutor(registry, ObservationNormalizer(), trace).execute(state, plan)
    classifier = [item for item in outcome.results if item.tool_name == "run_asd_classifier"][0]
    assert state.model_readiness is ModelReadiness.BLOCKED
    assert classifier.status is ToolStatus.SKIPPED


def test_trace_redacts_paths_and_secrets(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    trace = TraceLogger(path, "t")
    trace.event("test", {"local_path": "/private/file", "api_key": "abc", "safe": "yes"})
    content = path.read_text(encoding="utf-8")
    assert "/private/file" not in content
    assert '"abc"' not in content
    assert '"safe": "yes"' in content

