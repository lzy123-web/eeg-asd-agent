import json

import pytest

from eeg_asd_agent.intent import RuleBasedIntentParser
from eeg_asd_agent.planning import DeterministicPlanner, LLMPlanner, PlanValidationError
from eeg_asd_agent.providers import OfflineProvider
from eeg_asd_agent.schemas import AssessmentCaseState, InputKind, PlanSpec, PlanStep, RequestedGoal
from eeg_asd_agent.tools.builtin import build_default_registry


@pytest.mark.parametrize(
    ("query", "goal"),
    [
        ("只检查 EEG 数据质量", RequestedGoal.QUALITY_CHECK),
        ("只提取 EEG 特征", RequestedGoal.FEATURE_EXTRACTION),
        ("检索 ASD 知识", RequestedGoal.KNOWLEDGE_SEARCH),
        ("给出干预知识", RequestedGoal.INTERVENTION),
        ("完成 ASD 辅助评估", RequestedGoal.FULL_ASSESSMENT),
        ("直接确诊 ASD", RequestedGoal.UNSUPPORTED),
    ],
)
def test_intent_parser(query: str, goal: RequestedGoal) -> None:
    assert RuleBasedIntentParser().parse(query) is goal


def test_state_aware_planner_skips_raw_tools_for_features() -> None:
    registry = build_default_registry()
    state = AssessmentCaseState(
        case_id="features",
        input_kind=InputKind.FEATURE_TABLE,
        feature_table={"f1": 1.0},
    )
    plan = DeterministicPlanner(registry).plan(state, RequestedGoal.FULL_ASSESSMENT)
    names = [step.tool_name for step in plan.steps]
    assert "inspect_eeg" not in names
    assert "preprocess_eeg" not in names
    assert "check_model_readiness" in names


def test_planner_skips_completed_workflow() -> None:
    registry = build_default_registry()
    state = AssessmentCaseState(
        case_id="done",
        input_kind=InputKind.RAW_EEG,
        raw_files=["synthetic://x"],
        completed_workflows=["inspect_case"],
    )
    plan = DeterministicPlanner(registry).plan(state, RequestedGoal.QUALITY_CHECK)
    assert "inspect_case" not in [step.tool_name for step in plan.steps]


def test_planner_rejects_unregistered_tool() -> None:
    registry = build_default_registry()
    planner = DeterministicPlanner(registry)
    plan = PlanSpec(
        case_id="c",
        goal=RequestedGoal.QUALITY_CHECK,
        planner_kind="bad",
        steps=[PlanStep(step_id="s1", tool_name="shell")],
    )
    with pytest.raises(PlanValidationError):
        planner.validate(plan)


def test_llm_planner_falls_back_on_invalid_json() -> None:
    class InvalidProvider:
        def generate_json(self, purpose: str, payload: dict[str, object]) -> str:
            return "not-json"

    registry = build_default_registry()
    state = AssessmentCaseState(
        case_id="fallback",
        input_kind=InputKind.RAW_EEG,
        raw_files=["synthetic://x"],
    )
    plan = LLMPlanner(registry, InvalidProvider()).plan(state, RequestedGoal.QUALITY_CHECK)
    assert plan.planner_kind == "deterministic_fallback"


def test_offline_llm_plan_is_strict_json() -> None:
    registry = build_default_registry()
    state = AssessmentCaseState(
        case_id="offline",
        input_kind=InputKind.RAW_EEG,
        raw_files=["synthetic://x"],
    )
    plan = LLMPlanner(registry, OfflineProvider()).plan(state, RequestedGoal.QUALITY_CHECK)
    assert json.loads(plan.model_dump_json())["schema_version"] == "1.0"
