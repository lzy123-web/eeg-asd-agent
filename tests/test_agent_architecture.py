from __future__ import annotations

import json
from pathlib import Path

import pytest

from eeg_asd_agent.agent import (
    AgentContextBuilder,
    AgentRuntime,
    LLMAgentController,
    SQLiteAgentMemory,
)
from eeg_asd_agent.execution import AllowlistProjector, ObservationNormalizer, StateReducer
from eeg_asd_agent.providers import OpenAIResponsesProvider
from eeg_asd_agent.schemas import (
    AgentTurnDecision,
    AssessmentCaseState,
    InputKind,
    ModelReadiness,
    RequestedGoal,
)
from eeg_asd_agent.tools.domain_catalog import build_agent_registry
from eeg_asd_agent.tracing import TraceLogger
from eeg_asd_agent.workflow import run_workflow

ROOT = Path(__file__).resolve().parents[1]


def _case(name: str) -> AssessmentCaseState:
    return AssessmentCaseState.model_validate_json(
        (ROOT / "demo_cases" / name).read_text(encoding="utf-8")
    )


def _turn(
    tool_name: str | None,
    *,
    revision: int,
    observations: list[str],
    reason: str = "scripted semantic choice",
) -> str:
    terminal = tool_name is None
    return json.dumps(
        {
            "plan": {
                "revision": revision,
                "objective": "Complete the current EEG-ASD user goal",
                "strategy": "Choose tools from manifests and replan after observations.",
                "steps": (
                    [
                        {
                            "step_id": f"p{revision:02d}",
                            "objective": tool_name.replace("_", " "),
                            "status": "in_progress",
                            "candidate_tools": [tool_name],
                            "depends_on": [],
                        }
                    ]
                    if tool_name is not None
                    else []
                ),
                "completion_criteria": ["The requested task is satisfied."],
            },
            "action": {
                "tool_name": tool_name,
                "arguments": {},
                "reason": reason,
                "terminal": terminal,
                "based_on_observations": observations,
            },
            "goal_satisfied": terminal,
            "assistant_message": "done" if terminal else None,
            "replan_reason": "latest observation changed the current state",
        }
    )


def test_domain_catalog_declares_full_agent_surface_without_executing_stubs() -> None:
    registry = build_agent_registry()
    review = registry.describe("request_human_review")
    assert review.enabled is True
    assert review.real_or_fake.value == "real"
    for name in (
        "analyze_eeg_patterns",
        "build_eeg_asd_profile",
        "estimate_asd_severity",
        "estimate_ability_dimensions",
        "compare_assessment_results",
        "parse_assessment_report",
        "conduct_structured_assessment",
        "visualize_eeg_analysis",
        "train_asd_research_model",
    ):
        manifest = registry.describe(name)
        assert manifest.enabled is False
        assert manifest.real_or_fake.value == "stub"
        assert manifest.description
        assert manifest.use_when
        assert registry.execute(name, _case("core_a_feature_assessment.json")).status.value == (
            "blocked"
        )


def test_human_review_tool_pauses_case_with_explicit_state_update() -> None:
    registry = build_agent_registry()
    state = _case("core_a_feature_assessment.json")
    state.completed_workflows = ["inspect_case"]
    outcome = registry.execute_outcome("request_human_review", state, {})
    observation = ObservationNormalizer().normalize(outcome.result)
    StateReducer.apply(state, outcome.state_update, observation)
    assert outcome.result.status.value == "success"
    assert state.execution_status.value == "waiting"
    assert state.pending_requirements


def test_agent_context_exposes_options_not_raw_assets_or_feature_values(
    tmp_path: Path,
) -> None:
    registry = build_agent_registry()
    state = AssessmentCaseState(
        case_id="context-case",
        input_type=InputKind.RAW_EEG,
        raw_files=["/Users/private/participant.edf"],
        feature_table={"parietal_alpha": 12345.6789},
        existing_model_result={"predicted_pattern": "imported"},
        completed_workflows=["inspect_case"],
        requested_goal=RequestedGoal.FULL_ASSESSMENT,
        user_request="验证已有结果并重新分析 /Users/private/participant.edf",
    )
    memory = SQLiteAgentMemory(tmp_path / "memory.sqlite")
    context = AgentContextBuilder(
        registry,
        AllowlistProjector(registry),
        memory,
    ).build(state, registry.available_actions(state), max_steps=24)
    serialized = json.dumps(context, ensure_ascii=False)
    assert "/Users/private" not in serialized
    assert "12345.6789" not in serialized
    assert "participant.edf" not in serialized
    assert context["case_state"]["asset_inventory"]["feature_table_present"] is True
    assert {"inspect_eeg", "inspect_existing_model_result", "build_asd_rag_query"} <= set(
        context["allowed_next_actions"]
    )


def test_same_assets_different_user_goals_can_choose_different_actions(
    tmp_path: Path,
) -> None:
    class GoalAwareProvider:
        name = "goal_aware"

        def __init__(self) -> None:
            self.payloads: list[dict[str, object]] = []

        def generate_json(self, purpose: str, payload: dict[str, object]) -> str:
            assert purpose == "agent_turn"
            self.payloads.append(payload)
            tool = (
                "run_asd_baseline_model"
                if "重新" in str(payload["user_request"])
                else "inspect_existing_model_result"
            )
            return _turn(tool, revision=1, observations=[])

    registry = build_agent_registry()
    provider = GoalAwareProvider()
    memory = SQLiteAgentMemory(tmp_path / "memory.sqlite")
    controller = LLMAgentController(
        registry,
        provider,
        AgentContextBuilder(registry, AllowlistProjector(registry), memory),
    )
    base = _case("core_b_existing_model.json")
    base.completed_workflows = [
        "inspect_case",
        "check_data_quality",
        "check_model_readiness",
    ]
    base.data_quality = "pass"
    base.model_readiness = ModelReadiness.RESTRICTED
    base.requested_goal = RequestedGoal.FULL_ASSESSMENT
    base.feature_table = {
        "higuchi_fd": 1.1,
        "arousal_mean": 0.2,
    }

    reuse = base.model_copy(deep=True)
    reuse.user_request = "解释已有模型结果"
    recompute = base.model_copy(deep=True)
    recompute.user_request = "重新运行模型验证已有结果"

    assert controller.decide(reuse).action.tool_name == "inspect_existing_model_result"
    assert controller.decide(recompute).action.tool_name == "run_asd_baseline_model"
    for payload in provider.payloads:
        assert {
            "inspect_existing_model_result",
            "run_asd_baseline_model",
        } <= set(payload["allowed_next_actions"])


def test_runtime_replans_after_real_observation_and_persists_memory(
    tmp_path: Path,
) -> None:
    class ObservationDrivenProvider:
        name = "observation_driven"

        def generate_json(self, purpose: str, payload: dict[str, object]) -> str:
            assert purpose == "agent_turn"
            observations = list(payload["previous_observations"])
            observation_ids = [str(item["observation_id"]) for item in observations]
            if not observations:
                return _turn("check_data_quality", revision=1, observations=[])
            if len(observations) == 1:
                assert observations[-1]["tool_name"] == "check_data_quality"
                assert observations[-1]["visible_facts"]["quality_status"] == "pass"
                return _turn(
                    "build_asd_rag_query",
                    revision=2,
                    observations=observation_ids,
                    reason="quality passed, so retrieve knowledge requested by the user",
                )
            return _turn(None, revision=3, observations=observation_ids)

    registry = build_agent_registry()
    state = _case("core_a_feature_assessment.json")
    state.user_request = "检查数据后根据结果决定是否检索 ASD 知识"
    state.requested_goal = RequestedGoal.FULL_ASSESSMENT
    state.completed_workflows = ["inspect_case"]
    memory = SQLiteAgentMemory(tmp_path / "memory.sqlite")
    controller = LLMAgentController(
        registry,
        ObservationDrivenProvider(),
        AgentContextBuilder(registry, AllowlistProjector(registry), memory),
    )
    outcome = AgentRuntime(
        registry,
        controller,
        memory,
        ObservationNormalizer(),
        TraceLogger(tmp_path / "trace.jsonl", state.trace_id),
    ).execute(state)
    assert [turn.action.tool_name for turn in outcome.turns] == [
        "check_data_quality",
        "build_asd_rag_query",
        None,
    ]
    assert outcome.turns[1].action.based_on_observations == [
        outcome.state.observations[0].observation_id
    ]
    restored = memory.load_state(state.case_id)
    assert restored is not None
    assert restored.agent_plan is not None
    assert restored.agent_plan.revision == 3
    assert len(memory.snapshot_for_llm(state.case_id)["recent_turns"]) == 3


def test_invalid_llm_action_is_rejected_without_hidden_pipeline_fallback(
    tmp_path: Path,
) -> None:
    class InvalidProvider:
        name = "invalid"

        def generate_json(self, purpose: str, payload: dict[str, object]) -> str:
            del purpose, payload
            return _turn("shell", revision=1, observations=[])

    registry = build_agent_registry()
    state = _case("core_a_feature_assessment.json")
    state.user_request = "分析 EEG"
    state.requested_goal = RequestedGoal.FULL_ASSESSMENT
    memory = SQLiteAgentMemory(tmp_path / "memory.sqlite")
    outcome = AgentRuntime(
        registry,
        LLMAgentController(
            registry,
            InvalidProvider(),
            AgentContextBuilder(registry, AllowlistProjector(registry), memory),
        ),
        memory,
        ObservationNormalizer(),
        TraceLogger(tmp_path / "trace.jsonl", state.trace_id),
    ).execute(state)
    assert outcome.results == []
    assert state.execution_status.value == "blocked"
    assert any("unknown tools" in warning for warning in state.warnings)


def test_raw_eeg_only_runs_complete_agent_protocol_without_questionnaire(
    tmp_path: Path,
) -> None:
    state = _case("01_raw_eeg_full_assessment.json")
    artifacts = run_workflow(
        state,
        "分析这个 EEG，完成 ASD 相关研究评估并生成报告",
        tmp_path / "raw-agent",
    )
    plan = json.loads(artifacts["plan"].read_text(encoding="utf-8"))
    actions = [item["tool_name"] for item in plan["decisions"]]
    assert plan["planner_kind"] == "offline_agentic_simulation"
    assert actions[0] == "inspect_case"
    assert "extract_eeg_features" in actions
    assert "run_asd_baseline_model" in actions
    assert "retrieve_asd_knowledge" in actions
    assert actions[-1] is None
    assert "conduct_structured_assessment" not in actions
    assert state.execution_status.value == "completed"
    assert artifacts["memory"].exists()
    assert artifacts["agent_turns"].exists()
    assert artifacts["agent_contexts"].exists()


def test_openai_provider_has_strict_agent_turn_schema() -> None:
    schema, name = OpenAIResponsesProvider._schema("agent_turn")
    assert name == "agent_turn"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])


def test_agent_turn_requires_observation_before_terminal_stop() -> None:
    raw = json.loads(_turn("inspect_case", revision=1, observations=[]))
    raw["action"]["terminal"] = True
    with pytest.raises(ValueError, match="observe its result"):
        AgentTurnDecision.model_validate(raw)


def test_sqlite_memory_resumes_without_overwriting_turns_or_step_budget(
    tmp_path: Path,
) -> None:
    state = _case("core_a_feature_assessment.json")
    first_output = tmp_path / "first"
    first = run_workflow(
        state,
        "只检查数据质量",
        first_output,
    )
    memory = SQLiteAgentMemory(first["memory"])
    first_turn_count = memory.turn_count(state.case_id)
    first_step_count = state.step_count
    assert first_turn_count >= 3

    resumed = memory.load_state(state.case_id)
    assert resumed is not None
    resumed.action_history = []
    resumed.current_action = None
    resumed.latest_assistant_message = None
    second = run_workflow(
        resumed,
        "重新检索 ASD 知识并更新报告",
        tmp_path / "second",
        memory_path=first["memory"],
    )
    plan = json.loads(second["plan"].read_text(encoding="utf-8"))
    actions = [item["tool_name"] for item in plan["decisions"]]
    assert actions[:2] == ["build_asd_rag_query", "retrieve_asd_knowledge"]
    assert "generate_report_spec" in actions
    assert "render_report" in actions
    assert memory.turn_count(state.case_id) > first_turn_count
    assert resumed.step_count > first_step_count
