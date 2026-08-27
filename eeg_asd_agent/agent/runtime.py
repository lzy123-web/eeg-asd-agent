from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from eeg_asd_agent.agent.controller import AgentDecisionError, LLMAgentController
from eeg_asd_agent.agent.memory import SQLiteAgentMemory
from eeg_asd_agent.execution import ObservationNormalizer, StateReducer
from eeg_asd_agent.schemas import (
    AgentTurnDecision,
    AssessmentCaseState,
    CaseStatus,
    ToolExecutionOutcome,
    ToolResult,
    ToolStatus,
)
from eeg_asd_agent.tools.registry import ToolRegistry
from eeg_asd_agent.tracing import TraceLogger


@dataclass
class AgentRuntimeOutcome:
    state: AssessmentCaseState
    turns: list[AgentTurnDecision]
    results: list[ToolResult]


class AgentRuntime:
    """Plan → act → observe → check → replan until the Agent finishes."""

    def __init__(
        self,
        registry: ToolRegistry,
        controller: LLMAgentController,
        memory: SQLiteAgentMemory,
        normalizer: ObservationNormalizer,
        trace: TraceLogger,
        *,
        max_steps: int = 24,
        stuck_limit: int = 2,
    ) -> None:
        self.registry = registry
        self.controller = controller
        self.memory = memory
        self.normalizer = normalizer
        self.trace = trace
        self.max_steps = max_steps
        self.stuck_limit = stuck_limit

    def execute(self, state: AssessmentCaseState) -> AgentRuntimeOutcome:
        state.execution_status = CaseStatus.RUNNING
        turns: list[AgentTurnDecision] = []
        results: list[ToolResult] = []
        signatures: dict[str, int] = {}
        initial_step_count = state.step_count
        next_turn_index = self.memory.next_turn_index(state.case_id)
        self.controller.begin_run(initial_step_count)
        self.memory.save_state(state)
        self.trace.event(
            "agent_runtime_started",
            {"case_id": state.case_id, "control_mode": "llm_agentic_next_action"},
        )

        while state.step_count - initial_step_count < self.max_steps:
            try:
                turn = self.controller.decide(state)
            except AgentDecisionError as exc:
                state.execution_status = CaseStatus.BLOCKED
                state.warnings.append(str(exc))
                self.trace.event("agent_decision_rejected", {"error": str(exc)})
                self.memory.save_state(state)
                break

            turns.append(turn)
            state.agent_plan = turn.plan
            state.latest_assistant_message = turn.assistant_message
            state.action_history.append(turn.action)
            state.current_action = turn.action.tool_name
            self.trace.event("agent_turn_decided", {"turn": turn})

            if turn.action.tool_name is None:
                state.execution_status = CaseStatus.COMPLETED
                self.memory.record_turn(
                    state.case_id,
                    next_turn_index + len(turns) - 1,
                    turn,
                    None,
                )
                self.memory.save_state(state)
                break

            signature = self._signature(turn, state)
            signatures[signature] = signatures.get(signature, 0) + 1
            if signatures[signature] >= self.stuck_limit:
                state.execution_status = CaseStatus.BLOCKED
                state.warnings.append(
                    f"stuck agent action detected for {turn.action.tool_name}"
                )
                self.trace.event(
                    "agent_stuck_detected", {"tool_name": turn.action.tool_name}
                )
                self.memory.record_turn(
                    state.case_id,
                    next_turn_index + len(turns) - 1,
                    turn,
                    None,
                )
                self.memory.save_state(state)
                break

            outcome = self._execute_with_retry(turn, state)
            result = outcome.result
            results.append(result)
            observation = self.normalizer.normalize(result)
            update = outcome.state_update.model_copy(
                update={
                    "agent_plan": turn.plan,
                    "latest_assistant_message": turn.assistant_message,
                    "current_action": turn.action.tool_name,
                }
            )
            StateReducer.apply(state, update, observation)
            self.memory.record_turn(
                state.case_id,
                next_turn_index + len(turns) - 1,
                turn,
                observation,
            )
            self.memory.save_state(state)
            self.trace.event(
                "agent_tool_observed",
                {
                    "turn": turn,
                    "result": result,
                    "observation": observation,
                    "state_update": update,
                },
            )
            if turn.action.terminal or state.execution_status in {
                CaseStatus.BLOCKED,
                CaseStatus.FAILED,
                CaseStatus.WAITING,
                CaseStatus.CANCELLED,
            }:
                break
        else:
            state.execution_status = CaseStatus.BLOCKED
            state.warnings.append("agent max_steps reached")
            self.memory.save_state(state)

        if state.execution_status is CaseStatus.RUNNING:
            state.execution_status = CaseStatus.COMPLETED
        self.trace.event(
            "agent_runtime_finished",
            {
                "status": state.execution_status,
                "turn_count": len(turns),
                "tool_count": len(results),
            },
        )
        return AgentRuntimeOutcome(state=state, turns=turns, results=results)

    def _execute_with_retry(
        self,
        turn: AgentTurnDecision,
        state: AssessmentCaseState,
    ) -> ToolExecutionOutcome:
        tool_name = turn.action.tool_name
        assert tool_name is not None
        manifest = self.registry.describe(tool_name)
        retries = 0
        outcome = self.registry.execute_outcome(
            tool_name, state, turn.action.arguments
        )
        while outcome.result.status is ToolStatus.FAILURE and retries < manifest.retry_limit:
            retries += 1
            self.trace.event(
                "agent_tool_retry", {"tool_name": tool_name, "retry": retries}
            )
            outcome = self.registry.execute_outcome(
                tool_name, state, turn.action.arguments
            )
        return outcome

    @staticmethod
    def _signature(
        turn: AgentTurnDecision,
        state: AssessmentCaseState,
    ) -> str:
        state_view = {
            "quality": state.data_quality,
            "readiness": state.model_readiness.value,
            "completed": sorted(state.completed_workflows),
            "failed": sorted(state.failed_workflows),
            "observation_count": len(state.observations),
            "plan_revision": turn.plan.revision,
        }
        raw = json.dumps(
            [turn.action.tool_name, turn.action.arguments, state_view],
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(raw.encode()).hexdigest()
