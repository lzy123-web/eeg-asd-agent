from __future__ import annotations

import json
import re
from typing import Protocol

from eeg_asd_agent.agent.context import AgentContextBuilder
from eeg_asd_agent.schemas import AgentTurnDecision, AssessmentCaseState
from eeg_asd_agent.tools.registry import ToolRegistry


class AgentProvider(Protocol):
    name: str

    def generate_json(self, purpose: str, payload: dict[str, object]) -> str: ...


class AgentDecisionError(ValueError):
    pass


class LLMAgentController:
    """LLM-owned planning and next-action selection over a closed harness."""

    def __init__(
        self,
        registry: ToolRegistry,
        provider: AgentProvider,
        context_builder: AgentContextBuilder,
        *,
        max_steps: int = 24,
    ) -> None:
        self.registry = registry
        self.provider = provider
        self.context_builder = context_builder
        self.max_steps = max_steps
        self.step_offset = 0
        self.payload_history: list[dict[str, object]] = []

    def begin_run(self, current_step_count: int) -> None:
        """Start a new request budget without discarding cumulative case history."""
        self.step_offset = current_step_count

    def decide(self, state: AssessmentCaseState) -> AgentTurnDecision:
        allowed = self.registry.available_actions(state)
        payload = self.context_builder.build(
            state,
            allowed,
            max_steps=self.step_offset + self.max_steps,
        )
        self.payload_history.append(payload)
        try:
            raw = self.provider.generate_json("agent_turn", payload)
        except Exception as exc:
            raise AgentDecisionError(
                f"Agent Provider call failed: {type(exc).__name__}"
            ) from exc
        try:
            decision = AgentTurnDecision.model_validate_json(
                self._extract_json_object(raw)
            )
        except Exception as exc:
            raise AgentDecisionError(
                f"provider returned an invalid AgentTurnDecision: {type(exc).__name__}"
            ) from exc
        self._validate(decision, allowed, state)
        return decision

    def _validate(
        self,
        decision: AgentTurnDecision,
        allowed: list[str],
        state: AssessmentCaseState,
    ) -> None:
        known_tools = self.registry.names()
        for step in decision.plan.steps:
            unknown = set(step.candidate_tools) - known_tools
            if unknown:
                raise AgentDecisionError(
                    f"plan references unknown tools: {sorted(unknown)}"
                )

        action = decision.action
        if action.tool_name is None:
            return
        planned_tools = {
            tool_name
            for step in decision.plan.steps
            for tool_name in step.candidate_tools
        }
        if action.tool_name not in planned_tools:
            raise AgentDecisionError(
                f"tool {action.tool_name!r} is not represented in the current AgentPlan"
            )
        if action.tool_name not in allowed:
            raise AgentDecisionError(
                f"tool {action.tool_name!r} is outside allowed_next_actions"
            )
        manifest = self.registry.describe(action.tool_name)
        allowed_arguments = set(manifest.required_inputs + manifest.optional_inputs)
        unexpected = set(action.arguments) - allowed_arguments
        if unexpected:
            raise AgentDecisionError(f"illegal tool arguments: {sorted(unexpected)}")
        missing = set(manifest.required_inputs) - set(action.arguments)
        if missing:
            raise AgentDecisionError(f"missing tool arguments: {sorted(missing)}")
        ready, unmet = self.registry.preconditions_met(action.tool_name, state)
        if not ready:
            raise AgentDecisionError(f"hard tool preconditions are unmet: {unmet}")
        observation_ids = {item.observation_id for item in state.observations}
        unknown_observations = set(action.based_on_observations) - observation_ids
        if unknown_observations:
            raise AgentDecisionError(
                f"decision cites unknown observations: {sorted(unknown_observations)}"
            )

    @staticmethod
    def _extract_json_object(raw: str) -> str:
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        first = stripped.find("{")
        last = stripped.rfind("}")
        if first < 0 or last <= first:
            raise ValueError("provider returned no JSON object")
        candidate = stripped[first : last + 1]
        json.loads(candidate)
        return candidate
