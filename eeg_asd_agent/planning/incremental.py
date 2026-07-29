from __future__ import annotations

from eeg_asd_agent.schemas import (
    ActionDecision,
    AssessmentCaseState,
    InputKind,
    ModelReadiness,
    RequestedGoal,
)
from eeg_asd_agent.tools import ToolRegistry


class NextActionPlanner:
    """Choose exactly one next action from current state and closed manifests."""

    def __init__(self, registry: ToolRegistry, *, use_manifests: bool = True) -> None:
        self.registry = registry
        self.use_manifests = use_manifests

    def decide(self, state: AssessmentCaseState) -> ActionDecision:
        observed = [item.observation_id for item in state.observations]
        goal = state.requested_goal or RequestedGoal.UNSUPPORTED

        if goal is RequestedGoal.UNSUPPORTED:
            return self._choose(
                state,
                "stop_with_unsupported_request",
                "The parsed request is outside the closed EEG-ASD workflow.",
                observed,
                terminal=True,
            )

        if "inspect_case" not in state.completed_workflows:
            return self._choose(
                state,
                "inspect_case",
                "Inspect the typed case before selecting a domain action.",
                observed,
            )

        has_usable_input = bool(
            state.feature_table or state.existing_model_result or state.existing_report
        )
        if state.input_type is InputKind.INCOMPLETE or not has_usable_input:
            return self._choose(
                state,
                "stop_with_insufficient_input",
                "No feature table, model result, or report is available.",
                observed,
                terminal=True,
            )

        if goal in {
            RequestedGoal.FULL_ASSESSMENT,
            RequestedGoal.QUALITY_CHECK,
            RequestedGoal.READINESS_CHECK,
        } and state.feature_table and "check_data_quality" not in state.completed_workflows:
            return self._choose(
                state,
                "check_data_quality",
                "Data quality must be observed before model readiness is interpreted.",
                observed,
            )

        if goal in {
            RequestedGoal.FULL_ASSESSMENT,
            RequestedGoal.READINESS_CHECK,
        } and state.feature_table and "check_model_readiness" not in state.completed_workflows:
            return self._choose(
                state,
                "check_model_readiness",
                "Readiness is unknown and must be checked from current case state.",
                observed,
            )

        if goal in {RequestedGoal.QUALITY_CHECK, RequestedGoal.READINESS_CHECK}:
            return ActionDecision(
                reason="The requested data/readiness check is complete; no RAG or report is allowed.",
                terminal=True,
                based_on_observations=observed,
            )

        if (
            state.existing_model_result is not None
            and "inspect_existing_model_result" not in state.completed_workflows
            and "research_replay_classifier" not in state.completed_workflows
        ):
            return self._choose(
                state,
                "inspect_existing_model_result",
                "An existing model result is available and must be allowlisted without re-running inference.",
                observed,
            )

        if (
            goal is RequestedGoal.FULL_ASSESSMENT
            and state.existing_model_result is None
            and state.model_readiness is not ModelReadiness.BLOCKED
            and "research_replay_classifier" not in state.completed_workflows
            and "research_replay_classifier" not in state.failed_workflows
        ):
            available, _ = self._available("research_replay_classifier", state)
            if available:
                return self._choose(
                    state,
                    "research_replay_classifier",
                    "The readiness observation permits a historical OOF replay.",
                    observed,
                )

        needs_knowledge = goal in {
            RequestedGoal.FULL_ASSESSMENT,
            RequestedGoal.KNOWLEDGE_SEARCH,
        }
        if (
            needs_knowledge
            and "retrieve_asd_knowledge" not in state.completed_workflows
            and "retrieve_asd_knowledge" not in state.failed_workflows
        ):
            return self._choose(
                state,
                "retrieve_asd_knowledge",
                "Knowledge interpretation was requested and has not yet been attempted.",
                observed,
                arguments={"query": state.user_request},
            )

        if "generate_report_spec" not in state.completed_workflows:
            return self._choose(
                state,
                "generate_report_spec",
                "Executed observations are sufficient for a bounded structured report.",
                observed,
            )

        if "render_report" not in state.completed_workflows:
            return self._choose(
                state,
                "render_report",
                "A validated ReportSpec is available for deterministic rendering.",
                observed,
            )

        return ActionDecision(
            reason="The requested workflow and deterministic rendering are complete.",
            terminal=True,
            based_on_observations=observed,
        )

    def _choose(
        self,
        state: AssessmentCaseState,
        name: str,
        reason: str,
        observed: list[str],
        *,
        arguments: dict[str, object] | None = None,
        terminal: bool = False,
    ) -> ActionDecision:
        if name not in self.registry.names():
            raise ValueError(f"planner selected unregistered tool: {name}")
        available, unmet = self._available(name, state)
        if not available:
            if name.startswith("stop_with_"):
                raise ValueError(f"stop tool preconditions are invalid: {unmet}")
            return self._choose(
                state,
                "stop_with_insufficient_input",
                f"{name} preconditions are unmet: {', '.join(unmet)}",
                observed,
                terminal=True,
            )
        return ActionDecision(
            tool_name=name,
            arguments=arguments or {},
            reason=reason,
            terminal=terminal,
            based_on_observations=observed,
        )

    def _available(
        self,
        name: str,
        state: AssessmentCaseState,
    ) -> tuple[bool, list[str]]:
        if not self.use_manifests:
            return True, []
        return self.registry.preconditions_met(name, state)
