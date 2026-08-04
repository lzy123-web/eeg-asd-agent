from __future__ import annotations

import json
import re
from typing import Protocol

from eeg_asd_agent.execution import AllowlistProjector, ContextBuilder
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
            state.raw_files
            or state.feature_table
            or state.existing_model_result
            or state.existing_report
            or goal is RequestedGoal.KNOWLEDGE_SEARCH
        )
        if state.input_type is InputKind.INCOMPLETE or not has_usable_input:
            return self._choose(
                state,
                "stop_with_insufficient_input",
                "No feature table, model result, or report is available.",
                observed,
                terminal=True,
            )

        if state.input_type is InputKind.RAW_EEG:
            if "inspect_eeg" not in state.completed_workflows:
                return self._choose(
                    state,
                    "inspect_eeg",
                    "Inspect EEG metadata without exposing the signal payload.",
                    observed,
                )
            if "check_eeg_quality" not in state.completed_workflows:
                return self._choose(
                    state,
                    "check_eeg_quality",
                    "Raw EEG quality must be observed before signal processing.",
                    observed,
                )
            if goal is RequestedGoal.QUALITY_CHECK:
                return ActionDecision(
                    reason="The requested EEG quality check is complete; model, RAG, and report are not allowed.",
                    terminal=True,
                    based_on_observations=observed,
                )
            if state.data_quality != "pass":
                return self._choose(
                    state,
                    "stop_with_insufficient_input",
                    "EEG quality gates block downstream feature and model execution.",
                    observed,
                    terminal=True,
                )
            for tool_name, reason in (
                ("preprocess_eeg", "Apply the versioned deterministic preprocessing profile."),
                ("extract_events", "Read recorded events or explicitly infer fixed windows."),
                ("create_epochs", "Create deterministic epochs before feature extraction."),
                ("extract_eeg_features", "Convert the processed recording into model-compatible summaries."),
            ):
                if tool_name not in state.completed_workflows:
                    return self._choose(state, tool_name, reason, observed)
            if goal is RequestedGoal.FEATURE_EXTRACTION:
                return ActionDecision(
                    reason="The requested EEG feature table is complete; no model or report was requested.",
                    terminal=True,
                    based_on_observations=observed,
                )

        if goal in {
            RequestedGoal.FULL_ASSESSMENT,
            RequestedGoal.QUALITY_CHECK,
            RequestedGoal.READINESS_CHECK,
        } and (
            state.input_type is not InputKind.RAW_EEG
            and state.feature_table
            and "check_data_quality" not in state.completed_workflows
        ):
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
            and "run_asd_baseline_model" not in state.completed_workflows
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
            and "run_asd_baseline_model" not in state.completed_workflows
            and "run_asd_baseline_model" not in state.failed_workflows
        ):
            available, _ = self._available("run_asd_baseline_model", state)
            if available:
                return self._choose(
                    state,
                    "run_asd_baseline_model",
                    "Readiness permits restricted current-case inference with the pretrained research baseline.",
                    observed,
                )

        if (
            goal is RequestedGoal.FULL_ASSESSMENT
            and state.existing_model_result is None
            and state.model_readiness is not ModelReadiness.BLOCKED
            and "research_replay_classifier" not in state.completed_workflows
            and "research_replay_classifier" not in state.failed_workflows
            and "run_asd_baseline_model" in state.failed_workflows
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
        if needs_knowledge and "build_asd_rag_query" not in state.completed_workflows:
            return self._choose(
                state,
                "build_asd_rag_query",
                "Build a case-grounded query from executed observations before retrieval.",
                observed,
            )
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


class NextActionProvider(Protocol):
    name: str

    def generate_json(self, purpose: str, payload: dict[str, object]) -> str: ...


class LLMNextActionPlanner:
    """Ask a provider for one closed-world action after every observation.

    Invalid JSON, unknown tools, illegal arguments, and unmet preconditions fall
    back to the deterministic state planner. The provider is never given raw EEG,
    absolute paths, full feature values, or internal ToolResult objects.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        provider: NextActionProvider,
        *,
        fallback: NextActionPlanner | None = None,
        max_steps: int = 16,
    ) -> None:
        self.registry = registry
        self.provider = provider
        self.fallback = fallback or NextActionPlanner(registry)
        self.max_steps = max_steps
        self.payload_history: list[dict[str, object]] = []
        self.fallback_count = 0

    def decide(self, state: AssessmentCaseState) -> ActionDecision:
        fallback = self.fallback.decide(state)
        goal = state.requested_goal or RequestedGoal.UNSUPPORTED
        projector = AllowlistProjector(self.registry)
        context_builder = ContextBuilder(self.registry, projector)
        available = self._legal_actions(state, goal, fallback)
        payload: dict[str, object] = {
            "user_request": self._safe_request(state.user_request),
            "goal": goal.value,
            "current_state": context_builder.planning_context(state),
            "previous_observations": [
                projector.project(item, goal).model_dump(mode="json")
                for item in state.observations
            ],
            "capability_manifests": [
                {
                    "name": manifest.name,
                    "version": manifest.version,
                    "category": manifest.category,
                    "required_inputs": manifest.required_inputs,
                    "optional_inputs": manifest.optional_inputs,
                    "preconditions": manifest.preconditions,
                    "produced_outputs": manifest.produced_outputs,
                    "limitations": manifest.limitations,
                    "unsupported_claims": manifest.unsupported_claims,
                    "implementation_kind": manifest.real_or_fake.value,
                }
                for manifest in self.registry.manifests()
                if goal in manifest.supported_goals
            ],
            "allowed_next_actions": available,
            "completed_workflows": state.completed_workflows,
            "failed_workflows": state.failed_workflows,
            "remaining_step_budget": max(0, self.max_steps - state.step_count),
            "rules": [
                "Return exactly one ActionDecision JSON object.",
                "Choose only from allowed_next_actions, or return terminal=true with tool_name=null.",
                "Never claim an unexecuted tool succeeded.",
                "Use capability manifests and prior observations; do not assume the deterministic fallback choice.",
            ],
        }
        self.payload_history.append(payload)
        try:
            provider_payload = payload
            if self.provider.name.startswith("offline"):
                provider_payload = {
                    **payload,
                    "fixture_fallback_action": fallback.model_dump(mode="json"),
                }
            raw = self.provider.generate_json("next_action", provider_payload)
            decision = ActionDecision.model_validate_json(self._extract_json_object(raw))
            self._validate(decision, available, state)
            return decision
        except Exception as exc:
            self.fallback_count += 1
            return fallback.model_copy(
                update={"reason": f"LLM planner fallback ({type(exc).__name__}): {fallback.reason}"}
            )

    def _validate(
        self,
        decision: ActionDecision,
        available: list[str],
        state: AssessmentCaseState,
    ) -> None:
        if decision.tool_name is None:
            if not decision.terminal:
                raise ValueError("tool_name=null requires terminal=true")
            if available:
                raise ValueError("provider attempted to stop before workflow completion")
            return
        if decision.tool_name not in available:
            raise ValueError("provider selected a tool outside allowed_next_actions")
        manifest = self.registry.describe(decision.tool_name)
        allowed_arguments = set(manifest.required_inputs + manifest.optional_inputs)
        if set(decision.arguments) - allowed_arguments:
            raise ValueError("provider supplied illegal tool arguments")
        missing = set(manifest.required_inputs) - set(decision.arguments)
        if missing:
            raise ValueError("provider omitted required tool arguments")
        ready, unmet = self.registry.preconditions_met(decision.tool_name, state)
        if not ready:
            raise ValueError(f"provider selected tool with unmet preconditions: {unmet}")
        observation_ids = {item.observation_id for item in state.observations}
        unknown_observations = set(decision.based_on_observations) - observation_ids
        if unknown_observations:
            raise ValueError("provider cited unknown observations")

    def _legal_actions(
        self,
        state: AssessmentCaseState,
        goal: RequestedGoal,
        fallback: ActionDecision,
    ) -> list[str]:
        if goal is RequestedGoal.UNSUPPORTED:
            return [fallback.tool_name] if fallback.tool_name is not None else []
        legal: list[str] = []
        for manifest in self.registry.manifests():
            name = manifest.name
            if goal not in manifest.supported_goals:
                continue
            if name in state.completed_workflows or name in state.failed_workflows:
                continue
            if manifest.category == "control" and name != fallback.tool_name:
                continue
            ready, _unmet = self.registry.preconditions_met(name, state)
            if ready:
                legal.append(name)
        if fallback.tool_name is not None and fallback.tool_name not in legal:
            ready, _unmet = self.registry.preconditions_met(fallback.tool_name, state)
            if ready:
                legal.append(fallback.tool_name)
        return sorted(legal)

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

    @staticmethod
    def _safe_request(value: str) -> str:
        redacted = re.sub(r"(?:/[^\s]+)+", "[LOCAL_PATH_REDACTED]", value)
        return redacted[:1000]
