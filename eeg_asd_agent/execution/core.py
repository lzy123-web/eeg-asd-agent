from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from eeg_asd_agent.schemas import (
    ActionDecision,
    AssessmentCaseState,
    CaseStatus,
    FinalAssessmentReportSpec,
    LLMVisibleToolResult,
    ModelReadiness,
    Observation,
    PlanSpec,
    RequestedGoal,
    StateUpdate,
    ToolExecutionOutcome,
    ToolResult,
    ToolStatus,
)
from eeg_asd_agent.tools import ToolRegistry
from eeg_asd_agent.tracing import TraceLogger

_DENIED_KEY = re.compile(r"(raw|signal|samples?|path|file|password|secret|token|api[_-]?key)", re.I)


class ObservationNormalizer:
    def normalize(self, result: ToolResult) -> Observation:
        return Observation(
            invocation_id=result.invocation_id,
            tool_name=result.tool_name,
            status=result.status,
            facts=self._safe(result.data),
            warnings=result.warnings,
            limitations=result.limitations,
            provenance=[f"tool:{result.tool_name}@{result.manifest_version}"],
            implementation_kind=result.implementation_kind,
        )

    def _safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): self._safe(item)
                for key, item in value.items()
                if not _DENIED_KEY.search(str(key))
            }
        if isinstance(value, list):
            return [self._safe(item) for item in value]
        if isinstance(value, str) and (value.startswith("/") or re.match(r"^[A-Za-z]:\\", value)):
            return "[LOCAL_PATH_REDACTED]"
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)


class AllowlistProjector:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def project(
        self,
        observation: Observation,
        workflow: RequestedGoal,
    ) -> LLMVisibleToolResult:
        manifest = self.registry.describe(observation.tool_name)
        allowed = set(
            manifest.allowlisted_fields.get(workflow.value, [])
            + manifest.allowlisted_fields.get("*", [])
        )
        visible = self._project_paths(observation.facts, allowed)
        return LLMVisibleToolResult(
            observation_id=observation.observation_id,
            tool_name=observation.tool_name,
            status=observation.status,
            visible_facts=visible,
            limitations=list(dict.fromkeys(manifest.limitations + observation.limitations)),
            provenance_labels=observation.provenance,
            implementation_kind=observation.implementation_kind,
        )

    def _project_paths(self, facts: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
        projected: dict[str, Any] = {}
        for path in sorted(allowed):
            parts = path.replace("[]", ".*").split(".")
            self._copy_path(facts, projected, parts)
        return projected

    def _copy_path(self, source: Any, target: Any, parts: list[str]) -> None:
        if not parts:
            return
        part = parts[0]
        if _DENIED_KEY.search(part):
            return
        if part == "*":
            if not isinstance(source, list) or not isinstance(target, list):
                return
            while len(target) < len(source):
                target.append({})
            for index, item in enumerate(source):
                if len(parts) == 1:
                    target[index] = self._sanitize_visible(item)
                elif isinstance(item, dict):
                    self._copy_path(item, target[index], parts[1:])
            return
        if not isinstance(source, dict) or part not in source or not isinstance(target, dict):
            return
        if len(parts) == 1:
            target[part] = self._sanitize_visible(source[part])
            return
        next_part = parts[1]
        if next_part == "*":
            target.setdefault(part, [])
        else:
            target.setdefault(part, {})
        self._copy_path(source[part], target[part], parts[1:])

    def _sanitize_visible(self, value: Any) -> Any:
        return ObservationNormalizer()._safe(value)


class ContextBuilder:
    def __init__(self, registry: ToolRegistry, projector: AllowlistProjector) -> None:
        self.registry = registry
        self.projector = projector

    def planning_context(self, state: AssessmentCaseState) -> dict[str, object]:
        return {
            "case_id": state.case_id,
            "input_type": state.input_type.value,
            "modalities": state.modalities,
            "completed_workflows": state.completed_workflows,
            "failed_workflows": state.failed_workflows,
            "model_readiness": state.model_readiness.value,
            "data_quality": state.data_quality,
            "step_count": state.step_count,
            "execution_status": state.execution_status.value,
        }

    def tool_selection_context(self, state: AssessmentCaseState) -> dict[str, object]:
        return {
            "state": self.planning_context(state),
            "capabilities": [
                {
                    "name": manifest.name,
                    "supported_goals": [goal.value for goal in manifest.supported_goals],
                    "limitations": manifest.limitations,
                    "real_or_fake": manifest.real_or_fake.value,
                }
                for manifest in self.registry.manifests()
            ],
        }

    def report_context(
        self,
        state: AssessmentCaseState,
        goal: RequestedGoal,
    ) -> dict[str, object]:
        visible = [
            self.projector.project(item, goal)
            for item in state.observations
            if self.registry.describe(item.tool_name).category != "report"
        ]
        used_manifests = {
            manifest.name: manifest
            for manifest in self.registry.manifests()
            if any(item.tool_name == manifest.name for item in visible)
        }
        return {
            "case_id": state.case_id,
            "user_request": re.sub(
                r"(?:/[^\s,;]+)+", "[LOCAL_PATH_REDACTED]", state.user_request
            )[:1000],
            "executed_workflows": state.completed_workflows,
            "visible_results": [item.model_dump(mode="json") for item in visible],
            "tool_limitations": [
                {
                    "tool_name": manifest.name,
                    "unsupported_claims": manifest.unsupported_claims,
                    "limitations": manifest.limitations,
                }
                for manifest in used_manifests.values()
            ],
            "missing_information": state.pending_requirements,
            "report_spec_schema": FinalAssessmentReportSpec.model_json_schema(),
            "non_diagnostic_rules": [
                "Do not diagnose or rule out ASD.",
                "Use only visible_results; absent fields are unavailable.",
                "Replay and offline outputs must retain their provenance limitations.",
            ],
        }


@dataclass
class ExecutionOutcome:
    state: AssessmentCaseState
    results: list[ToolResult]


class WorkflowExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        normalizer: ObservationNormalizer,
        trace: TraceLogger,
        max_steps: int = 16,
        stuck_limit: int = 2,
    ) -> None:
        self.registry = registry
        self.normalizer = normalizer
        self.trace = trace
        self.max_steps = max_steps
        self.stuck_limit = stuck_limit

    def execute(self, state: AssessmentCaseState, plan: PlanSpec) -> ExecutionOutcome:
        state.status = CaseStatus.RUNNING
        state.plan = plan
        results: list[ToolResult] = []
        signatures: dict[str, int] = {}
        self.trace.event("execution_started", {"plan": plan})
        for index, step in enumerate(plan.steps):
            if index >= self.max_steps:
                state.warnings.append("max_steps reached")
                break
            signature = self._signature(step.tool_name, step.arguments, state)
            signatures[signature] = signatures.get(signature, 0) + 1
            if signatures[signature] >= self.stuck_limit:
                state.warnings.append(f"stuck signature detected for {step.tool_name}")
                self.trace.event("stuck_detected", {"tool_name": step.tool_name})
                break
            if step.tool_name in state.completed_workflows:
                result = self._skipped(step.tool_name, "workflow already completed")
            elif (
                step.tool_name == "run_asd_classifier"
                and state.model_readiness is ModelReadiness.BLOCKED
            ):
                result = self._skipped(step.tool_name, "model readiness blocked")
            else:
                result = self._execute_with_retry(step.tool_name, state, step.arguments, step.max_retries)
            results.append(result)
            observation = self.normalizer.normalize(result)
            state.observations.append(observation)
            self._apply_state_transition(state, result)
            self.trace.event("tool_finished", {"step": step, "result": result})
            if result.status in (ToolStatus.SUCCESS, ToolStatus.SKIPPED):
                state.completed_workflows.append(step.tool_name)
            if result.status in (ToolStatus.FAILURE, ToolStatus.BLOCKED) and step.on_failure == "stop":
                state.status = CaseStatus.BLOCKED
                break
        if state.status is CaseStatus.RUNNING:
            state.status = CaseStatus.COMPLETED
        self.trace.event("execution_finished", {"status": state.status, "steps": len(results)})
        return ExecutionOutcome(state=state, results=results)

    def _execute_with_retry(
        self,
        tool_name: str,
        state: AssessmentCaseState,
        arguments: dict[str, object],
        max_retries: int,
    ) -> ToolResult:
        result = self.registry.execute(tool_name, state, arguments)
        retries = 0
        while result.status is ToolStatus.FAILURE and retries < max_retries:
            retries += 1
            self.trace.event("tool_retry", {"tool_name": tool_name, "retry": retries})
            result = self.registry.execute(tool_name, state, arguments)
        return result

    def _skipped(self, tool_name: str, reason: str) -> ToolResult:
        manifest = self.registry.describe(tool_name)
        return ToolResult(
            tool_name=tool_name,
            manifest_version=manifest.version,
            status=ToolStatus.SKIPPED,
            warnings=[reason],
            limitations=manifest.limitations,
            implementation_kind=manifest.real_or_fake,
        )

    @staticmethod
    def _signature(
        tool_name: str,
        arguments: dict[str, object],
        state: AssessmentCaseState,
    ) -> str:
        state_view = {
            "quality": state.data_quality,
            "readiness": state.model_readiness.value,
            "completed": sorted(state.completed_workflows),
        }
        raw = json.dumps([tool_name, arguments, state_view], sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def _apply_state_transition(state: AssessmentCaseState, result: ToolResult) -> None:
        if result.tool_name == "check_data_quality" and result.status is ToolStatus.SUCCESS:
            state.data_quality = str(result.data.get("quality_status", "unknown"))
        if result.tool_name == "extract_eeg_features" and result.status is ToolStatus.SUCCESS:
            features = result.data.get("features", {})
            if isinstance(features, dict):
                state.feature_table.update(features)
        if result.tool_name == "check_model_readiness":
            readiness = str(result.data.get("readiness", "blocked"))
            state.model_readiness = ModelReadiness(readiness)
        if result.tool_name == "run_asd_classifier" and result.status is ToolStatus.SUCCESS:
            state.existing_model_result = result.data


class StateReducer:
    """Apply typed StateUpdate values; tools never receive write authority."""

    @staticmethod
    def apply(state: AssessmentCaseState, update: StateUpdate, observation: Observation) -> None:
        state.observations.append(observation)
        state.completed_workflows = StateReducer._merge(
            state.completed_workflows, update.completed_workflows_add
        )
        state.failed_workflows = StateReducer._merge(
            state.failed_workflows, update.failed_workflows_add
        )
        state.pending_requirements = StateReducer._merge(
            state.pending_requirements, update.pending_requirements_add
        )
        removals = set(update.pending_requirements_remove)
        state.pending_requirements = [
            item for item in state.pending_requirements if item not in removals
        ]
        state.feature_table.update(update.feature_updates)
        if update.model_readiness is not None:
            state.model_readiness = update.model_readiness
        if update.data_quality is not None:
            state.data_quality = update.data_quality
        if update.existing_model_result is not None:
            state.existing_model_result = update.existing_model_result
        if update.rag_query is not None:
            state.rag_query = update.rag_query
        if update.report_spec is not None:
            state.report_spec = update.report_spec
        state.warnings = StateReducer._merge(state.warnings, update.warnings_add)
        state.limitations = StateReducer._merge(state.limitations, update.limitations_add)
        state.current_action = update.current_action
        if update.execution_status is not None:
            state.execution_status = update.execution_status
        state.step_count += update.increment_step_count

    @staticmethod
    def _merge(current: list[str], additions: list[str]) -> list[str]:
        return list(dict.fromkeys([*current, *additions]))


@dataclass
class IncrementalExecutionOutcome:
    state: AssessmentCaseState
    decisions: list[ActionDecision]
    results: list[ToolResult]


class IncrementalWorkflowExecutor:
    """Observe → decide one action → execute → update → decide again."""

    def __init__(
        self,
        registry: ToolRegistry,
        normalizer: ObservationNormalizer,
        trace: TraceLogger,
        max_steps: int = 16,
        stuck_limit: int = 2,
    ) -> None:
        self.registry = registry
        self.normalizer = normalizer
        self.trace = trace
        self.max_steps = max_steps
        self.stuck_limit = stuck_limit

    def execute(self, state: AssessmentCaseState, planner: Any) -> IncrementalExecutionOutcome:
        state.execution_status = CaseStatus.RUNNING
        decisions: list[ActionDecision] = []
        results: list[ToolResult] = []
        signatures: dict[str, int] = {}
        self.trace.event("incremental_execution_started", {"case_id": state.case_id})

        while state.step_count < self.max_steps:
            decision: ActionDecision = planner.decide(state)
            decisions.append(decision)
            state.action_history.append(decision)
            state.current_action = decision.tool_name
            self.trace.event("next_action_decided", {"decision": decision})
            if decision.tool_name is None:
                if state.execution_status is CaseStatus.RUNNING:
                    state.execution_status = CaseStatus.COMPLETED
                break

            signature = self._signature(decision, state)
            signatures[signature] = signatures.get(signature, 0) + 1
            if signatures[signature] >= self.stuck_limit:
                state.warnings.append(f"stuck signature detected for {decision.tool_name}")
                state.execution_status = CaseStatus.BLOCKED
                self.trace.event("stuck_detected", {"tool_name": decision.tool_name})
                break

            outcome = self._execute_with_retry(decision, state)
            result = outcome.result
            results.append(result)
            observation = self.normalizer.normalize(result)
            StateReducer.apply(
                state,
                outcome.state_update.model_copy(update={"current_action": decision.tool_name}),
                observation,
            )
            self.trace.event(
                "tool_finished",
                {"decision": decision, "result": result, "state_update": outcome.state_update},
            )
            if decision.terminal:
                break
        else:
            state.warnings.append("max_steps reached")
            state.execution_status = CaseStatus.BLOCKED

        if state.execution_status is CaseStatus.RUNNING:
            state.execution_status = CaseStatus.COMPLETED
        self.trace.event(
            "incremental_execution_finished",
            {"status": state.execution_status, "steps": state.step_count},
        )
        return IncrementalExecutionOutcome(state=state, decisions=decisions, results=results)

    def _execute_with_retry(
        self,
        decision: ActionDecision,
        state: AssessmentCaseState,
    ) -> ToolExecutionOutcome:
        assert decision.tool_name is not None
        manifest = self.registry.describe(decision.tool_name)
        retries = 0
        outcome = self.registry.execute_outcome(decision.tool_name, state, decision.arguments)
        while outcome.result.status is ToolStatus.FAILURE and retries < manifest.retry_limit:
            retries += 1
            self.trace.event(
                "tool_retry", {"tool_name": decision.tool_name, "retry": retries}
            )
            outcome = self.registry.execute_outcome(decision.tool_name, state, decision.arguments)
        return outcome

    @staticmethod
    def _signature(decision: ActionDecision, state: AssessmentCaseState) -> str:
        state_view = {
            "quality": state.data_quality,
            "readiness": state.model_readiness.value,
            "completed": sorted(state.completed_workflows),
            "failed": sorted(state.failed_workflows),
            "observation_statuses": [
                (item.tool_name, item.status.value) for item in state.observations
            ],
        }
        raw = json.dumps(
            [decision.tool_name, decision.arguments, state_view],
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(raw.encode()).hexdigest()
