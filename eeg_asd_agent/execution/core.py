from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from eeg_asd_agent.schemas import (
    AssessmentCaseState,
    CaseStatus,
    LLMVisibleToolResult,
    ModelReadiness,
    Observation,
    PlanSpec,
    RequestedGoal,
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
        visible = {
            key: value
            for key, value in observation.facts.items()
            if key in allowed and not _DENIED_KEY.search(key)
        }
        return LLMVisibleToolResult(
            tool_name=observation.tool_name,
            status=observation.status,
            visible_facts=visible,
            limitations=list(dict.fromkeys(manifest.limitations + observation.limitations)),
            provenance_labels=observation.provenance,
            implementation_kind=observation.implementation_kind,
        )


class ContextBuilder:
    def __init__(self, registry: ToolRegistry, projector: AllowlistProjector) -> None:
        self.registry = registry
        self.projector = projector

    def planning_context(self, state: AssessmentCaseState) -> dict[str, object]:
        return {
            "case_id": state.case_id,
            "input_kind": state.input_kind.value,
            "modalities": state.modalities,
            "completed_workflows": state.completed_workflows,
            "model_readiness": state.model_readiness.value,
            "data_quality": state.data_quality,
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
        visible = [self.projector.project(item, goal) for item in state.observations]
        return {
            "case_id": state.case_id,
            "goal": goal.value,
            "case_state": self.planning_context(state),
            "visible_results": [item.model_dump(mode="json") for item in visible],
            "capability_context": [
                {
                    "tool_name": manifest.name,
                    "unsupported_claims": manifest.unsupported_claims,
                    "limitations": manifest.limitations,
                }
                for manifest in self.registry.manifests()
                if any(item.tool_name == manifest.name for item in visible)
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

