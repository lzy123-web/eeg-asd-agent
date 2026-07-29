from __future__ import annotations

import json
from typing import Protocol

from eeg_asd_agent.schemas import (
    AssessmentCaseState,
    InputKind,
    PlanSpec,
    PlanStep,
    RequestedGoal,
)
from eeg_asd_agent.tools import ToolRegistry


class PlanValidationError(ValueError):
    pass


class JSONProvider(Protocol):
    def generate_json(self, purpose: str, payload: dict[str, object]) -> str: ...


class DeterministicPlanner:
    def __init__(self, registry: ToolRegistry, max_steps: int = 16) -> None:
        self.registry = registry
        self.max_steps = max_steps

    def plan(self, state: AssessmentCaseState, goal: RequestedGoal) -> PlanSpec:
        names: list[str] = ["inspect_case"]
        rationale = [f"input={state.input_kind.value}", f"goal={goal.value}"]
        if goal is RequestedGoal.UNSUPPORTED:
            names.append("request_human_review")
        elif state.input_kind is InputKind.RAW_EEG:
            names.extend(["inspect_eeg", "check_data_quality"])
            if goal not in (RequestedGoal.QUALITY_CHECK,):
                names.extend(["preprocess_eeg", "extract_eeg_features"])
            if goal in (RequestedGoal.FULL_ASSESSMENT, RequestedGoal.REPORT_ONLY):
                names.extend(self._assessment_tail())
        elif state.input_kind is InputKind.FEATURE_TABLE:
            names.append("check_data_quality")
            if goal is RequestedGoal.FEATURE_EXTRACTION:
                rationale.append("feature table already exists; extraction skipped")
            elif goal in (RequestedGoal.FULL_ASSESSMENT, RequestedGoal.REPORT_ONLY):
                names.extend(self._assessment_tail())
            elif goal in (RequestedGoal.KNOWLEDGE_SEARCH, RequestedGoal.INTERVENTION):
                names.extend(self._knowledge_tail(goal))
        elif state.input_kind is InputKind.MODEL_RESULT:
            if goal in (RequestedGoal.FULL_ASSESSMENT, RequestedGoal.REPORT_ONLY):
                names.extend(
                    ["generate_medical_analysis", "predict_severity", "retrieve_asd_knowledge"]
                )
            else:
                names.extend(self._knowledge_tail(goal))
        elif state.input_kind is InputKind.MEDICAL_REPORT:
            names.extend(self._knowledge_tail(goal))
        elif state.input_kind is InputKind.KNOWLEDGE_ONLY:
            names.extend(self._knowledge_tail(goal))
        else:
            names.append("request_human_review")

        if goal is RequestedGoal.INTERVENTION and "retrieve_intervention_knowledge" not in names:
            names.extend(["retrieve_intervention_knowledge", "generate_intervention_candidates"])
        names = [name for name in names if name in self.registry.names()]
        names = [name for name in names if name not in state.completed_workflows]
        if len(names) > self.max_steps:
            names = names[: self.max_steps]
            rationale.append(f"truncated to max_steps={self.max_steps}")
        steps: list[PlanStep] = []
        previous: str | None = None
        for index, name in enumerate(names, start=1):
            step_id = f"s{index:02d}"
            steps.append(
                PlanStep(
                    step_id=step_id,
                    tool_name=name,
                    depends_on=[previous] if previous else [],
                    on_failure=self._failure_policy(name),
                    max_retries=self.registry.describe(name).retry_limit,
                )
            )
            previous = step_id
        spec = PlanSpec(
            case_id=state.case_id,
            goal=goal,
            planner_kind="deterministic",
            steps=steps,
            stop_conditions=["max_steps", "stuck_signature", "blocked_critical_tool"],
            rationale=rationale,
        )
        self.validate(spec)
        return spec

    def validate(self, plan: PlanSpec) -> None:
        unknown = {step.tool_name for step in plan.steps} - self.registry.names()
        if unknown:
            raise PlanValidationError(f"unregistered tools: {sorted(unknown)}")
        if len(plan.steps) > self.max_steps:
            raise PlanValidationError("plan exceeds max_steps")

    @staticmethod
    def _assessment_tail() -> list[str]:
        return [
            "check_model_readiness",
            "run_asd_classifier",
            "generate_medical_analysis",
            "predict_severity",
            "retrieve_asd_knowledge",
        ]

    @staticmethod
    def _knowledge_tail(goal: RequestedGoal) -> list[str]:
        if goal is RequestedGoal.INTERVENTION:
            return ["retrieve_intervention_knowledge", "generate_intervention_candidates"]
        return ["retrieve_asd_knowledge"]

    @staticmethod
    def _failure_policy(tool_name: str) -> str:
        if tool_name in {"inspect_case", "inspect_eeg", "check_data_quality"}:
            return "stop"
        if tool_name == "check_model_readiness":
            return "skip_model_and_continue_limited"
        return "continue_limited"


class FixedPipelinePlanner(DeterministicPlanner):
    def plan(self, state: AssessmentCaseState, goal: RequestedGoal) -> PlanSpec:
        names = [
            "inspect_case",
            "inspect_eeg",
            "check_data_quality",
            "preprocess_eeg",
            "extract_eeg_features",
            "check_model_readiness",
            "run_asd_classifier",
            "generate_medical_analysis",
            "predict_severity",
            "retrieve_asd_knowledge",
        ]
        names = [name for name in names if name in self.registry.names()]
        steps = [
            PlanStep(
                step_id=f"s{index:02d}",
                tool_name=name,
                depends_on=[f"s{index - 1:02d}"] if index > 1 else [],
                on_failure="continue",
            )
            for index, name in enumerate(names, start=1)
        ]
        return PlanSpec(
            case_id=state.case_id,
            goal=goal,
            planner_kind="fixed_pipeline",
            steps=steps,
            stop_conditions=["max_steps"],
        )


class LLMPlanner(DeterministicPlanner):
    def __init__(
        self,
        registry: ToolRegistry,
        provider: JSONProvider,
        include_manifests: bool = True,
        max_steps: int = 16,
    ) -> None:
        super().__init__(registry, max_steps)
        self.provider = provider
        self.include_manifests = include_manifests

    def plan(self, state: AssessmentCaseState, goal: RequestedGoal) -> PlanSpec:
        payload: dict[str, object] = {
            "case": {
                "case_id": state.case_id,
                "input_kind": state.input_kind.value,
                "completed_workflows": state.completed_workflows,
                "model_readiness": state.model_readiness.value,
            },
            "goal": goal.value,
            "allowed_tool_names": sorted(self.registry.names()),
        }
        if self.include_manifests:
            payload["manifests"] = [
                manifest.model_dump(mode="json") for manifest in self.registry.manifests()
            ]
        try:
            raw = self.provider.generate_json("planning", payload)
            plan = PlanSpec.model_validate(json.loads(raw))
            self.validate(plan)
            return plan.model_copy(update={"planner_kind": "llm_manifest" if self.include_manifests else "llm"})
        except Exception:
            fallback = super().plan(state, goal)
            return fallback.model_copy(
                update={
                    "planner_kind": "deterministic_fallback",
                    "rationale": fallback.rationale + ["LLM plan invalid; deterministic fallback used"],
                }
            )
