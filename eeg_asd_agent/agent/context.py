from __future__ import annotations

import re
from typing import Any

from eeg_asd_agent.agent.memory import SQLiteAgentMemory
from eeg_asd_agent.execution import AllowlistProjector
from eeg_asd_agent.schemas import AssessmentCaseState, RequestedGoal
from eeg_asd_agent.tools.registry import ToolRegistry


class AgentContextBuilder:
    """Build the only payload an Agent controller is allowed to see."""

    def __init__(
        self,
        registry: ToolRegistry,
        projector: AllowlistProjector,
        memory: SQLiteAgentMemory,
    ) -> None:
        self.registry = registry
        self.projector = projector
        self.memory = memory

    def build(
        self,
        state: AssessmentCaseState,
        allowed_actions: list[str],
        *,
        max_steps: int,
    ) -> dict[str, object]:
        projection_goal = state.requested_goal or RequestedGoal.FULL_ASSESSMENT
        visible_observations = [
            self.projector.project(item, projection_goal).model_dump(mode="json")
            for item in state.observations[-8:]
        ]
        return {
            "agent_contract": {
                "domain": "EEG-ASD research assessment",
                "control_mode": "llm_agentic_next_action",
                "loop": "plan -> act -> observe -> check -> replan",
                "harness_role": "validate and execute; never choose workflow order",
            },
            "user_request": self._safe_request(state.user_request),
            "parsed_goal_hint": (
                state.requested_goal.value if state.requested_goal is not None else None
            ),
            "case_state": {
                "case_id": state.case_id,
                "input_type_hint": state.input_type.value,
                "asset_inventory": {
                    "raw_eeg_count": len(state.raw_files),
                    "feature_table_present": bool(state.feature_table),
                    "feature_count": len(state.feature_table),
                    "feature_names": sorted(state.feature_table)[:64],
                    "imported_model_result_present": state.existing_model_result is not None,
                    "model_result_history_count": len(state.model_result_history),
                    "existing_report_present": state.existing_report is not None,
                    "modalities": state.modalities,
                },
                "data_quality": state.data_quality,
                "model_readiness": state.model_readiness.value,
                "completed_actions": state.completed_workflows,
                "failed_actions": state.failed_workflows,
                "pending_requirements": state.pending_requirements,
                "execution_status": state.execution_status.value,
                "step_count": state.step_count,
            },
            "current_plan": (
                state.agent_plan.model_dump(mode="json")
                if state.agent_plan is not None
                else None
            ),
            "previous_observations": visible_observations,
            "action_history": [
                {
                    "tool_name": item.tool_name,
                    "reason": item.reason[:300],
                    "terminal": item.terminal,
                    "based_on_observations": item.based_on_observations,
                }
                for item in state.action_history[-8:]
            ],
            "memory": self.memory.snapshot_for_llm(state.case_id),
            "capability_catalog": [
                self._manifest_payload(manifest.name)
                for manifest in self.registry.manifests()
            ],
            "allowed_next_actions": allowed_actions,
            "remaining_step_budget": max(0, max_steps - state.step_count),
            "rules": [
                "Revise or confirm the high-level plan on every turn.",
                "Choose at most one tool from allowed_next_actions.",
                "Existing assets are optional evidence, not automatic skip instructions.",
                "Decide whether to reuse, inspect, compare, recompute, or ignore assets from the user goal.",
                "Tool order is an Agent decision except for hard manifest preconditions.",
                "RAG may be used before, during, or after signal/model analysis when relevant.",
                "Never expose raw EEG, feature values, local paths, secrets, or internal ToolResult objects.",
                "Finish only when the user goal is satisfied by executed observations.",
                "Do not make a clinical ASD diagnosis or treatment prescription.",
            ],
        }

    def _manifest_payload(self, name: str) -> dict[str, Any]:
        manifest = self.registry.describe(name)
        return {
            "name": manifest.name,
            "version": manifest.version,
            "category": manifest.category,
            "description": manifest.description,
            "use_when": manifest.use_when,
            "avoid_when": manifest.avoid_when,
            "required_inputs": manifest.required_inputs,
            "optional_inputs": manifest.optional_inputs,
            "hard_preconditions": manifest.preconditions,
            "produced_outputs": manifest.produced_outputs,
            "limitations": manifest.limitations,
            "unsupported_claims": manifest.unsupported_claims,
            "implementation_kind": manifest.real_or_fake.value,
            "enabled": manifest.enabled,
            "repeatable": manifest.repeatable,
            "idempotent": manifest.idempotent,
        }

    @staticmethod
    def _safe_request(value: str) -> str:
        redacted = re.sub(r"(?:/[^\s,;]+)+", "[LOCAL_PATH_REDACTED]", value)
        return redacted[:2000]
