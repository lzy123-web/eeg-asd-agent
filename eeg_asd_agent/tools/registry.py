from __future__ import annotations

from time import perf_counter

from eeg_asd_agent.schemas import (
    AssessmentCaseState,
    ImplementationKind,
    StateUpdate,
    ToolCapabilityManifest,
    ToolExecutionOutcome,
    ToolResult,
    ToolStatus,
)
from eeg_asd_agent.tools.base import Tool


class DuplicateToolError(ValueError):
    pass


class UnknownToolError(KeyError):
    pass


class ToolPreconditionError(ValueError):
    pass


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.manifest.name in self._tools:
            raise DuplicateToolError(tool.manifest.name)
        self._tools[tool.manifest.name] = tool

    def manifests(self) -> list[ToolCapabilityManifest]:
        return [self._tools[name].manifest for name in sorted(self._tools)]

    def describe(self, name: str) -> ToolCapabilityManifest:
        return self._get(name).manifest

    def names(self) -> set[str]:
        return set(self._tools)

    def preconditions_met(
        self,
        name: str,
        state: AssessmentCaseState,
    ) -> tuple[bool, list[str]]:
        """Evaluate the small, closed precondition language used by core manifests."""
        manifest = self.describe(name)
        unmet = [
            field
            for field in manifest.required_state_fields
            if not self._state_has_value(state, field)
        ]
        checks = {
            "feature_table_present": bool(state.feature_table),
            "model_result_absent": state.existing_model_result is None,
            "model_result_present": state.existing_model_result is not None,
            "model_readiness_not_blocked": state.model_readiness.value != "blocked",
            "model_checked": state.model_readiness.value != "unknown",
            "report_spec_present": state.report_spec is not None,
            "request_supported": state.requested_goal is not None
            and state.requested_goal.value != "unsupported",
        }
        unmet.extend(condition for condition in manifest.preconditions if not checks.get(condition, False))
        return not unmet, unmet

    def execute(
        self,
        name: str,
        state: AssessmentCaseState,
        arguments: dict[str, object] | None = None,
    ) -> ToolResult:
        tool = self._get(name)
        args = arguments or {}
        missing = set(tool.manifest.required_inputs) - set(args)
        if missing:
            raise ToolPreconditionError(f"missing required arguments: {sorted(missing)}")
        absent_state = [
            field
            for field in tool.manifest.required_state_fields
            if not self._state_has_value(state, field)
        ]
        if absent_state:
            return ToolResult(
                tool_name=name,
                manifest_version=tool.manifest.version,
                status=ToolStatus.BLOCKED,
                errors=[f"missing state fields: {', '.join(absent_state)}"],
                limitations=tool.manifest.limitations,
                implementation_kind=tool.manifest.real_or_fake,
            )
        started = perf_counter()
        try:
            result = tool.run(state, args)
        except Exception as exc:  # registry boundary converts tool crashes into typed results
            return ToolResult(
                tool_name=name,
                manifest_version=tool.manifest.version,
                status=ToolStatus.FAILURE,
                duration_ms=(perf_counter() - started) * 1000,
                errors=[f"{type(exc).__name__}: {exc}"],
                limitations=tool.manifest.limitations,
                implementation_kind=tool.manifest.real_or_fake,
            )
        elapsed = (perf_counter() - started) * 1000
        if isinstance(result, ToolExecutionOutcome):
            return result.result.model_copy(update={"duration_ms": elapsed})
        return result.model_copy(update={"duration_ms": elapsed})

    def execute_outcome(
        self,
        name: str,
        state: AssessmentCaseState,
        arguments: dict[str, object] | None = None,
    ) -> ToolExecutionOutcome:
        """Execute a tool and require an explicit state update for incremental runs."""
        tool = self._get(name)
        args = arguments or {}
        missing = set(tool.manifest.required_inputs) - set(args)
        if missing:
            raise ToolPreconditionError(f"missing required arguments: {sorted(missing)}")
        absent_state = [
            field
            for field in tool.manifest.required_state_fields
            if not self._state_has_value(state, field)
        ]
        if absent_state:
            result = ToolResult(
                tool_name=name,
                manifest_version=tool.manifest.version,
                status=ToolStatus.BLOCKED,
                errors=[f"missing state fields: {', '.join(absent_state)}"],
                limitations=tool.manifest.limitations,
                implementation_kind=tool.manifest.real_or_fake,
            )
            return ToolExecutionOutcome(
                result=result,
                state_update=StateUpdate(
                    failed_workflows_add=[name],
                    pending_requirements_add=absent_state,
                    limitations_add=result.limitations,
                ),
            )
        started = perf_counter()
        try:
            raw = tool.run(state, args)
        except Exception as exc:
            result = ToolResult(
                tool_name=name,
                manifest_version=tool.manifest.version,
                status=ToolStatus.FAILURE,
                duration_ms=(perf_counter() - started) * 1000,
                errors=[f"{type(exc).__name__}: {exc}"],
                limitations=tool.manifest.limitations,
                implementation_kind=tool.manifest.real_or_fake,
            )
            return ToolExecutionOutcome(
                result=result,
                state_update=StateUpdate(
                    failed_workflows_add=[name],
                    warnings_add=result.errors,
                    limitations_add=result.limitations,
                ),
            )
        elapsed = (perf_counter() - started) * 1000
        if not isinstance(raw, ToolExecutionOutcome):
            raise TypeError(
                f"incremental tool {name!r} returned ToolResult without explicit StateUpdate"
            )
        return raw.model_copy(
            update={"result": raw.result.model_copy(update={"duration_ms": elapsed})}
        )

    def _get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise UnknownToolError(name) from exc

    @staticmethod
    def _state_has_value(state: AssessmentCaseState, dotted: str) -> bool:
        value: object = state
        for part in dotted.split("."):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = getattr(value, part, None)
            if value is None:
                return False
        return value not in ("", [], {})


def unavailable_result(manifest: ToolCapabilityManifest, reason: str) -> ToolResult:
    return ToolResult(
        tool_name=manifest.name,
        manifest_version=manifest.version,
        status=ToolStatus.BLOCKED,
        errors=[reason],
        limitations=manifest.limitations,
        implementation_kind=ImplementationKind.UNAVAILABLE,
    )
