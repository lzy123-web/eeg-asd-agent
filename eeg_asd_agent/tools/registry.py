from __future__ import annotations

from time import perf_counter

from eeg_asd_agent.schemas import (
    AssessmentCaseState,
    ImplementationKind,
    ToolCapabilityManifest,
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
        return result.model_copy(update={"duration_ms": (perf_counter() - started) * 1000})

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

