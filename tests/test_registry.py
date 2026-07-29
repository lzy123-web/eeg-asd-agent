import pytest

from eeg_asd_agent.schemas import (
    AssessmentCaseState,
    ImplementationKind,
    InputKind,
    ToolCapabilityManifest,
    ToolResult,
    ToolStatus,
)
from eeg_asd_agent.tools import DuplicateToolError, FunctionTool, ToolRegistry, UnknownToolError


def _tool(name: str = "safe_tool") -> FunctionTool:
    manifest = ToolCapabilityManifest(
        name=name,
        category="test",
        implementation_ref="tests",
        real_or_fake=ImplementationKind.REAL,
    )

    def run(state: AssessmentCaseState, arguments: dict[str, object]) -> ToolResult:
        return ToolResult(
            tool_name=name,
            manifest_version="1.0",
            status=ToolStatus.SUCCESS,
            data={"ok": True},
            implementation_kind=ImplementationKind.REAL,
        )

    return FunctionTool(manifest, run)


def test_registry_rejects_duplicate_registration() -> None:
    registry = ToolRegistry()
    registry.register(_tool())
    with pytest.raises(DuplicateToolError):
        registry.register(_tool())


def test_registry_rejects_unknown_tool() -> None:
    registry = ToolRegistry()
    state = AssessmentCaseState(case_id="c", input_kind=InputKind.INCOMPLETE)
    with pytest.raises(UnknownToolError):
        registry.execute("not_registered", state)


def test_registry_executes_typed_tool() -> None:
    registry = ToolRegistry()
    registry.register(_tool())
    state = AssessmentCaseState(case_id="c", input_kind=InputKind.INCOMPLETE)
    result = registry.execute("safe_tool", state)
    assert result.status is ToolStatus.SUCCESS
    assert result.data == {"ok": True}


def test_registry_turns_tool_crash_into_failure() -> None:
    manifest = ToolCapabilityManifest(
        name="crash_tool",
        category="test",
        implementation_ref="tests",
        real_or_fake=ImplementationKind.REAL,
    )

    def crash(state: AssessmentCaseState, arguments: dict[str, object]) -> ToolResult:
        raise RuntimeError("boom")

    registry = ToolRegistry()
    registry.register(FunctionTool(manifest, crash))
    result = registry.execute(
        "crash_tool",
        AssessmentCaseState(case_id="c", input_kind=InputKind.INCOMPLETE),
    )
    assert result.status is ToolStatus.FAILURE
    assert "RuntimeError" in result.errors[0]

