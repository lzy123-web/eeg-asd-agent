from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from eeg_asd_agent.schemas import AssessmentCaseState, ToolCapabilityManifest, ToolResult


class Tool(Protocol):
    manifest: ToolCapabilityManifest

    def run(self, state: AssessmentCaseState, arguments: dict[str, object]) -> ToolResult: ...


class FunctionTool:
    def __init__(
        self,
        manifest: ToolCapabilityManifest,
        function: Callable[[AssessmentCaseState, dict[str, object]], ToolResult],
    ) -> None:
        self.manifest = manifest
        self._function = function

    def run(self, state: AssessmentCaseState, arguments: dict[str, object]) -> ToolResult:
        return self._function(state, arguments)

