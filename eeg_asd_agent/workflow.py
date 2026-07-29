from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Protocol

from eeg_asd_agent.execution import (
    AllowlistProjector,
    ContextBuilder,
    IncrementalWorkflowExecutor,
    ObservationNormalizer,
)
from eeg_asd_agent.intent import RuleBasedIntentParser
from eeg_asd_agent.planning import DeterministicPlanner, LLMPlanner, NextActionPlanner
from eeg_asd_agent.providers import OfflineProvider, OfflineReportGenerator
from eeg_asd_agent.reporting import HTMLRenderer
from eeg_asd_agent.schemas import ActionDecision, AssessmentCaseState, PlanSpec, RequestedGoal
from eeg_asd_agent.tools.builtin import build_default_registry
from eeg_asd_agent.tools.core_workflow import build_core_registry
from eeg_asd_agent.tracing import ArtifactStore, TraceLogger, render_trace_html


class WorkflowProvider(Protocol):
    name: str

    def generate_json(self, purpose: str, payload: dict[str, object]) -> str: ...


def build_plan(
    state: AssessmentCaseState,
    request: str,
    *,
    planner_kind: str = "deterministic",
    provider: WorkflowProvider | None = None,
) -> PlanSpec:
    registry = build_default_registry()
    goal = RuleBasedIntentParser().parse(request)
    state.user_request = request
    state.requested_goal = goal
    planner: DeterministicPlanner
    active_provider = provider or OfflineProvider()
    if planner_kind == "llm":
        planner = LLMPlanner(registry, active_provider, include_manifests=True)
    elif planner_kind == "llm-no-manifest":
        planner = LLMPlanner(registry, active_provider, include_manifests=False)
    else:
        planner = DeterministicPlanner(registry)
    return planner.plan(state, goal)


def run_workflow(
    state: AssessmentCaseState,
    request: str,
    output_dir: Path,
    *,
    planner_kind: str = "deterministic",
    provider: WorkflowProvider | None = None,
) -> dict[str, Path]:
    run_started = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in ("observations.jsonl", "trace.jsonl"):
        stale = output_dir / stale_name
        if stale.exists():
            stale.unlink()
    tool_result_dir = output_dir / "tool_results"
    if tool_result_dir.exists():
        for stale in tool_result_dir.glob("*.json"):
            stale.unlink()
    store = ArtifactStore(output_dir)
    trace = TraceLogger(output_dir / "trace.jsonl", state.trace_id)
    active_provider = provider or OfflineReportGenerator()
    registry = build_core_registry(report_provider=active_provider)
    goal = RuleBasedIntentParser().parse(request)
    state.user_request = request
    state.requested_goal = goal
    planner = NextActionPlanner(registry, use_manifests=planner_kind != "deterministic-no-manifest")
    normalizer = ObservationNormalizer()
    outcome = IncrementalWorkflowExecutor(registry, normalizer, trace).execute(state, planner)
    store.write_json(
        "plan.json",
        {
            "planner_kind": "incremental_state_driven",
            "case_id": state.case_id,
            "decisions": outcome.decisions,
        },
    )
    store.write_text("plan.mmd", decisions_to_mermaid(outcome.decisions))
    for index, result in enumerate(outcome.results, start=1):
        store.write_json(f"tool_results/{index:02d}_{result.tool_name}.json", result)
    for observation in state.observations:
        store.append_jsonl("observations.jsonl", observation)

    projector = AllowlistProjector(registry)
    context = ContextBuilder(registry, projector).report_context(state, goal)
    store.write_json("llm_payload.json", context)
    render_result = next(
        (item for item in outcome.results if item.tool_name == "render_report"),
        None,
    )
    if state.report_spec is not None:
        store.write_json("report_spec.json", state.report_spec)
        report_md = (
            str(render_result.data["content"])
            if render_result is not None and "content" in render_result.data
            else ""
        )
        store.write_text("report.md", report_md)
        store.write_text("report.html", HTMLRenderer().render(state.report_spec))
    store.write_json("case_state.json", state)
    usage = getattr(active_provider, "last_usage", {})
    store.write_json(
        "run_stats.json",
        {
            "total_duration_ms": round((perf_counter() - run_started) * 1000, 3),
            "tool_duration_ms": {
                result.tool_name: round(result.duration_ms, 3) for result in outcome.results
            },
            "provider": active_provider.name,
            "usage": usage,
            "evaluation_mode": (
                "offline_structural_validation"
                if active_provider.name.startswith("offline")
                else "provider"
            ),
        },
    )
    trace.event(
        "workflow_artifacts_complete",
        {
            "schema_valid": state.report_spec is not None,
            "rendered": render_result is not None,
        },
    )
    store.write_text("trace.html", render_trace_html(output_dir / "trace.jsonl"))
    artifacts = {
        "plan": output_dir / "plan.json",
        "case_state": output_dir / "case_state.json",
        "observations": output_dir / "observations.jsonl",
        "payload": output_dir / "llm_payload.json",
        "trace": output_dir / "trace.jsonl",
        "trace_html": output_dir / "trace.html",
        "plan_mermaid": output_dir / "plan.mmd",
        "run_stats": output_dir / "run_stats.json",
    }
    if state.report_spec is not None:
        artifacts.update(
            {
                "report_spec": output_dir / "report_spec.json",
                "report_md": output_dir / "report.md",
                "report_html": output_dir / "report.html",
            }
        )
    return artifacts


def default_goal(request: str) -> RequestedGoal:
    return RuleBasedIntentParser().parse(request)


def plan_to_mermaid(plan: PlanSpec) -> str:
    lines = ["flowchart TD"]
    for step in plan.steps:
        label = step.tool_name.replace('"', "'")
        lines.append(f'    {step.step_id}["{label}"]')
        for dependency in step.depends_on:
            lines.append(f"    {dependency} --> {step.step_id}")
    if not plan.steps:
        lines.append('    empty["No executable steps"]')
    return "\n".join(lines) + "\n"


def decisions_to_mermaid(decisions: Sequence[ActionDecision]) -> str:
    lines = ["flowchart TD"]
    for index, raw in enumerate(decisions, start=1):
        decision = raw
        tool_name = getattr(decision, "tool_name", None) or "STOP"
        lines.append(f'    s{index:02d}["{str(tool_name).replace(chr(34), chr(39))}"]')
        if index > 1:
            lines.append(f"    s{index - 1:02d} --> s{index:02d}")
    return "\n".join(lines) + "\n"
