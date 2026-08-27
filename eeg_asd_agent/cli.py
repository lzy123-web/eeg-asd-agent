from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from eeg_asd_agent.agent import SQLiteAgentMemory
from eeg_asd_agent.benchmarks import BenchmarkSuite, CorePlanningBenchmark
from eeg_asd_agent.dashboard import build_dashboard
from eeg_asd_agent.integrity import audit_integrity
from eeg_asd_agent.providers import OpenAIResponsesProvider, ProviderConfigurationError
from eeg_asd_agent.schemas import AssessmentCaseState
from eeg_asd_agent.tools.builtin import build_default_registry
from eeg_asd_agent.tools.domain_catalog import build_agent_registry
from eeg_asd_agent.tracing import render_trace_html
from eeg_asd_agent.workflow import build_plan, run_workflow

app = typer.Typer(help="Capability-aware EEG-ASD Workflow Agent", no_args_is_help=True)
tools_app = typer.Typer(help="Inspect the closed tool registry", no_args_is_help=True)
profile_app = typer.Typer(help="Inspect execution profiles", no_args_is_help=True)
benchmark_app = typer.Typer(help="Run reproducible fixture benchmarks", no_args_is_help=True)
trace_app = typer.Typer(help="Inspect trace artifacts", no_args_is_help=True)
dashboard_app = typer.Typer(help="Build a static artifact dashboard", no_args_is_help=True)
app.add_typer(tools_app, name="tools")
app.add_typer(profile_app, name="profile")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(trace_app, name="trace")
app.add_typer(dashboard_app, name="dashboard")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_case(path: Path) -> AssessmentCaseState:
    return AssessmentCaseState.model_validate_json(path.read_text(encoding="utf-8"))


def _load_provider(offline: bool) -> OpenAIResponsesProvider | None:
    if offline:
        return None
    try:
        return OpenAIResponsesProvider.from_env()
    except ProviderConfigurationError as exc:
        typer.echo(f"Provider unavailable; using offline fallback: {exc}", err=True)
        return None


@tools_app.command("list")
def tools_list() -> None:
    registry = build_agent_registry()
    rows = [
        {
            "name": manifest.name,
            "category": manifest.category,
            "version": manifest.version,
            "implementation": manifest.real_or_fake.value,
            "enabled": manifest.enabled,
            "repeatable": manifest.repeatable,
        }
        for manifest in registry.manifests()
    ]
    typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))


@tools_app.command("describe")
def tools_describe(name: str) -> None:
    agent_registry = build_agent_registry()
    registry = agent_registry if name in agent_registry.names() else build_default_registry()
    manifest = registry.describe(name)
    typer.echo(manifest.model_dump_json(indent=2))


@profile_app.command("show")
def profile_show() -> None:
    profile = json.loads((PROJECT_ROOT / "profiles" / "v1.json").read_text(encoding="utf-8"))
    profile["report_schema"] = "FinalAssessmentReportSpec@1.0"
    typer.echo(json.dumps(profile, ensure_ascii=False, indent=2))


@app.command("plan")
def plan_command(
    case: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    request: Annotated[str, typer.Option()],
    output: Annotated[Path | None, typer.Option()] = None,
    planner: Annotated[str, typer.Option()] = "deterministic",
    offline: Annotated[bool, typer.Option()] = True,
) -> None:
    state = _load_case(case)
    provider = _load_provider(offline)
    active_planner = planner if offline or planner != "deterministic" else "llm"
    plan = build_plan(state, request, planner_kind=active_planner, provider=provider)
    content = plan.model_dump_json(indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
    typer.echo(content)


@app.command("run")
def run_command(
    case: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    request: Annotated[str, typer.Option()],
    output_dir: Annotated[Path, typer.Option()] = Path("artifacts/demo"),
    planner: Annotated[str, typer.Option()] = "agent",
    offline: Annotated[bool, typer.Option()] = True,
) -> None:
    state = _load_case(case)
    provider = _load_provider(offline)
    active_planner = planner if offline or planner != "deterministic" else "llm"
    artifacts = run_workflow(
        state,
        request,
        output_dir,
        planner_kind=active_planner,
        provider=provider,
    )
    typer.echo(json.dumps({key: str(value) for key, value in artifacts.items()}, ensure_ascii=False, indent=2))


@app.command("resume")
def resume_command(
    memory: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    case_id: Annotated[str, typer.Option()],
    request: Annotated[str, typer.Option()],
    output_dir: Annotated[Path, typer.Option()] = Path("artifacts/resumed"),
    offline: Annotated[bool, typer.Option()] = True,
) -> None:
    store = SQLiteAgentMemory(memory)
    state = store.load_state(case_id)
    if state is None:
        raise typer.BadParameter(f"case_id {case_id!r} is not present in Agent memory")
    # SQLite keeps the durable cross-run history. These fields describe only the
    # active request, so a resumed task receives a fresh local action window.
    state.action_history = []
    state.current_action = None
    state.latest_assistant_message = None
    provider = _load_provider(offline)
    artifacts = run_workflow(
        state,
        request,
        output_dir,
        planner_kind="agent",
        provider=provider,
        memory_path=memory,
    )
    typer.echo(
        json.dumps(
            {key: str(value) for key, value in artifacts.items()},
            ensure_ascii=False,
            indent=2,
        )
    )


def _benchmark_command(suite: str, output_dir: Path) -> None:
    result = BenchmarkSuite(PROJECT_ROOT / "benchmark_cases").run(suite, output_dir)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@benchmark_app.command("planning")
def benchmark_planning(
    output_dir: Annotated[Path, typer.Option()] = Path("artifacts/benchmarks/planning"),
    offline: Annotated[bool, typer.Option()] = True,
) -> None:
    del offline
    _benchmark_command("planning", output_dir)


@benchmark_app.command("core-planning")
def benchmark_core_planning(
    output_dir: Annotated[Path, typer.Option()] = Path(
        "artifacts/benchmarks/core-planning"
    ),
    offline: Annotated[bool, typer.Option()] = True,
) -> None:
    del offline
    result = CorePlanningBenchmark(
        PROJECT_ROOT / "benchmark_cases" / "core_planning_cases.jsonl"
    ).run(output_dir)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@benchmark_app.command("boundaries")
def benchmark_boundaries(
    output_dir: Annotated[Path, typer.Option()] = Path("artifacts/benchmarks/boundaries"),
    offline: Annotated[bool, typer.Option()] = True,
) -> None:
    del offline
    _benchmark_command("boundaries", output_dir)


@benchmark_app.command("reports")
def benchmark_reports(
    output_dir: Annotated[Path, typer.Option()] = Path("artifacts/benchmarks/reports"),
    offline: Annotated[bool, typer.Option()] = True,
) -> None:
    del offline
    _benchmark_command("reports", output_dir)


@benchmark_app.command("ablations")
def benchmark_ablations(
    output_dir: Annotated[Path, typer.Option()] = Path("artifacts/benchmarks/ablations"),
    offline: Annotated[bool, typer.Option()] = True,
) -> None:
    del offline
    _benchmark_command("ablations", output_dir)


@benchmark_app.command("all")
def benchmark_all(
    output_dir: Annotated[Path, typer.Option()] = Path("artifacts/benchmarks"),
    offline: Annotated[bool, typer.Option()] = True,
) -> None:
    del offline
    _benchmark_command("all", output_dir)


@trace_app.command("show")
def trace_show(path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)], tail: int = 20) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines[-tail:]:
        typer.echo(line)


@trace_app.command("html")
def trace_html(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()] = Path("artifacts/trace.html"),
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_trace_html(path), encoding="utf-8")
    typer.echo(str(output))


@dashboard_app.command("build")
def dashboard_build(
    artifact_root: Annotated[Path, typer.Option()] = Path("artifacts"),
    output: Annotated[Path, typer.Option()] = Path("artifacts/dashboard.html"),
) -> None:
    typer.echo(str(build_dashboard(artifact_root, output)))


@app.command("audit-integrity")
def audit_integrity_command(
    source_root: Annotated[
        Path,
        typer.Option(),
    ] = Path("/Users/nick/eeg/迁移/daima/autodl_latest_data"),
) -> None:
    typer.echo(json.dumps(audit_integrity(source_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
