from __future__ import annotations

import json
from pathlib import Path

from eeg_asd_agent.benchmarks import CorePlanningBenchmark

ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = ROOT / "benchmark_cases" / "core_planning_cases.jsonl"


def test_core_planning_cases_have_required_independent_evaluation_fields() -> None:
    cases = [
        json.loads(line)
        for line in CASE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(cases) == 18
    required = {
        "user_request",
        "initial_state",
        "required_tools",
        "forbidden_tools",
        "order_constraints",
        "expected_stop",
        "rationale",
    }
    assert all(required <= set(case) for case in cases)


def test_core_planning_benchmark_runs_three_baselines_without_leakage(
    tmp_path: Path,
) -> None:
    result = CorePlanningBenchmark(CASE_PATH).run(tmp_path)
    assert result["case_count"] == 18
    assert set(result["variants"]) == {
        "fixed_pipeline",
        "deterministic_state_planner",
        "deterministic_without_manifest",
    }
    state_metrics = result["variants"]["deterministic_state_planner"]
    assert state_metrics["required_tool_coverage"] == 1.0
    assert state_metrics["forbidden_invocation_rate"] == 0.0
    assert state_metrics["order_constraint_accuracy"] == 1.0
    assert state_metrics["correct_stop_rate"] == 1.0
    assert result["leakage_guard"]["expected_fields_passed_to_planner"] == []
    assert all(
        detail["planner_input_keys"] == ["user_request", "initial_state"]
        for detail in result["details"]
    )
    assert result["llm_planner_with_manifest"]["status"] == "NOT RUN"
    assert result["llm_planner_with_manifest"]["fixture_substitution"] is False
    assert (tmp_path / "benchmark_results.json").exists()
    assert (tmp_path / "case_results.jsonl").exists()
    assert (tmp_path / "error_slices.csv").exists()
    assert (tmp_path / "config_snapshot.json").exists()
    assert (tmp_path / "benchmark_summary.md").exists()


def test_manifest_baseline_avoids_blocked_model_invocation(tmp_path: Path) -> None:
    result = CorePlanningBenchmark(CASE_PATH).run(tmp_path)
    with_manifest = result["variants"]["deterministic_state_planner"]
    without_manifest = result["variants"]["deterministic_without_manifest"]
    assert with_manifest["forbidden_invocation_rate"] == 0.0
    assert without_manifest["forbidden_invocation_rate"] > 0.0
