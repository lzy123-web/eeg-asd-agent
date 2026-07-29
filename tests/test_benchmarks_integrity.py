from pathlib import Path

from eeg_asd_agent.benchmarks import BenchmarkSuite
from eeg_asd_agent.integrity import audit_integrity

ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_case_counts_and_outputs(tmp_path: Path) -> None:
    result = BenchmarkSuite(ROOT / "benchmark_cases").run("all", tmp_path)
    assert result["evaluation_mode"] == "fixture"
    assert result["scientific_claim_allowed"] is False
    assert result["suites"]["planning"]["case_count"] == 30
    assert result["suites"]["boundaries"]["case_count"] == 25
    assert result["suites"]["reports"]["case_count"] == 20
    assert (tmp_path / "benchmark_results.json").exists()
    assert (tmp_path / "benchmark_summary.md").exists()
    assert (tmp_path / "error_slices.csv").exists()


def test_deterministic_planning_acceptance_thresholds() -> None:
    result = BenchmarkSuite(ROOT / "benchmark_cases").planning()
    metrics = result["variants"]["deterministic"]
    assert metrics["plan_schema_validity"] == 1.0
    assert metrics["forbidden_invocation_rate"] == 0.0
    assert metrics["required_tool_coverage"] >= 0.9
    assert metrics["order_constraint_accuracy"] >= 0.9
    assert metrics["correct_stop_rate"] >= 0.9


def test_integrity_audit_reports_legacy_risk() -> None:
    root = Path("/Users/nick/eeg/迁移/daima/autodl_latest_data")
    result = audit_integrity(root)
    assert result["status"] in {"risk_detected", "no_exact_overlap_detected"}
    assert result["model_readiness"] in {"blocked", "restricted"}
