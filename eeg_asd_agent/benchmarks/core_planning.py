from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, cast

from eeg_asd_agent.execution import IncrementalWorkflowExecutor, ObservationNormalizer
from eeg_asd_agent.intent import RuleBasedIntentParser
from eeg_asd_agent.planning import NextActionPlanner
from eeg_asd_agent.schemas import AssessmentCaseState, InputKind, RequestedGoal
from eeg_asd_agent.tools.core_workflow import build_core_registry
from eeg_asd_agent.tracing import ArtifactStore, TraceLogger


class PlanningEvaluator:
    """Evaluate predictions after planning; expected fields never reach a planner."""

    @staticmethod
    def evaluate(case: dict[str, Any], actions: list[str]) -> dict[str, Any]:
        required = [str(item) for item in cast(list[object], case["required_tools"])]
        forbidden = [str(item) for item in cast(list[object], case["forbidden_tools"])]
        constraints = [
            [str(item) for item in cast(list[object], pair)]
            for pair in cast(list[object], case["order_constraints"])
        ]
        missing = [name for name in required if name not in actions]
        invoked_forbidden = [name for name in forbidden if name in actions]
        order_failures = [
            pair
            for pair in constraints
            if not (
                pair[0] in actions
                and pair[1] in actions
                and actions.index(pair[0]) < actions.index(pair[1])
            )
        ]
        predicted_stop = PlanningEvaluator._stop_category(actions)
        expected_stop = str(case["expected_stop"])
        stop_correct = (
            predicted_stop == expected_stop
            or expected_stop in {"completed_report", "completed_limited_report"}
            and predicted_stop == "completed_report"
        )
        unnecessary = [
            action for action in actions if action not in required and action not in forbidden
        ]
        return {
            "required_coverage": 1.0 - len(missing) / max(1, len(required)),
            "forbidden_invocations": len(invoked_forbidden),
            "forbidden_denominator": max(1, len(forbidden)),
            "order_accuracy": 1.0 - len(order_failures) / max(1, len(constraints)),
            "correct_stop": float(stop_correct),
            "tool_precision": len([item for item in actions if item in required])
            / max(1, len(actions)),
            "unnecessary_calls": len(unnecessary),
            "missing_required": missing,
            "invoked_forbidden": invoked_forbidden,
            "order_failures": order_failures,
            "predicted_stop": predicted_stop,
        }

    @staticmethod
    def _stop_category(actions: list[str]) -> str:
        if "stop_with_unsupported_request" in actions:
            return "blocked_unsupported"
        if "stop_with_insufficient_input" in actions:
            return "blocked_insufficient"
        if "render_report" in actions:
            return "completed_report"
        if actions and actions[-1] == "check_model_readiness":
            return "readiness_only"
        if actions and actions[-1] == "check_data_quality":
            return "quality_only"
        return "incomplete"


class CorePlanningBenchmark:
    variants = (
        "fixed_pipeline",
        "deterministic_state_planner",
        "deterministic_without_manifest",
    )

    def __init__(self, case_path: Path) -> None:
        self.case_path = case_path

    def run(self, output_dir: Path) -> dict[str, Any]:
        cases = self._load_cases()
        output_dir.mkdir(parents=True, exist_ok=True)
        registry = build_core_registry()
        details: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        aggregate: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for case in cases:
            request, state_payload = self._planner_input(case)
            for variant in self.variants:
                state = AssessmentCaseState.model_validate(state_payload)
                if variant == "fixed_pipeline":
                    actions = self._fixed_pipeline(state, request)
                elif variant == "deterministic_without_manifest":
                    actions = self._without_manifest(state, request)
                else:
                    actions = self._state_planner(
                        state,
                        request,
                        registry,
                        output_dir / "traces" / f"{case['case_id']}.jsonl",
                    )
                evaluation = PlanningEvaluator.evaluate(case, actions)
                aggregate[variant].append(evaluation)
                detail = {
                    "case_id": case["case_id"],
                    "variant": variant,
                    "planner_input_keys": ["user_request", "initial_state"],
                    "actions": actions,
                    "required_tools": case["required_tools"],
                    "forbidden_tools": case["forbidden_tools"],
                    "expected_stop": case["expected_stop"],
                    "rationale": case["rationale"],
                    **evaluation,
                }
                details.append(detail)
                for error_type, values in (
                    ("missing_required", evaluation["missing_required"]),
                    ("invoked_forbidden", evaluation["invoked_forbidden"]),
                    ("order_failure", evaluation["order_failures"]),
                ):
                    if values:
                        errors.append(
                            {
                                "case_id": str(case["case_id"]),
                                "variant": variant,
                                "error_type": error_type,
                                "details": json.dumps(values, ensure_ascii=False),
                            }
                        )
                if not evaluation["correct_stop"]:
                    errors.append(
                        {
                            "case_id": str(case["case_id"]),
                            "variant": variant,
                            "error_type": "stop_mismatch",
                            "details": str(evaluation["predicted_stop"]),
                        }
                    )

        metrics = {
            name: self._aggregate(items) for name, items in aggregate.items()
        }
        provider_available = bool(os.getenv("OPENAI_API_KEY"))
        result = {
            "evaluation_mode": "deterministic_offline",
            "scientific_claim_allowed": False,
            "case_count": len(cases),
            "variants": metrics,
            "llm_planner_with_manifest": {
                "status": "NOT RUN",
                "reason": (
                    "Provider credentials are not configured."
                    if not provider_available
                    else "Provider detected but not run by deterministic benchmark command."
                ),
                "fixture_substitution": False,
                "call_limit": 20,
            },
            "leakage_guard": {
                "planner_input_keys": ["user_request", "initial_state"],
                "expected_fields_passed_to_planner": [],
                "evaluator_module": "eeg_asd_agent.benchmarks.core_planning:PlanningEvaluator",
            },
            "details": details,
        }
        store = ArtifactStore(output_dir)
        store.write_json("benchmark_results.json", result)
        store.write_json(
            "config_snapshot.json",
            {
                "case_file": self.case_path.name,
                "case_count": len(cases),
                "variants": list(self.variants),
                "max_steps": 16,
                "real_llm_calls": 0,
            },
        )
        for detail in details:
            store.append_jsonl("case_results.jsonl", detail)
        self._write_errors(output_dir / "error_slices.csv", errors)
        store.write_text("benchmark_summary.md", self._summary(result))
        return result

    @staticmethod
    def _planner_input(case: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        request = str(case["user_request"])
        initial = dict(cast(dict[str, Any], case["initial_state"]))
        payload = {"case_id": str(case["case_id"]), "user_request": request, **initial}
        return request, payload

    @staticmethod
    def _state_planner(
        state: AssessmentCaseState,
        request: str,
        registry: Any,
        trace_path: Path,
    ) -> list[str]:
        state.user_request = request
        state.requested_goal = RuleBasedIntentParser().parse(request)
        outcome = IncrementalWorkflowExecutor(
            registry,
            ObservationNormalizer(),
            TraceLogger(trace_path, state.trace_id),
        ).execute(state, NextActionPlanner(registry))
        return [
            decision.tool_name
            for decision in outcome.decisions
            if decision.tool_name is not None
        ]

    @staticmethod
    def _fixed_pipeline(
        state: AssessmentCaseState,
        request: str,
    ) -> list[str]:
        del state, request
        return [
            "inspect_case",
            "check_data_quality",
            "check_model_readiness",
            "inspect_existing_model_result",
            "research_replay_classifier",
            "retrieve_asd_knowledge",
            "generate_report_spec",
            "render_report",
        ]

    @staticmethod
    def _without_manifest(
        state: AssessmentCaseState,
        request: str,
    ) -> list[str]:
        goal = RuleBasedIntentParser().parse(request)
        completed = set(state.completed_workflows)
        actions: list[str] = []

        def add(name: str) -> None:
            if name not in completed:
                actions.append(name)

        if goal is RequestedGoal.UNSUPPORTED:
            return ["stop_with_unsupported_request"]
        add("inspect_case")
        has_input = bool(
            state.feature_table or state.existing_model_result or state.existing_report
        )
        if state.input_type is InputKind.INCOMPLETE or not has_input:
            add("stop_with_insufficient_input")
            return actions
        if goal is RequestedGoal.QUALITY_CHECK:
            add("check_data_quality")
            return actions
        if goal is RequestedGoal.READINESS_CHECK:
            add("check_data_quality")
            add("check_model_readiness")
            return actions
        if state.existing_model_result is not None:
            add("inspect_existing_model_result")
        if goal is RequestedGoal.FULL_ASSESSMENT and state.feature_table:
            add("check_data_quality")
            add("check_model_readiness")
            if state.existing_model_result is None:
                add("research_replay_classifier")
        if goal in {RequestedGoal.FULL_ASSESSMENT, RequestedGoal.KNOWLEDGE_SEARCH}:
            add("retrieve_asd_knowledge")
        add("generate_report_spec")
        add("render_report")
        return actions

    @staticmethod
    def _aggregate(items: list[dict[str, Any]]) -> dict[str, float]:
        total_forbidden = sum(int(item["forbidden_invocations"]) for item in items)
        forbidden_denominator = sum(int(item["forbidden_denominator"]) for item in items)
        return {
            "required_tool_coverage": round(
                mean(float(item["required_coverage"]) for item in items), 4
            ),
            "forbidden_invocation_rate": round(
                total_forbidden / max(1, forbidden_denominator), 4
            ),
            "order_constraint_accuracy": round(
                mean(float(item["order_accuracy"]) for item in items), 4
            ),
            "correct_stop_rate": round(
                mean(float(item["correct_stop"]) for item in items), 4
            ),
            "tool_precision": round(
                mean(float(item["tool_precision"]) for item in items), 4
            ),
            "mean_unnecessary_calls": round(
                mean(float(item["unnecessary_calls"]) for item in items), 4
            ),
        }

    def _load_cases(self) -> list[dict[str, Any]]:
        cases = [
            json.loads(line)
            for line in self.case_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not 15 <= len(cases) <= 20:
            raise ValueError("core planning benchmark must contain 15-20 cases")
        return cases

    @staticmethod
    def _write_errors(path: Path, errors: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["case_id", "variant", "error_type", "details"],
            )
            writer.writeheader()
            writer.writerows(errors)

    @staticmethod
    def _summary(result: dict[str, Any]) -> str:
        lines = [
            "# Core Planning Benchmark",
            "",
            f"- Evaluation mode: `{result['evaluation_mode']}`",
            f"- Cases: {result['case_count']}",
            "- Real LLM planner: `NOT RUN` (no fixture substitution)",
            "",
            "| Variant | Required coverage | Forbidden rate | Order | Stop | Precision |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        variants = cast(dict[str, dict[str, float]], result["variants"])
        for name, metrics in variants.items():
            lines.append(
                f"| {name} | {metrics['required_tool_coverage']:.4f} | "
                f"{metrics['forbidden_invocation_rate']:.4f} | "
                f"{metrics['order_constraint_accuracy']:.4f} | "
                f"{metrics['correct_stop_rate']:.4f} | "
                f"{metrics['tool_precision']:.4f} |"
            )
        lines.extend(
            [
                "",
                "These deterministic offline results evaluate the benchmark harness and planner "
                "behavior only. They are not evidence about a real LLM planner.",
                "",
            ]
        )
        return "\n".join(lines)
