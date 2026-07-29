# Core Planning Benchmark

**Date:** 2026-07-29  
**Mode:** deterministic offline implementation evaluation  
**Cases:** 18

## Scope

The cases cover narrow quality and readiness requests, feature tables, existing model results,
existing reports, completed steps, blocked model readiness, unavailable RAG, missing inputs,
clinical-diagnosis requests, unregistered capabilities, and multi-goal requests.

Each JSONL case contains:

- `user_request`
- `initial_state`
- `required_tools`
- `forbidden_tools`
- `order_constraints`
- `expected_stop`
- `rationale`

Only `user_request` and `initial_state` are passed to a planner. `PlanningEvaluator` reads expected
fields after prediction. The production planner lives in `planning/incremental.py`; benchmark
evaluation lives in `benchmarks/core_planning.py`.

## Compared systems

1. `fixed_pipeline`: always emits the same over-complete sequence.
2. `deterministic_state_planner`: executes the incremental Observe/Decide/Execute/Update loop and
   checks the closed Registry manifests.
3. `deterministic_without_manifest`: state/goal rules without capability precondition checks.

`llm_planner_with_manifest` is `NOT RUN`: no Provider credentials were configured, the call count is
zero, and there is no Fixture substitution.

## Results

| Variant | Required coverage | Forbidden rate | Order accuracy | Correct stop | Tool precision |
|---|---:|---:|---:|---:|---:|
| fixed_pipeline | 0.8611 | 0.8727 | 0.9444 | 0.6667 | 0.4722 |
| deterministic_state_planner | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| deterministic_without_manifest | 1.0000 | 0.0182 | 1.0000 | 1.0000 | 0.9921 |

The no-Manifest failure occurs when a feature case has no deployable model or replay row: the
baseline selects historical replay, whereas the manifest-aware Agent observes blocked readiness and
does not invoke it.

## Interpretation boundary

These results show that the implementation responds correctly on this fixed, hand-reviewed suite and
that the evaluator catches fixed-pipeline and precondition failures. They do not establish unseen
request generalization, real LLM planning quality, statistical superiority, or clinical utility.

## Reproduction

```bash
.venv/bin/python -m eeg_asd_agent.cli benchmark core-planning \
  --offline \
  --output-dir artifacts/benchmarks/core-planning
```

Outputs:

- `benchmark_results.json`
- `case_results.jsonl`
- `error_slices.csv`
- `config_snapshot.json`
- `benchmark_summary.md`
