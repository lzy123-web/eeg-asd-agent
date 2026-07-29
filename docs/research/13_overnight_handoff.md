# Overnight Handoff

**Date:** 2026-07-29  
**Repository:** `/Users/nick/programs/eeg-system/eeg-asd-workflow-agent`  
**Phase 2 gate commit:** `0e2014b core workflow validated`

## 1. Current system classification

The production `run` path is now an **incremental state-driven agent**.

It is no longer the pre-audit plan-then-execute path: `NextActionPlanner.decide()` returns one Action,
the tool returns an explicit `StateUpdate`, `StateReducer` applies it, and the planner runs again.

The `plan` CLI subcommand and `WorkflowExecutor` remain as legacy compatibility paths; they are not
used as evidence for the current Agent classification.

## 2. Complete current call chain

```text
cli.py::run_command
  → cli.py::_load_case → AssessmentCaseState
  → workflow.py::run_workflow
  → intent/parser.py::RuleBasedIntentParser.parse
  → tools/core_workflow.py::build_core_registry
  → planning/incremental.py::NextActionPlanner.decide
  → tools/registry.py::ToolRegistry.preconditions_met
  → tools/registry.py::ToolRegistry.execute_outcome
  → core FunctionTool
  → ToolExecutionOutcome(ToolResult, StateUpdate)
  → execution/core.py::ObservationNormalizer.normalize
  → execution/core.py::StateReducer.apply
  → NextActionPlanner.decide again
  → generate_report_spec tool
     → AllowlistProjector.project (recursive deny-by-default)
     → ContextBuilder.report_context
     → StructuredReportGenerator.generate
     → FinalAssessmentReportSpec validation
     → StateUpdate(report_spec=...)
  → render_report tool
     → MarkdownRenderer.render (deterministic)
  → Trace + ArtifactStore
```

The report Provider receives only the user request, completed workflows, allowlisted visible results,
used-tool limitations, missing information, ReportSpec Schema, and non-diagnostic rules.

## 3. Three core Case results

### Case A — feature assessment

Request: `根据现有特征完成 ASD 辅助评估并生成报告。`

```text
inspect_case
→ check_data_quality
→ check_model_readiness
→ research_replay_classifier
→ retrieve_asd_knowledge
→ generate_report_spec
→ render_report
→ STOP
```

Result: completed; readiness restricted; classifier mode REPLAY; knowledge mode
`offline_local_index`; strict report and Markdown/HTML output created.

### Case B — explain existing result

Request: `只解释当前模型结果并检索相关知识。`

```text
inspect_case
→ inspect_existing_model_result
→ retrieve_asd_knowledge
→ generate_report_spec
→ render_report
→ STOP
```

Result: completed; historical classifier was not called.

### Case C — readiness only

Request: `只检查数据和模型是否具备执行条件，不生成完整报告。`

```text
inspect_case
→ check_data_quality
→ check_model_readiness
→ STOP
```

Result: completed readiness check; model blocked; no classifier, RAG, ReportSpec, or Renderer call.

## 4. Concrete dynamic decision

In Case A, the first two Observations left readiness unknown. `check_model_readiness` then wrote
`model_readiness=restricted`. Only on the next loop did the planner choose
`research_replay_classifier`; that Action records the first three Observation IDs in
`based_on_observations`.

In the blocked readiness test, the same decision point produces a different next Action:
`retrieve_asd_knowledge`, not replay. This is an executed Observation-dependent branch, not an
up-front fixed Plan.

## 5. Core tool truth matrix

| Tool | Kind | Current truth |
|---|---|---|
| `inspect_case` | REAL | Deterministic typed-state inspection |
| `check_data_quality` | REAL | Finite feature check; intentionally narrow |
| `check_model_readiness` | REAL | Availability/risk rules, not performance validation |
| `inspect_existing_model_result` | ADAPTER | Projects supplied result; does not validate origin |
| `research_replay_classifier` | REPLAY | Stored senior OOF row replay only |
| `retrieve_asd_knowledge` | OFFLINE | Senior export local index |
| `generate_report_spec` | OFFLINE by default | `OfflineReportGenerator`; structural validation only |
| `render_report` | REAL | Deterministic Markdown output |
| both stop tools | REAL | Typed control-state/artifact updates |

## 6. REAL / OFFLINE / FAKE / REPLAY status

- REAL: state inspection, quality/readiness rules, stop controls, Schema validation, Renderer, Trace.
- ADAPTER: existing result projection.
- OFFLINE: senior exported knowledge index and default report generator.
- REPLAY: senior historical OOF classifier output.
- FAKE: none in the core Registry.
- Legacy non-core tools still contain fixture/stub paths; they are excluded from the current core run.

## 7. Senior RAG status

**Live senior RAG is not integrated successfully.**

Actual inspection found Neo4j/Cypher retrieval followed by an external Spark generation API, not a
reusable embedding/vector RAG entry. The runtime probe failed because:

- `neo4j` is absent from the environment;
- local Bolt port `127.0.0.1:7687` is closed;
- the source relies on hard-coded credentials, external service access, and Windows paths.

No credential was copied or used and no senior source was modified. The fallback
`SeniorKnowledgeAdapter.retrieve()` returns `StructuredKnowledgeResult` with source, chunk, score,
metadata, and `mode=offline_local_index`.

## 8. Raw EEG tool status

No real participant EEG file was found, and raw EEG was not selected as a Phase 3 extension.

The legacy Registry still contains synthetic/metadata-oriented raw EEG tools, but the current core
Registry does not register them. Therefore the overnight result does **not** claim a real raw-EEG
E2E. Feature tables are the validated core input.

## 9. Fields actually filtered by the allowlist

The nested allowlist test proves removal of:

- an undeclared top-level `new_top_level`;
- nested `internal_stack`;
- nested `absolute_path`.

The normalizer also removes raw/signal/sample/path/file/password/secret/token/API-key-like fields and
redacts absolute path values. The full `feature_table`, OOF model internals, Adapter metadata, raw EEG,
and complete ToolResult do not appear in `llm_payload.json`.

The model projection allows only:

`model_name`, `model_version`, `readiness_status`, `predicted_pattern`, optional real `score`,
`output_meaning`, `limitations`.

The knowledge projection allows only:

`evidence_id`, `title`, `normalized_summary`, `source`, `relevance`, `limitations` plus query/mode.

## 10. ReportSpec and Renderer stability

`FinalAssessmentReportSpec` strictly requires:

`report_type`, `request_summary`, `executed_workflows`, `data_summary`, `model_assessment`,
`knowledge_interpretation`, `unavailable_sections`, `limitations`, `conclusion`,
`human_review_required`, and typed provenance.

Provider output receives one local safe JSON-structure repair attempt. If it still fails, the system
creates a deterministic limited ReportSpec. Invalid/missing ReportSpec cannot be rendered.

The same ReportSpec produces byte-identical Markdown. Cross-run semantic reports are stable after
excluding generated timestamp and random Observation IDs.

## 11. Test and reproduction commands

```bash
.venv/bin/python -m eeg_asd_agent.cli run \
  --case demo_cases/core_a_feature_assessment.json \
  --request "根据现有特征完成 ASD 辅助评估并生成报告。" \
  --offline --output-dir artifacts/core-a

.venv/bin/python -m eeg_asd_agent.cli run \
  --case demo_cases/core_b_existing_model.json \
  --request "只解释当前模型结果并检索相关知识。" \
  --offline --output-dir artifacts/core-b

.venv/bin/python -m eeg_asd_agent.cli run \
  --case demo_cases/core_c_readiness_only.json \
  --request "只检查数据和模型是否具备执行条件，不生成完整报告。" \
  --offline --output-dir artifacts/core-c

.venv/bin/python -m eeg_asd_agent.cli benchmark core-planning \
  --offline --output-dir artifacts/benchmarks/core-planning

.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/mypy eeg_asd_agent
.venv/bin/python -m compileall -q eeg_asd_agent
```

Final recorded gate: 62 pytest tests pass; ruff passes; mypy reports no issues in 32 source files;
compileall passes.

## 12. Phase gate status

| Phase | Status | Gate evidence |
|---|---|---|
| Phase 1 | PASS | Full source call-chain audit, truth matrix, benchmark boundary, five gaps |
| Phase 2 | PASS | Three cases, real dynamic branch, allowlist/ReportSpec/Renderer tests, all gates, required commit |
| Phase 3A | PASS with fallback | Live probe documented; structured same-interface local index tested |
| Phase 3C | PASS | 18 cases, evaluator isolation, three baselines, complete artifacts |
| Phase 4 | PASS | README corrected, handoff created, final audit commands pass |

## 13. Unresolved issues

1. No live Neo4j database or safe senior RAG deployment.
2. No real LLM planner/reporter evaluation; Provider status is `NOT RUN`.
3. No participant raw EEG input and no validated prospective ASD classifier.
4. The 18 planning cases are hand-designed and do not establish unseen-request generalization.
5. Existing model and replay validity remain restricted by legacy split/overlap concerns.
6. Legacy plan/benchmark modules coexist for compatibility and must not be confused with the core run.

## 14. Is this direction worth continuing?

**Yes, as a workflow-agent/software-safety thesis direction; no, as current evidence for clinical ASD
prediction quality.**

The overnight work now makes the main mechanism visible and testable: state changes alter actions,
manifests block forbidden invocation, allowlists have nested enforcement, and reports are
execution-grounded. This is a materially stronger and more demonstrable result than Claim–Evidence
field bookkeeping.

However, the two frozen research claims are not fully established. The deterministic benchmark is
mechanism evidence only, and the report-boundary claim still needs a real Provider experiment.

## 15. Highest-priority next three tasks

1. Run a blinded real-Provider planning/report experiment (maximum 20 calls per configured run, three
   repeats where stochastic variance matters) without exposing expected fields.
2. Add independently authored or held-out planning cases and adjudicate errors before using the
   benchmark in the thesis.
3. Obtain one safe real EEG sample and a prospectively valid model contract, or explicitly freeze the
   thesis scope as workflow orchestration/safety rather than clinical model performance.

## Research-claim boundary

Following the experiment-plan constraint, only two claims remain:

1. Case State + Capability Manifest improves tool selection/order/stop versus fixed or
   no-Manifest baselines.
2. Execution-grounded Context + deny-by-default Allowlist + JSON Schema reduces report boundary
   violations.

Current evidence supports implementation correctness and one deterministic precondition contrast.
It does not yet support real-LLM superiority, statistical generalization, or clinical effectiveness.
