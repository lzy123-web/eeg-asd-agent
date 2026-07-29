# Decision Log

## 2026-07-29 — Core loop uses explicit outcomes and a single reducer

Decision: core tools return `ToolExecutionOutcome`, containing an immutable `ToolResult` and an
explicit `StateUpdate`. `StateReducer` is the only component allowed to apply the update.

Reason: the pre-audit executor inferred state mutations from tool names. That made tool contracts
implicit and prevented a reliable next-action loop.

Consequence: legacy tools remain available only to the legacy planner path. The incremental core
Registry rejects a core tool that returns a bare `ToolResult`.

## 2026-07-29 — Feature-table scenario is the Phase 2 anchor

Decision: validate the incremental architecture with stored features, a legacy OOF replay, offline
knowledge, strict ReportSpec, and deterministic rendering.

Reason: no participant EEG file or validated prospective classifier is available. Adding raw EEG
processing would not strengthen the state-driven Agent claim at this gate.

Consequence: Phase 2 does not claim a complete participant-level EEG pipeline. Raw EEG remains an
optional later extension.

## 2026-07-29 — Existing model results receive their own projection tool

Decision: add `inspect_existing_model_result` instead of putting the existing result directly into
the report Context.

Reason: Case B must avoid re-running inference while still applying the model-result allowlist and
provenance rules.

Consequence: supplied model fields outside the manifest are dropped before Provider visibility.

## 2026-07-29 — Offline report generation is structural validation only

Decision: use the explicitly named `OfflineReportGenerator`. Invalid Provider output gets one local
JSON structure repair attempt; if validation still fails, the system creates a deterministic limited
ReportSpec.

Reason: no real Provider run is available, and a fixture must not be used as evidence for LLM
boundary quality.

Consequence: benchmark and report artifacts state `offline_structural_validation`; no “LLM improved”
claim is allowed.

## 2026-07-29 — Phase 3 selection

Decision: after the Phase 2 commit, perform extension A (senior RAG minimum real integration attempt)
and extension C (15–20 case independent planning benchmark). Do not start extension B unless both are
complete and a clear need remains.

Reason: A tests the most important adapter truth boundary; C directly evaluates the first frozen
research claim with three deterministic baseline families.
