# Known Limitations

1. The senior model path is historical OOF replay. It is not prospective inference and cannot support
   an ASD diagnosis, exclusion, calibrated probability, or clinical severity claim.
2. Phase 2 knowledge retrieval reads exported senior KG CSV files using deterministic lexical
   selection. The senior live RAG runtime has not yet been proven runnable.
3. `OfflineReportGenerator` is deterministic fixture infrastructure. It validates schema,
   orchestration, allowlist projection, and fallback behavior—not real LLM quality.
4. No participant-level EDF/BDF/SET/FIF file is available in the scanned assets. Phase 2 therefore
   uses feature tables and existing results rather than claiming a real raw-EEG E2E.
5. The current `NextActionPlanner` is deterministic and intentionally small. It demonstrates
   incremental state-driven decisions but has not yet been compared on the Phase 3 independent
   planning benchmark.
6. Core allowlists are structural and deny by default, but semantic correctness still depends on
   truthful tool normalization. The tests cover unknown nested fields, paths, raw-like keys, and
   secret-like keys.
7. Existing historical benchmark artifacts are fixture-mode harness checks. They are not evidence
   that an LLM planner or reporter outperforms a baseline.
