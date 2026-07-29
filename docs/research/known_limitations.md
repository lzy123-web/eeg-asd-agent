# Known Limitations

1. The senior model path is historical OOF replay. It is not prospective inference and cannot support
   an ASD diagnosis, exclusion, calibrated probability, or clinical severity claim.
2. The senior live RAG runtime is not runnable here: the Neo4j package is absent and the local Bolt
   port is closed. `SeniorKnowledgeAdapter` uses exported senior KG/paper CSV files and always reports
   `mode=offline_local_index`.
3. `OfflineReportGenerator` is deterministic fixture infrastructure. It validates schema,
   orchestration, allowlist projection, and fallback behavior—not real LLM quality.
4. No participant-level EDF/BDF/SET/FIF file is available in the scanned assets. Phase 2 therefore
   uses feature tables and existing results rather than claiming a real raw-EEG E2E.
5. The current `NextActionPlanner` is deterministic and intentionally small. It scores perfectly on
   the 18-case local benchmark, but those cases are hand-designed implementation checks and do not
   establish generalization to unseen requests.
6. Core allowlists are structural and deny by default, but semantic correctness still depends on
   truthful tool normalization. The tests cover unknown nested fields, paths, raw-like keys, and
   secret-like keys.
7. Existing historical benchmark artifacts are fixture-mode harness checks. They are not evidence
   that an LLM planner or reporter outperforms a baseline.
8. No real LLM Provider was configured. `llm_planner_with_manifest` is `NOT RUN`; no fixture result is
   substituted for it.
