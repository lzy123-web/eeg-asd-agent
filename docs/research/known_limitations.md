# Known Limitations

1. The default run is an Offline manifest-driven Agent simulation unless a real Provider is configured.
   It proves the protocol and harness execute; it does not establish LLM planning quality.
2. The real OpenAI-compatible Agent/Report Provider interfaces and strict schemas are implemented but
   have not been exercised with credentials in this development run.
3. No participant EEG is available. Raw EEG tests use deterministic synthetic signals or local fixture
   formats and do not support scientific or clinical claims.
4. The mounted senior MLP is a restricted research placeholder. Its target, split, scaler reconstruction
   and feature semantics do not support ASD diagnosis or performance claims.
5. Senior live Neo4j is unavailable in the current environment. RAG uses the same exported AbilityKG
   nodes/relationships plus paper BM25 and reports that fallback mode explicitly.
6. Several full-domain capabilities are declared but disabled Stubs: EEG pattern/profile analysis,
   ability/severity estimation, comparison, structured assessment, report parsing, visualization and
   training. Registry will not execute them until real implementations replace the Stubs.
7. Rule-based intent parsing remains a goal hint and offline-fixture aid. In real Agent mode, semantic
   planning belongs to the LLM, but no open-ended Provider evaluation has yet been run.
8. SQLite Memory is local case/episodic memory. It has no cross-case preference learning, vector recall,
   multi-user isolation or production encryption layer.
9. Deny-by-default Context projection prevents known raw/path/secret fields, but production deployment
   would still require stronger data governance, authentication, audit retention and adversarial tests.
10. The system is research software, not a clinical device. It must not diagnose ASD, exclude diagnosis,
    prescribe interventions or replace qualified human assessment.
