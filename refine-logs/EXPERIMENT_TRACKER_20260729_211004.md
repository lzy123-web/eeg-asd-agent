# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | 流程 sanity | Offline all | 30/25/20 | schema/artifacts | MUST | DONE | fixture only |
| R002 | M0 | E2E raw | Synthetic EEG | demo-01 | artifacts/safety | MUST | DONE | demo only |
| R003 | M0 | E2E feature | Senior OOF replay | demo-04 | trace/report | MUST | DONE | restricted |
| R004 | M0 | E2E report | Existing report | demo-06 | tool skipping | MUST | DONE | fixture |
| R101 | M1 | Planning anchor | Fixed/Det/LLM/Manifest | 30×3 | planning metrics | MUST | BLOCKED | needs real provider |
| R201 | M2 | Boundary anchor | 4 contexts | 25×3 | boundary metrics | MUST | BLOCKED | needs real provider |
| R301 | M3 | Report anchor | 4 contexts | 20×3 | report metrics | MUST | BLOCKED | needs real provider |
| R401 | M4 | Ablations | 8 configs | frozen | mixed | MUST | TODO | after R101-R301 |
| R501 | M4 | Multi-Agent | coordinator vs single | subset | quality/cost | NICE | TODO | P2 only |

