# Refinement Report

**Problem**: 把模糊的 EEG + ASD + Agent + 师兄 RAG 想法收敛成明确、可实现的毕设系统。
**Date**: 2026-08-06
**Rounds**: 3 / 5
**Final Score**: 9.15 / 10
**Final Verdict**: READY

## Output Files

- Review summary: refine-logs/REVIEW_SUMMARY.md
- Final proposal: refine-logs/FINAL_PROPOSAL.md
- Score history: refine-logs/score-history.md
- Verbatim reviews: round-1-review.md, round-2-review.md, round-3-review.md

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 9 | 7 | 6 | 8 | 8 | 8 | 4 | 7.20 | REVISE |
| 2 | 9 | 9 | 8 | 9 | 9 | 8 | 4 | 8.45 | REVISE |
| 3 | 10 | 10 | 9 | 9 | 9 | 9 | 4 | 9.15 | READY |

## Final Proposal Snapshot

- Agent 是 EEG-ASD 个案工作流助手，不是自动诊断器。
- 核心贡献是 evidence-typed、可暂停恢复的 case control plane。
- LLM 只负责目标理解、语义歧义和受限报告；确定性层负责合法动作、EEG 数值和状态更新。
- 师兄 paper RAG 可由 EEG/model evidence 触发；个性化 AbilityKG 必须由 confirmed ability evidence 触发。
- Memory 简化为 SQLite CaseStore/EvidenceStore。

## Method Evolution Highlights

1. 将七项 Harness 能力收敛为一个控制平面，而不是七项贡献。
2. 把 EEG evidence 与 ability evidence 分层，修复师兄 RAG 的输入语义。
3. 用 GoalSpec、terminal status 和 resume scope 闭合暂停恢复任务。

## Pushback / Drift Log

| Reviewer Said | Author Response | Outcome |
|---|---|---|
| 引入 Claim–Evidence Ledger | 用户明确不希望该选题回到主线，仅保留 finding-level provenance | rejected as unnecessary scope |
| 提高顶会 novelty | 当前先完成系统，算法创新后置 | deferred |
| LLM 失败走规则回退 | 真正语义歧义时改为请求澄清/人工复核 | accepted |

## Remaining Weaknesses

设计已闭合，但尚需真实实现 SQLite resume、AbilityEvidence gate 和真实 Provider 语义场景；缺少去标识真实个案；当前不具备顶会算法新颖性或临床性能依据。

## Raw Reviewer Responses

三轮完整原始回复已逐字保存在对应 round-N-review.md 文件中，避免在此重复同一大段内容。

## Next Steps

进入实现：先补 EvidenceProfile/终态/优先级，再做 SQLite resume，最后跑通“先 EEG、后补能力档案”的中心演示。

