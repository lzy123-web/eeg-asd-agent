# Round 2 Refinement

## Problem Anchor

- **Bottom-line problem：** 用户希望先做出一个类似 paicli、具备 LLM、Context、RAG、ReAct 循环、Plan-and-Execute、工具系统和 Memory 的完整 Agent，并把它落到 EEG 的 ASD 评估场景；当前最需要解决的不是算法创新，而是明确这个 Agent 到底替谁完成什么真实任务，以及 EEG 分析、师兄 RAG 和 LLM Agent 如何形成一条不虚假的闭环。
- **Must-solve bottleneck：** “EEG 分类器 + RAG + 报告”只是模块拼接，不能说明为什么需要 Agent；同时师兄 AbilityKG 的入口是能力域证据，而原始 EEG 特征不能未经验证直接映射成社交、沟通、情绪等能力缺陷。系统必须给出明确的个案任务契约，并在输入不完整、质量失败、已有中间结果和证据不够时能够跳过、停止、补充检索或请求人工输入。
- **Non-goals：** 当前阶段不追求新的 EEG 算法、模型性能创新、临床诊断、自动处方、Multi-Agent、大模型微调或把 EEG 频带特征强行解释成 ASD 行为能力。
- **Constraints：** 本机没有真实受试者 EEG；现有师兄模型存在数据划分和特征契约风险；师兄 Neo4j/Spark 服务当前不可用但图谱 CSV 与论文语料可读；开发应优先复用现有 V1 架构和师兄资产；先完成可运行系统，再谈论文创新。
- **Success condition：** 给定自然语言任务和一个包含原始 EEG、特征表、已有模型结果、评估报告中任意组合的个案包，Agent 能根据执行态决定下一步，调用封闭 EEG/模型/RAG 工具，正确跳过或停止，并输出一个带执行记录、模型边界和 Evidence ID 的非诊断性 ASD 辅助评估档案；用户能清楚说明“为什么这里必须是 Agent，而不是一条固定脚本”。

## Anchor Check

- **Original bottleneck：** 给 EEG-ASD Agent 一个真实、必要且可落地的任务，而不是堆模块。
- **Why preserved：** 新增内容只闭合语义失败、任务终态和 resume 语义，没有增加算法或改变用户。
- **Reviewer suggestions rejected as drift：** 不解决顶会 novelty，不扩大成通用医疗 Agent，不让 AbilityKG Task 成为处方。

## Simplicity Check

- **Dominant contribution：** evidence-typed、可暂停与恢复的 EEG-ASD case control plane。
- **Components merged：** Intent Parser 与 LLM Planner 合并为 `Semantic Controller`；CaseStore/EvidenceStore 使用一个 SQLite 文件中的逻辑表；Plan 表示仅为 goal、pending action 和 state。
- **Complexity rejected：** 无独立路由服务、无 Claim Ledger、无复杂工作流数据库。
- **Smallest adequate route：** 确定性层做能确定的事，LLM 只在语义确实影响选择时出场。

## Changes Made

### 1. 闭合语义失败策略

- **Reviewer said：** 真正依赖语义的分支不能在 LLM 失败后任意走固定优先级。
- **Action：** 定义四类决策：唯一合法动作直接执行；有安全优先级则按规则；语义相关则调用 Semantic Controller；其非法或低置信时进入 `blocked_waiting_input`，请求澄清或人工复核。
- **Impact：** 回退不再可能静默违背用户目标。

### 2. 闭合任务完成和输出契约

- **Reviewer said：** 需要区分完成、有限完成、等待输入和质量/模型停止。
- **Action：** 定义六个 terminal status、每个 goal 的完成谓词和 `full_dossier | limited_dossier | task_result | resumable_checkpoint` 四类输出。
- **Impact：** quality-only、bad EEG、模型不支持和 resume 的行为不再冲突。

### 3. 补齐最小审计字段

- **Action：** imported model result 增加 origin/contract status；AbilityEvidence 增加确认人与 alias mapping 状态；artifact 增加 derived_from/stale。
- **Impact：** 支持最小依赖失效，不增加基础设施。

## Revised Proposal

# 研究方案：EEG-ASD 个案工作流助手

## Problem Anchor

- **Bottom-line problem：** 用户希望先做出一个类似 paicli、具备 LLM、Context、RAG、ReAct 循环、Plan-and-Execute、工具系统和 Memory 的完整 Agent，并把它落到 EEG 的 ASD 评估场景；当前最需要解决的不是算法创新，而是明确这个 Agent 到底替谁完成什么真实任务，以及 EEG 分析、师兄 RAG 和 LLM Agent 如何形成一条不虚假的闭环。
- **Must-solve bottleneck：** “EEG 分类器 + RAG + 报告”只是模块拼接，不能说明为什么需要 Agent；同时师兄 AbilityKG 的入口是能力域证据，而原始 EEG 特征不能未经验证直接映射成社交、沟通、情绪等能力缺陷。系统必须给出明确的个案任务契约，并在输入不完整、质量失败、已有中间结果和证据不够时能够跳过、停止、补充检索或请求人工输入。
- **Non-goals：** 当前阶段不追求新的 EEG 算法、模型性能创新、临床诊断、自动处方、Multi-Agent、大模型微调或把 EEG 频带特征强行解释成 ASD 行为能力。
- **Constraints：** 本机没有真实受试者 EEG；现有师兄模型存在数据划分和特征契约风险；师兄 Neo4j/Spark 服务当前不可用但图谱 CSV 与论文语料可读；开发应优先复用现有 V1 架构和师兄资产；先完成可运行系统，再谈论文创新。
- **Success condition：** 给定自然语言任务和一个包含原始 EEG、特征表、已有模型结果、评估报告中任意组合的个案包，Agent 能根据执行态决定下一步，调用封闭 EEG/模型/RAG 工具，正确跳过或停止，并输出一个带执行记录、模型边界和 Evidence ID 的非诊断性 ASD 辅助评估档案；用户能清楚说明“为什么这里必须是 Agent，而不是一条固定脚本”。

## Technical Gap

固定线性 Pipeline 不能覆盖不同起点、失败和继续运行；条件 DAG 能处理部分结构化分支，但不能独立解释自然语言目标或从非结构化报告中与用户交互。另一方面，师兄 AbilityKG 接收的是能力域证据，EEG 特征不能直接跨越成能力结论。系统缺少的不是更多算法，而是一个把用户目标、个案状态、证据类型、合法动作和可恢复执行连接起来的控制平面。

## Method Thesis

- **One sentence：** 构建一个 evidence-typed、可暂停恢复的 EEG-ASD case control plane：确定性策略管理工具资格、状态和失败，LLM Semantic Controller 只处理自然语言语义，师兄 RAG 的论文与 AbilityKG 路径由 evidence type 解锁。
- **Smallest intervention：** 复用 V1 全部主干，仅补 EvidenceProfile、completion/terminal contract 和 SQLite CaseStore。
- **Contribution：** 工程型毕设系统，不宣称新 EEG、Agent 或 RAG 算法。

## Product Contract

- **用户：** EEG/ASD 研究人员、数据分析人员、评估助理。
- **输入：** 自然语言 goal + raw EEG / feature table / model result / ability profile / assessment report 的任意组合。
- **任务：** 判断个案现在能做什么，执行可验证工具，遇到失败或缺失时停止/询问，补证据后继续。
- **输出：** 完整或有限的 Assessment Dossier、单项任务结果或可恢复 checkpoint。

## System Graph

```text
User request + Case Package
       ↓
Semantic Controller → strict GoalSpec
       ↓
Case State + EvidenceProfile
       ↓
Deterministic Eligibility / Completion Policy
       ↓
  ┌─ one legal action → execute
  ├─ safe fixed priority → execute
  ├─ semantic ambiguity → Semantic Controller selects
  └─ semantic failure → request_input / request_review
       ↓
Closed Tool → Observation → deterministic StateUpdate → loop
       ↓
evidence-typed RAG gate → Allowlisted Context → ReportSpec → Renderer
       ↓
SQLite CaseStore/EvidenceStore → resume
```

## Goal and Terminal Contract

`GoalSpec.goal` 使用受控枚举：

```text
quality_check
feature_extraction
asd_research_assessment
explain_existing_result
knowledge_lookup
resume_case
```

完成谓词：

- `quality_check`：目标资产已有 quality Observation，即完成，输出 `task_result`。
- `feature_extraction`：质量通过且存在版本化 feature artifact，即完成。
- `asd_research_assessment`：必要 EEG/feature 状态已处理，模型路径已成功或显式 unsupported，按证据条件完成 RAG，生成完整或有限 dossier。
- `explain_existing_result`：结果 provenance 已检查，论文 RAG 已执行，生成 dossier；不重新运行模型，除非用户明确要求。
- `knowledge_lookup`：完成合法检索即可，不运行 EEG/模型。
- `resume_case`：仅执行新证据解锁或 stale 的后继步骤。

终态：

```text
completed
completed_with_gaps
blocked_waiting_input
stopped_quality_failure
stopped_model_unsupported
user_cancelled
```

输出：

- `completed` → `full_dossier` 或目标对应 `task_result`；
- `completed_with_gaps`、质量/模型停止 → `limited_dossier`，列出已完成和缺口；
- `blocked_waiting_input` → `resumable_checkpoint`，不视为最终完成；
- `user_cancelled` → 保存 trace，不生成新结论。

## Decision Semantics

| 状态 | 决策者 | 失败行为 |
|---|---|---|
| 只有一个合法动作 | deterministic policy | 工具失败按 Manifest 重试/停止 |
| 多个动作但有固定安全优先级 | deterministic policy | 选择最少副作用、最接近完成谓词者 |
| 多个动作且取决于用户语义 | Semantic Controller | 非法/低置信进入 `blocked_waiting_input` |
| 不存在合法动作且 goal 未完成 | deterministic policy | request input/review 或有限停止 |

Semantic Controller 只返回严格 `GoalSpec` 或合法 action ID，不生成工具名和参数。两个必要语义例子：

1. 同时有 raw EEG 和 imported model result，用户说“只复核已有结果，不重新跑模型”；
2. resume 后用户说“只补充量表相关知识，不改动上次 EEG 结论”。

## Evidence and RAG Contract

```text
EvidenceProfile
  eeg_signal: artifact_id, quality, preprocessing/schema version
  model: origin(generated|imported), contract_status(verified|unverified),
         model_version, output, score_semantics, limitations
  ability[]: domain_label, source_artifact_id, source_span, instrument,
             observed_score, extraction_mode, confirmed_by, confirmed_at,
             human_confirmation_status, kg_alias_status
  knowledge[]: evidence_id, source, retrieval_mode, retriever_score, limitations
```

- EEG/model evidence 只允许构造不含个体数值的论文 Query：特征族、任务、模型类型和限制。
- 只有 `human_confirmation_status=confirmed` 且 `kg_alias_status=matched` 的 ability evidence 才解锁 `Ability → Dimension → Task`。
- 低置信抽取或 alias 进入 `blocked_waiting_input`，不自动放行。
- KG Task 是外部知识候选项，必须人工复核，不是干预处方。
- Neo4j 不可用时遍历同一只读 CSV 图谱；paper corpus 使用 BM25；运行模式与真实 retriever score 写入 Evidence。

## EEG and Model Tools

EEG 工具保持确定性：inspect、quality、preprocess、events、epochs、features。模型暂时只是一项可替换工具：readiness 通过才执行简单 baseline；旧模型契约不可信则返回 `stopped_model_unsupported` 或继续生成不含当前推理的 limited dossier。模型性能不是当前系统主线。

## Memory and Resume

一个 SQLite 文件包含：

- `cases`：GoalSpec、Case State、terminal status、pending actions；
- `artifacts`：hash、schema/tool version、`derived_from`、`stale`；
- `observations`：白名单摘要和执行状态；
- `evidence`：ability/knowledge source 与确认状态。

补充新 ability evidence 后，只将依赖它的 RAG/report 标记 stale；EEG 结果因 artifact hash 未变而复用。

## Report Contract

Report Provider 只接收 Allowlist Context。每条 finding 只有轻量来源字段：

```text
finding_type: case_observation | model_output | external_knowledge
source_refs: [tool_run_id or evidence_id]
assertion_strength: observed | model_estimated | literature_reported
text: bounded explanation
```

校验 source_refs 存在、类型匹配、数值来自允许的工具摘要、外部知识来自实际检索。该机制只是报告安全字段，不作为 Claim–Evidence 研究主线。

## Central End-to-End Scenario

第一次用户只提供 raw EEG。Agent 完成质量、预处理、特征、可用的简单模型和 paper RAG，因没有能力档案而输出 `completed_with_gaps + limited_dossier`。之后用户补充确认过的能力档案并要求继续；Agent 从 CaseStore 恢复，只运行 AbilityKG 和 stale report，不重复 EEG/模型。这个场景同时验证 Context、工具、Observation 循环、RAG、Memory 和计划恢复。

## Failure Handling

- 质量失败：停止模型，生成有限结果；
- 模型 unsupported：保留 EEG 发现，明确缺口；
- 能力证据未确认：等待输入，不检索个性化 KG；
- LLM 语义选择失败：等待澄清，不静默改目标；
- LLM 报告非法：一次修复后确定性有限报告；
- 上游 artifact 变化：依赖节点 stale，按需重跑。

## Minimal Validation

### Claim 1：状态控制平面正确处理跳过、停止和恢复

- **对比：** fixed linear Pipeline vs deterministic eligibility。
- **场景：** raw EEG、features、existing result、quality-only、bad quality、resume。
- **指标：** required/forbidden invocation、skip/stop/resume accuracy、terminal correctness。

### Claim 2：受限语义控制提高目标遵循，证据门控减少非法 RAG

- **语义对比：** static priority vs Semantic Controller，仅测试两个以上真实歧义场景；指标为 goal adherence 与 illegal action rate。
- **RAG 对比：** no evidence typing vs full gate；指标为 unsupported AbilityKG invocation、alias confirmation 和 evidence ID coverage。
- **解释边界：** 若 LLM 无增益，只主张状态化控制有效，不主张 LLM Planner 必要。

## Compute & Timeline

- 0 GPU 小时；LLM 通过配置化 API，EEG/model/RAG 均可 CPU 运行。
- 第一阶段实现 EvidenceProfile、goal/terminal policy、SQLite resume；第二阶段跑真实 Provider 语义场景；第三阶段补一例去标识真实数据与完整演示。
- 当前目标是 READY TO IMPLEMENT 的毕设 V1，不追求顶会算法新颖性。
