# Round 2 Review

<details>
<summary>GPT-5.6-Sol raw review</summary>

## Round 2 结论

**Problem Anchor 保留完整，没有发生实质漂移。** 修订后的方法明显更具体、更聚焦，也更符合“先完成可信毕设 V1，再讨论算法创新”的优先级。

上一轮最重要的问题——“为什么几乎所有步骤都要由 LLM Planner 决定”——已经基本解决。现在的主线可以清楚概括为：

> 确定性控制平面保证合法性与可恢复性，LLM只处理自然语言目标和真正的语义歧义，证据类型决定 RAG 能否进入个案路径。

这已经是一项集中的系统贡献，而不是 EEG、Agent、RAG、Memory 的并列拼装。

## Re-scoring

| 维度 | Round 1 | Round 2 | 评审判断 |
|---|---:|---:|---|
| 1. Problem Fidelity | 9 | **9/10** | 原问题、用户、证据边界和 non-goals 均保留；resume 场景进一步强化了原任务。 |
| 2. Method Specificity | 7 | **9/10** | EvidenceProfile、AbilityEvidence、Eligibility Policy、失效传播和存储职责已具体到可实现程度。 |
| 3. Contribution Quality | 6 | **8/10** | `evidence-typed case control plane` 已成为清楚的 dominant contribution；AbilityKG 是受控应用实例，而非平行贡献。 |
| 4. Frontier Leverage | 8 | **9/10** | LLM被放在语言和歧义位置，数值、安全、状态更新保持确定性，符合当前 foundation-model 系统设计。 |
| 5. Feasibility | 8 | **9/10** | 零训练、本地降级、人工结构化 ability profile 和明确的 legacy-model 阻断使三周 V1 可信。 |
| 6. Validation Focus | 8 | **8/10** | 场景和 decisive checks 很集中；仍需把“状态适应性”和“LLM语义价值”分开归因。 |
| 7. Venue Readiness | 4 | **4/10** | 毕设系统成熟度明显提高，但没有改变顶会算法新颖性不足这一事实。 |

**OVERALL SCORE：8.45/10**

计算：`9×15% + 9×25% + 8×25% + 9×15% + 9×10% + 8×5% + 4×5% = 8.45`

相较 Round 1 的 **7.20**，提升 **1.25**。

**CALIBRATION: none**

**GAP：** 当前差距已经不再是架构方向错误，而是少量但关键的执行语义还没有完全闭合：语义规划失败时如何安全退化、不同 goal 的完成条件是什么、何时生成完整 dossier 或有限结果，以及 deterministic policy 与 LLM planner 的效果应该分别支持什么 claim。补齐这些契约后，它可以成为很扎实的 thesis V1；但即使全部实现，也不自动转化为顶会算法贡献。

## Problem Anchor：Preserved

没有发现方法漂移。几个修改都在原问题范围内：

- 确定性 Eligibility Policy 没有把 Agent 退化回固定脚本，而是承认大部分 EEG 工具依赖本来就应该确定执行。
- AbilityEvidence contract 直接修复了原始 EEG 到能力缺陷的非法桥接。
- CaseStore + EvidenceStore 服务于“不完整输入、停止、补证据、继续运行”。
- resume-after-new-evidence 是目前最强的端到端用例，因为它同时需要状态、依赖失效、RAG 门控和有限重执行。

唯一需要调整的是这句话：“固定 Pipeline 无法自然覆盖……”。建议精确改成：“固定线性 Pipeline 无法覆盖；条件 DAG 或工作流引擎可以处理部分分支，但不能独立完成自然语言目标解释和非结构化证据交互。”否则容易被指出 deterministic workflow 同样能实现条件跳转。

## Dominant Contribution：明显更锋利

现在的 dominant contribution 足够集中：一个 evidence-typed、可暂停和恢复的 EEG-ASD case control plane。Eligibility Policy、AbilityKG gate、typed report provenance、CaseStore/EvidenceStore 和 LLM Semantic Planner 都是该控制平面的组成机制，不再存在明显 contribution sprawl。

不过，正文中不要反复把 LLM、Context、RAG、ReAct、Plan-and-Execute、Tools、Memory 作为七项能力列举。这是产品需求清单，不是七项贡献。论文和答辩应始终回到一个控制平面。

## Method Complexity：已经简化，未明显过度设计

当前复杂度总体合理。特别正确的决定包括：没有建立独立 Evidence Router 服务；没有把 Claim–Evidence Ledger 提升为论文主线；V1 可以先只接受人工结构化 ability profile；Planner 不再控制无歧义的数值执行顺序；Neo4j、CSV 和 BM25 是同一接口下的运行模式，而不是多个系统贡献。

保留 finding-level typed provenance 是恰当折中。它只需要验证 `source_refs` 存在、source 类型与 finding 类型一致、数值来自对应 ToolResult、外部知识只引用实际检索内容，不需要重新引入独立 Claim–Evidence 子系统。

## Frontier Leverage：使用位置合适

LLM 的位置现在是合理且现代的：将自然语言请求映射到受控 goal；在多个合法动作间解决用户语义歧义；将非结构化材料抽取为待确认候选；在严格来源约束下生成报告。

剩余风险是：在当前 A–F 场景中，LLM可能几乎没有真正做出语义决策。至少应定义两个真正的语义歧义例子：同时存在 raw EEG 和已有模型结果，用户要求“复核已有结果，不要重新跑模型”；resume 后用户要求“只补充量表相关知识，不改动上次 EEG 结论”。如果 Intent Parser 和 Semantic Planner 高度重合，应合并为一个 `Semantic Controller`。

## 剩余 P0 系统契约

### 1. 语义失败时不能简单由 deterministic policy 接管

```text
唯一合法动作
→ deterministic execute

多个动作，但存在固定安全优先级
→ deterministic priority

多个动作，选择依赖用户语义
→ LLM Semantic Planner

LLM 非法、低置信或动作理由与 goal 不一致
→ request_input / request_review
```

这是当前最重要的剩余修订。

### 2. 补齐 goal completion 和 terminal status

```text
completed
completed_with_gaps
blocked_waiting_input
stopped_quality_failure
stopped_model_unsupported
user_cancelled
```

需要明确：quality-only 不必生成完整 Dossier；full assessment 遇到坏质量时生成有限 dossier；blocked_waiting_input 保存状态但不视为最终完成；resume 后只有依赖新证据的节点变 stale。

## Validation 需要重新分层归因

| 对比 | 能支持的结论 |
|---|---|
| fixed linear Pipeline vs deterministic eligibility | 状态化控制能正确跳过、停止、恢复 |
| static-priority eligibility vs LLM semantic selection | LLM是否改善自然语言目标遵循 |
| no evidence typing vs full gate | 证据门控是否减少无依据 AbilityKG 调用 |

如果 deterministic policy 已经正确完成 A–F，可以主张状态化控制平面有效；如果 LLM 与 static priority 相同，不能主张 LLM Planner 必要；如果 LLM只在语义歧义场景提高 goal adherence 且不增加非法动作，可以主张受限语义规划有实际作用；如果 naive RAG 产生无依据能力映射而 gate 后为零，可以支持证据类型边界这一系统 claim。

## 低于 7 分的维度

### Venue Readiness — 4/10

- **Weakness：** 当前结果仍然主要证明一个领域系统按契约运行，不能形成新的通用学习算法或 Agent 规划方法。
- **Method-level fix：** V1 阶段不要修复这一点。后续只有在多个领域任务中发现同一种 evidence-typed control abstraction，并能提出通用安全性质或新的规划机制时，才适合发展为顶会方法。
- **Priority：P2 / deferred**。强行在当前阶段增加算法创新会破坏 Problem Anchor。

## Simplification Opportunities

- 明确 Intent Parser 与 Semantic Planner 的边界；若实现上高度重合，合并成一个 Semantic Controller。
- CaseStore 和 EvidenceStore 可以是同一个 SQLite 数据库中的逻辑表，不要做成两个部署服务。
- Plan-and-Execute 不必另设长期计划模块；goal enum、pending actions 和 resume state 已经构成足够的计划表示。
- Paper RAG query 不应包含个案分数；只使用特征族、模型类型、研究任务和限制等非个体化字段。
- `relevance` 若由 LLM主观生成，应改为检索分数或明确标记为生成性判断。

## Modernization Opportunities

- 使用严格 schema/tool calling 输出 goal 和动作，而不是解析自由文本。
- imported model result 增加 `origin: generated | imported` 与 `contract_status: verified | unverified`；未验证结果不能等同于系统模型输出。
- AbilityEvidence 增加 `confirmed_by`、`confirmed_at` 和 KG alias mapping status。
- 用 `derived_from` 和 `stale` 字段完成最小依赖失效，不引入复杂工作流数据库。

## Drift Warning

当前没有 EEG-to-Ability 的非法桥接。后续仍需防止 LLM 从 source span 中概括原文未明确陈述的能力缺陷、KG alias 自动混同近似概念、AbilityKG Task 被写成个性化训练建议。低置信 alias 或间接描述应请求人工确认。

## Readiness 分离判断

- **Implementable thesis V1：8.8/10，基本可以开始实现。**
- **Coherent thesis contribution：8/10。**
- **Top-venue algorithmic novelty：4/10。**

## Remaining Action Items

1. **P0：** 语义失败时请求澄清，不做任意 deterministic fallback。
2. **P0：** 定义 goal enum、完成谓词、terminal status 和不同终态的输出类型。
3. **P1：** 给出最小 eligibility/priority 表。
4. **P1：** 分开评估状态控制与 LLM 语义选择。
5. **P1：** imported model result 添加来源和契约验证状态。
6. **P1：** 明确 ability domain 到 KG 节点的 alias 与人工确认规则。

## Verdict

**REVISE**

原因不是方向仍有问题，而是总分 **8.45 < 9**，且语义失败退化和终止状态两个系统契约尚未闭合。修完这两个 P0 后，该方案可视为 **READY TO IMPLEMENT 的 thesis V1**，但仍不会因此变成顶会算法方案。

`review_independence: same-family/local`  
`acceptance_status: provisional`

</details>
