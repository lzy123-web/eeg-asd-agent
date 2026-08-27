# Round 3 Review

<details>
<summary>GPT-5.6-Sol raw review</summary>

## Round 3 结论

两个 P0 契约已经闭合，**没有剩余的 V1 架构级阻塞项**。方案现在满足：

- 原 Problem Anchor 未漂移；
- dominant contribution 单一且清楚；
- 方法具体到可直接进入实现；
- LLM、RAG、确定性工具的职责边界合理；
- 没有明显复杂度膨胀；
- 验证能够区分状态控制、语义控制和证据门控的作用。

这次的 `READY` 仅表示“研究方案和毕设 V1 已准备好实施”，不表示已经达到顶会算法投稿标准。

## Scores

| 维度 | Round 2 | Round 3 | 判断 |
|---|---:|---:|---|
| 1. Problem Fidelity | 9 | **10/10** | 目标用户、原始瓶颈、证据边界、resume 场景和 non-goals 完全一致。 |
| 2. Method Specificity | 9 | **10/10** | GoalSpec、完成谓词、终态、输出类型、决策权、失败行为、证据门控和失效传播均已明确。 |
| 3. Contribution Quality | 8 | **9/10** | `evidence-typed resumable case control plane` 是一个集中、可展示、可验证的 thesis systems contribution。 |
| 4. Frontier Leverage | 9 | **9/10** | LLM只处理语言语义和歧义；RAG作为来源明确的知识工具；确定性层控制数值和安全。 |
| 5. Feasibility | 9 | **9/10** | 复用现有 V1、SQLite、本地 CSV/BM25、零训练，实施路径可信。 |
| 6. Validation Focus | 8 | **9/10** | 三种机制的归因已经拆开，且明确规定 LLM 无增益时收缩 claim。 |
| 7. Venue Readiness | 4 | **4/10** | 仍不是 NeurIPS/ICML/ICLR 主会算法贡献，但这是明确 non-goal。 |

**OVERALL SCORE：9.15/10**

计算：`10×15% + 10×25% + 9×25% + 9×15% + 9×10% + 9×5% + 4×5% = 9.15`

相较 Round 2 的 **8.45**，提升 **0.70**；相较 Round 1 的 **7.20**，提升 **1.95**。

**CALIBRATION: none**

**GAP：** 当前已经没有需要继续重构提案的核心方法缺口。剩余差距主要是“设计已经闭合，但实现和验证尚未发生”：需要把安全优先级落实成配置表，跑通中心 resume 场景，并验证 Semantic Controller 在真实歧义请求上不会越权。顶会层面的 gap 则仍是缺乏通用算法创新、跨领域验证和新的理论性质；这一 gap 是有意保留的，不应在 V1 阶段通过增加模块强行填补。

## Anchor Preservation

**Preserved，无实质漂移。** 修订后的方法仍然解决原问题：面对不同完整度、可能失败、可以补证据后继续的 EEG-ASD 个案，系统如何合法选择工具、保持证据边界，并生成非诊断性档案。

控制平面改为确定性优先并没有削弱 Agent 叙事。个案可从不同资产阶段进入；Observation 会改变后续合法动作；用户自然语言决定执行范围；缺失证据可能导致暂停而非失败；新证据加入后只恢复依赖步骤；非结构化语义和结构化执行由不同层负责。

需要保持措辞精度：不要声称条件 DAG 完全做不到这些，而应声称它不能独立完成自然语言目标理解和人机证据澄清。

## Contribution Focus

**Dominant contribution 已经足够锋利，不再过宽。** 推荐固定表述：

> 一个面向不完整 EEG-ASD 个案、按证据类型门控工具和知识路径、支持暂停恢复的 case control plane。

Semantic Controller 是语义接口；Eligibility/Completion Policy 是核心控制机制；AbilityKG 是证据门控的领域实例；SQLite resume 是核心任务所需状态能力；typed report provenance 是安全出口，而非独立贡献。

继续保持 Claim–Evidence 不进入 thesis mainline 是正确选择。现有 finding-level provenance 已足够支撑 V1 的报告边界。

## Implementation Specificity

目前已经足够开始编码：

```text
GoalSpec
→ legal actions
→ deterministic or semantic selection
→ Observation
→ deterministic StateUpdate
→ completion/terminal check
→ output or checkpoint
```

Ability 桥接也已经合法：

```text
EEG/model evidence
   ✗ 不得推出 ability domain

confirmed ability evidence
   + matched KG alias
   → 才可进入 AbilityKG
```

没有发现残留的非法 EEG-to-Ability 映射。

## Simplicity Assessment

**方法已是较小充分系统，没有明显 overbuilt。** 一个 Semantic Controller、一个确定性控制层、一个封闭工具注册表和一个 SQLite 文件已经足够。当前不需要独立 Planner 服务、Evidence Router、Claim–Evidence 子系统、工作流数据库、向量数据库重建、Multi-Agent 或新模型训练。

## Frontier Leverage

**使用方式恰当。** 最现代也最合理的地方不是让 LLM 控制 EEG 数值流程，而是结构化自然语言目标、在合法候选之间解决语义歧义、对不确定语义主动 abstain、受来源约束地组织报告，以及配合 RAG 和持久状态完成跨轮恢复。

## 是否仍有 Blocking V1 Contract

**没有。** 以下仅是实现级细化：

1. “低置信”定义为 schema 非法、action 不在候选集、action 与 GoalSpec 冲突或模型显式 `needs_clarification`，不依赖模型自报概率。
2. safe fixed priority 落实为每个 goal 的静态 action priority。
3. `resume_case` 保留原始 GoalSpec，并增加 `resume_scope`。

## Venue Readiness — 4/10

- **Weakness：** 目前没有新的 Agent 算法、学习机制、通用 RAG 方法或 EEG 建模结果。
- **Fix：** V1 阶段不修。未来若要转向顶会，应从多个应用中提炼通用的 evidence-typed control abstraction，而不是继续增加 EEG 模块。
- **Priority：P2 / deferred。**

## Readiness Separation

- **Implementable thesis V1：9.5/10 — READY**
- **Coherent thesis contribution：9/10 — READY**
- **Top-venue algorithmic novelty：4/10 — NOT READY，且当前无需解决**

合理答辩定位：

> 本工作贡献的是一个经过领域证据边界约束、支持不完整状态和增量恢复的 EEG-ASD 工作流 Agent 系统，而不是新的 ASD 分类算法或临床诊断方法。

## Simplification Opportunities

- CaseStore 和 EvidenceStore 保持为同一 SQLite 内的逻辑表。
- `stopped_model_unsupported` 按 goal 区分：模型专项目标停止；完整评估目标变为 `completed_with_gaps`。
- 未确认 ability evidence 是否阻塞按 goal 区分：普通评估有限完成；明确要求个性化 AbilityKG 时等待输入。
- V1只支持人工录入 confirmed ability profile；自动抽取延后。

## Modernization Opportunities

- 使用 JSON Schema 或 constrained tool calling 生成 GoalSpec/action ID。
- Manifest 增加静态 priority、retryable error codes 和 max attempts。
- 所有工具调用使用 artifact hash 和 idempotency key。
- Semantic Controller 优先显式 abstain，不使用自由文本置信度。
- 外部 Provider 只接收脱敏摘要，并记录 provider/model/version。

## Remaining Action Items

1. P1：goal × action 确定性优先级表。
2. P1：可程序检测的 semantic failure 条件。
3. P1：original goal 与 resume scope。
4. P1：按 goal 统一 model unsupported 和 unconfirmed ability 终态。
5. P1：A–F fixture 与测试 oracle。
6. P1：中心 resume 演示，只重跑 AbilityKG 与 stale report。
7. P2：一个去标识真实案例，仅用于演示。

## Drift Warning

当前没有明显漂移。实施阶段继续防止 imported/unverified model result 被写成系统验证结论、KG alias 自动升级为 confirmed ability、AbilityKG Task 被写成处方、为证明 Agent 感重新引入自由规划/Multi-Agent，以及把分类准确率变成 thesis 主线。

## Verdict

**READY**

满足规则：overall **9.15 ≥ 9**，Problem Anchor 保留，dominant contribution 单一，没有明显复杂度膨胀，没有剩余 V1 blocking contract。

此处 READY 意味着：**停止继续扩写方案，进入实现与测试。**

`review_independence: same-family/local`  
`acceptance_status: provisional`

</details>
