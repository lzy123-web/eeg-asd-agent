# Round 1 Review

<details>
<summary>GPT-5.6-Sol raw review</summary>

## 评分

| 维度 | 分数 | 评审判断 |
|---|---:|---|
| 1. Problem Fidelity | **9/10** | 始终围绕“不完整 EEG-ASD 个案如何形成可信评估档案”，证据边界和 non-goals 很清楚。 |
| 2. Method Specificity | **7/10** | 已达到可实施架构级别，但能力证据抽取、状态转移、模型隔离和报告主张校验仍缺少字段级契约。 |
| 3. Contribution Quality | **6/10** | 作为毕设系统贡献集中且合理；作为顶会方法贡献，目前主要是高质量系统整合，“为什么必须由 LLM 规划”尚未成立。 |
| 4. Frontier Leverage | **8/10** | LLM、工具调用、RAG、结构化输出的位置总体正确，没有让 LLM 代替信号算法；但逐步 Planner 略有过度使用。 |
| 5. Feasibility | **8/10** | 零训练、封闭工具、本地 CSV/BM25 降级路径使三周 V1 基本可信；旧模型契约和真实数据仍是主要风险。 |
| 6. Validation Focus | **8/10** | 两项 claims 和两项 ablations 很克制；但当前实验只能证明“有分支”，不能证明 LLM Agent 的必要性。 |
| 7. Venue Readiness | **4/10** | 可形成扎实毕设、系统演示或应用型论文，但尚不具备 NeurIPS/ICML/ICLR 主会所需的方法新颖性和普适结论。 |

**OVERALL SCORE：7.20/10**

计算：`9×15% + 7×25% + 6×25% + 8×15% + 8×10% + 8×5% + 4×5% = 7.20`

**CALIBRATION: none**

**GAP：** 这是一个约 **8/10 的毕设系统方案**，但只有约 **4/10 的顶会投稿成熟度**。差距不在于缺少更多模块或更强 EEG 模型，而在于核心因果论证尚未闭合：当前状态机、前置条件和确定性规则已经足以处理大多数六个场景，因此它证明了“线性 Pipeline 不够”，却还没有证明“逐步 LLM Planner 必须存在”。最小修订方向不是追加算法，而是明确 LLM 只负责哪些确定性程序不适合承担的语义工作，并把证据摄取、状态转移和报告主张落实为可验证契约。

## 低于 7 分的维度

### 3. Contribution Quality — 6/10

- **Weakness：** 当前 dominant contribution 仍可被概括为“把现有 Agent Harness 接到 EEG、模型和 RAG 上”。Evidence Router 是有价值的安全规则，但还不足以成为顶会级方法洞见。更关键的是，确定性规则 Planner 很可能在现有场景中达到同样表现。
- **Concrete method-level fix：** 将贡献收敛为一个“证据分层的个案控制平面”，并明确三层职责：
  1. 确定性 Eligibility Engine 根据前置条件计算合法动作；
  2. LLM 只在多个合法动作之间依据自然语言目标和非结构化材料进行语义选择；
  3. 所有工具后置条件、状态更新、阻断和失效传播均为确定性逻辑。

同时为非结构化能力报告定义最小抽取契约：

```text
domain_label
source_artifact_id
source_span / quote
instrument_name
raw_or_standardized_score
extraction_confidence
human_confirmation_status
```

只有来源片段存在且已确认的 `ability_domain` 才能解锁 AbilityKG。

- **Priority：P0**。这是决定系统究竟是“模块拼装”还是“有清晰控制语义的系统贡献”的关键修订。

### 7. Venue Readiness — 4/10

- **Weakness：** 没有新的学习方法、通用规划机制、数据资源或临床有效性结论；验证主要是软件正确性场景。即使全部成功，也只能支持“系统能按设计工作”，不能支持顶会级普适方法主张。
- **Concrete method-level fix：** 当前阶段不要人为制造算法 novelty。将交付目标明确设为：
  - 毕设：完整可运行系统、可审计 trace、真实去标识案例演示；
  - 论文：应用系统、demo、workshop 或领域信息学方向；
  - 后续研究：只有当该证据控制机制能跨任务复用，并表现出传统工作流难以实现的鲁棒性时，再提炼一般化方法。

论文标题和摘要也应避免“智能诊断”“新型 ASD 评估方法”等表述。

- **Priority：P1**。不影响 V1 开发，但必须尽早校准投稿叙事。

## 最关键的方法修订

当前最自然的 foundation-model 路线不是让 LLM 每一步都“思考下一工具”，而是：

```text
用户目标/非结构化报告
        ↓ LLM：结构化语义解析
确定性状态与证据门控
        ↓
合法动作集合
        ↓ LLM：仅在确有歧义时选择
确定性工具执行与状态更新
        ↓
Claim–Evidence Ledger
        ↓ LLM：受约束表述
确定性渲染与校验
```

这仍然是 Agent，也保留 Context、RAG、工具、Memory 和 Observation 循环，但更容易实现、更安全，也更能解释 LLM 的不可替代位置。

报告不应只按章节区分内容。建议每条报告主张都携带：

```text
claim_type: case_observation | model_output | external_knowledge
source_ids: [...]
value_refs: [...]
assertion_strength: observed | model_estimated | literature_reported
```

否则即使 Evidence ID 合法，LLM 仍可能用真实来源支持一个来源并未蕴含的结论。

## Simplification Opportunities

- 不必同时强调 ReAct 和 Plan-and-Execute；保留一个外显的 `State → Action → Observation → StateUpdate` 控制循环即可。
- Evidence Router 应实现为状态门控策略，不要发展成独立智能模块。
- 将“短期、情景、语义 Memory”简化为 `CaseStore + EvidenceStore`；AbilityKG 和论文库本身就是外部知识源。
- Reporter 优先使用确定性模板；LLM只生成被 schema 限制的解释字段。
- 若旧模型的数据划分或特征契约无法核实，Readiness 应直接返回 `unsupported_legacy_model`，而不是为了系统完整性强行输出分类分数。
- AbilityKG 中的 Task 只能作为“外部知识候选项供人工复核”，不能升级成个性化干预建议。

## Modernization Opportunities

- 把 LLM 用在非结构化量表/报告的带来源抽取，而不是原始 EEG 解释。
- 使用严格结构化输出和枚举动作；不要允许自由生成工具名、参数或 Evidence ID。
- 为每个 artifact 保存 hash、schema version、tool version 和生成关系，以支持继续运行、缓存和上游变更后的结果失效。
- LLM Planner 的最小验证应包含“相同个案、不同用户目标”和“相同目标、不同 Observation”。否则只能证明条件分支，不足以说明自然语言 Agent 的价值。
- 若规则 Planner 与 LLM Planner 表现相同，应诚实得出：**状态化 Agent 架构必要，但 LLM 对数值执行路径并非必要；LLM 的价值主要在目标理解、非结构化证据摄取和报告综合。**

## Drift Warning

当前方案没有明显偏离 Problem Anchor。最需要警惕的漂移有三类：

- 从 EEG 特征或模型分数重新偷渡出“社交能力弱”等能力结论；
- 把 AbilityKG 的 Task 写成针对本个案的治疗或训练处方；
- 为追求论文感加入 Multi-Agent、新 EEG 模型、向量数据库和大规模 benchmark。

此外，把“固定 Pipeline 不够”表述成“必须用 LLM Planner”也是一种论证漂移。前者已经成立；后者仍需限定适用位置或提供证据。

## Verdict

**REVISE**

这是一个方向正确、范围克制、很适合作为本科或硕士项目主线的方案。P0 修订不是增加功能，而是把控制权划分、能力证据摄取契约和 claim-level provenance 写实。完成后可以成为很好的 V1/毕设系统；不应在当前阶段包装成顶会算法贡献。

`review_independence: same-family/local`  
`acceptance_status: provisional`

</details>
