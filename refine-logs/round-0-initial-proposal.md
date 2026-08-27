# 研究方案：面向 EEG-ASD 辅助评估的个案工作流智能体

## Problem Anchor

- **Bottom-line problem：** 用户希望先做出一个类似 paicli、具备 LLM、Context、RAG、ReAct 循环、Plan-and-Execute、工具系统和 Memory 的完整 Agent，并把它落到 EEG 的 ASD 评估场景；当前最需要解决的不是算法创新，而是明确这个 Agent 到底替谁完成什么真实任务，以及 EEG 分析、师兄 RAG 和 LLM Agent 如何形成一条不虚假的闭环。
- **Must-solve bottleneck：** “EEG 分类器 + RAG + 报告”只是模块拼接，不能说明为什么需要 Agent；同时师兄 AbilityKG 的入口是能力域证据，而原始 EEG 特征不能未经验证直接映射成社交、沟通、情绪等能力缺陷。系统必须给出明确的个案任务契约，并在输入不完整、质量失败、已有中间结果和证据不够时能够跳过、停止、补充检索或请求人工输入。
- **Non-goals：** 当前阶段不追求新的 EEG 算法、模型性能创新、临床诊断、自动处方、Multi-Agent、大模型微调或把 EEG 频带特征强行解释成 ASD 行为能力。
- **Constraints：** 本机没有真实受试者 EEG；现有师兄模型存在数据划分和特征契约风险；师兄 Neo4j/Spark 服务当前不可用但图谱 CSV 与论文语料可读；开发应优先复用现有 V1 架构和师兄资产；先完成可运行系统，再谈论文创新。
- **Success condition：** 给定自然语言任务和一个包含原始 EEG、特征表、已有模型结果、评估报告中任意组合的个案包，Agent 能根据执行态决定下一步，调用封闭 EEG/模型/RAG 工具，正确跳过或停止，并输出一个带执行记录、模型边界和 Evidence ID 的非诊断性 ASD 辅助评估档案；用户能清楚说明“为什么这里必须是 Agent，而不是一条固定脚本”。

## Technical Gap

现有常见做法要么把 EEG-ASD 任务缩成一次分类，要么把预处理、模型和报告写成固定 Pipeline。前者无法容纳数据质量、缺失模态、已有结果、知识解释和人工复核；后者在所有个案上执行相同步骤，无法根据 Observation 改变动作，也不需要真正的 LLM Agent。

当前 V1 已经拥有执行所需的大部分基础设施，但系统叙事仍容易落入两个错误方向：

1. 把 Agent 说成“自动诊断 ASD 的医生”，超出 EEG 证据和临床边界；
2. 把师兄 RAG 当成任意文本搜索，在 EEG 输出后机械拼一段知识解释。

师兄真实 RAG 的结构是由个体弱能力进入 `Ability → Dimension → Task` 图谱，并联合论文知识生成解释或建议。因此，最小而诚实的切入点不是让 LLM 解释原始脑电，也不是让 EEG 直接推出能力缺陷，而是让 Agent 负责构建和管理一个**个案评估证据包**：EEG 工具产生客观信号证据，基线模型产生受限研究结果，已有量表/报告提供能力域证据，RAG 只在相应证据存在时检索对应知识，最终报告明确区分“本个案执行结果”和“外部知识”。

## Method Thesis

- **One-sentence thesis：** 构建一个 EEG-centered ASD 个案工作流智能体，把 LLM 固定为执行态工作流控制器和有来源报告生成器，由确定性工具负责信号计算，由 Case State 与 Observation 驱动逐步行动，由师兄 AbilityKG 在能力域证据存在时提供可追踪知识，而不是让 LLM 直接诊断或解释原始 EEG。
- **Why this is the smallest adequate intervention：** 不增加新算法或新抽象，只给现有模块一个明确的任务契约和一个关键桥接规则：EEG evidence 与 ability-domain evidence 分层，只有后者允许进入个性化 AbilityKG 路径。
- **Why this route is timely in the foundation-model era：** LLM 的价值位于异构工具选择、缺失信息处理、动态停止和结构化综合，而非替代成熟数值工具；这与 ReAct 的“基于环境 Observation 更新动作”和 RAG 的“外部非参数知识带来源生成”天然一致。

## Contribution Focus

- **Dominant contribution：** 一个可执行、可追踪、面向不完整个案状态的 EEG-ASD 评估工作流 Agent 系统。
- **Optional supporting contribution：** 一个诚实的 RAG 路由边界：论文知识可由 EEG/模型结果触发；个性化 AbilityKG 检索必须由真实能力域证据触发，证据不足时 Agent 请求补充量表/报告或降级为一般知识解释。
- **Explicit non-contributions：** 不主张新的 ASD 生物标志物、新的分类模型、临床效用、新 RAG 算法或通用 Agent 框架创新。

## Proposed Method

### Complexity Budget

- **Frozen / reused backbone：** 当前 `AssessmentCaseState`、增量 NextAction Planner、Closed Tool Registry、Capability Manifest、Observation/StateUpdate、Allowlist、Execution-grounded Context、ReportSpec、Renderer；现有 EEG Harness；师兄 AbilityKG 与论文导出。
- **New trainable components：** 0。当前基线模型仅作为可替换的确定性研究工具；Agent 系统不训练新模型。
- **Tempting additions intentionally not used：** Multi-Agent、自主代码执行、长链隐式思维保存、向量数据库重建、EEG foundation model 微调、自动诊断与自动干预。

### System Overview

```text
研究人员/评估人员的自然语言请求
  + Case Package(raw EEG | feature table | model result | assessment report)
  ↓
Case State：已有资产、目标、已完成步骤、质量与证据状态
  ↓
LLM Planner：每轮只选一个合法动作
  ↓
Closed Tools
  ├─ EEG evidence：inspect → quality → preprocess → epoch → features
  ├─ model evidence：readiness → simple baseline inference
  ├─ ability evidence：从已有量表/报告读取，不由 EEG 数值直接臆测
  ├─ RAG query router
  │    ├─ EEG/model result → senior paper evidence
  │    └─ ability-domain evidence → Ability→Dimension→Task evidence
  └─ human review / stop
  ↓
Observation → StateUpdate → 再次规划
  ↓
Allowlisted execution-grounded Context
  ↓
LLM 输出严格 ReportSpec → deterministic Markdown/HTML
  ↓
Assessment Dossier：结果、来源、限制、缺失信息、Trace
```

### Core Mechanism

- **Input / output：** 输入是自然语言请求与个案包；输出是“ASD 辅助评估研究档案”，包括 EEG 质量与特征摘要、受限模型结果、RAG Evidence、未完成项、人工复核项和完整执行 Trace。
- **Architecture or policy：** Planner 每轮读取用户目标、当前 Case State、上一 Observation、已完成步骤和可用 Manifest，只能输出 `run_tool`、`request_input`、`request_review` 或 `stop` 之一。Registry 执行后，Normalizer 把结果转成 Observation 并更新 State，再重新决策。
- **Representation design：** 短期工作记忆就是结构化 Case State；情景记忆保存每次运行的步骤、参数、失败与最终摘要；语义记忆由师兄 AbilityKG 和论文索引承担。原始 EEG、完整特征矩阵和绝对路径不进入 LLM Memory。
- **Training signal / loss：** 无新训练。Planner 与 Reporter 通过 JSON Schema、合法动作候选集、工具前置条件和确定性回退约束；可在后续用固定场景验证行为，不将该验证包装成模型性能创新。
- **Why this is the main contribution：** 核心不是又加一个 Planner，而是把“EEG 证据生产—ASD 个案状态—能力知识检索—可追踪报告”定义成一个可执行任务，使 Agent 的动态决策具有必要性。

### Optional Supporting Component

- **Evidence Router input / output：** 输入是当前已执行证据类型和用户目标；输出是 `paper_evidence_query`、`ability_graph_query`、`ask_for_ability_evidence` 或 `no_rag_needed`。
- **Rule：** 只有模型结果时，允许检索与 EEG/ASD 方法和限制有关的论文；只有真实量表、报告或能力档案明确给出能力域时，才允许将能力名称送入师兄 AbilityKG；不得从 alpha/theta 等频带数值直接推断“社交能力弱”。
- **Why it does not create contribution sprawl：** 它不是新的检索算法，只是把师兄 RAG 的输入契约嵌入现有 State/Manifest 前置条件。

### Modern Primitive Usage

- **LLM：** 作为逐步 Planner、缺失信息对话器和有来源 ReportSpec 生成器。
- **ReAct：** 采用 `State → Action → Tool Observation → StateUpdate` 的外部可见循环，不保存或展示自由形式隐式思维链。
- **RAG：** 作为语义长期记忆，检索师兄图谱和论文语料，返回稳定 Evidence ID；生成器只可引用检索实际返回的证据。
- **为什么比固定 Pipeline 自然：** 个案输入可能从 raw EEG、特征表、已有模型结果或报告任意阶段开始；质量失败、证据缺失和用户只要求单项检查时，下一步和停止点不同。

### Integration into Downstream Pipeline

现有 V1 代码保持为主干，只需围绕产品任务收敛接口：

1. Case State 增加或明确 `evidence_profile`，区分 `eeg_signal`、`model_result`、`ability_domain` 与 `retrieved_knowledge`。
2. RAG Query Builder 按 Evidence Router 选择论文通道或 AbilityKG 通道。
3. ReportSpec 固定分为 `本次执行发现`、`知识解释`、`证据缺口`、`限制与下一步`，两类内容不得混写。
4. Memory 先用 JSON/SQLite 保存脱敏摘要、工具版本和 Evidence ID，不引入新服务。

### Training Plan

当前阶段不训练 Agent。简单基线模型冻结为工具占位；重点使用确定性场景和少量真实 Provider 运行验证系统行为。若以后获得可信 subject-level 数据，再单独替换模型工具，不改变 Agent 契约。

### Inference Path

以“分析这个 EEG，完成 ASD 辅助评估并解释结果”为例：

1. Agent 发现 raw EEG，检查质量；若失败则停止并请求重新采集。
2. 质量通过后执行预处理、epoch 和特征提取。
3. Readiness 通过才调用简单基线；否则保留 EEG 结果并报告模型不可用。
4. 若只有 EEG/model evidence，动态构造论文检索 Query，检索相关 ASD-EEG 研究和限制。
5. 若同时有量表/报告形成 ability-domain evidence，再用弱能力名称检索 AbilityKG 的 Dimension/Task；没有就明确提示“无法个性化能力图谱检索”。
6. Reporter 仅接收 Allowlist 后的摘要和 Evidence，输出严格 JSON；Renderer 生成最终档案。
7. Memory 保存该次运行的脱敏摘要，后续请求可继续“补量表后重新生成”，不重复 EEG 处理。

### Failure Modes and Diagnostics

- **EEG 质量差：** 由质量工具阈值检测；停止下游模型并输出重新采集建议。
- **模型输入不兼容：** Readiness 阻断；不做伪推理，报告只保留质量/特征结果。
- **能力证据缺失：** RAG Router 禁止 AbilityKG 个性化路径；降级为论文知识或请求补充量表/报告。
- **师兄 Neo4j/Spark 不可用：** 遍历同一图谱 CSV 导出并使用本地论文 BM25，明确运行模式。
- **LLM 动作或报告 JSON 非法：** 一次结构修复，仍失败时由确定性 Planner/Reporter 接管。
- **LLM 编造数值或来源：** ReportSpec 校验数值/ID 是否来自 Allowlist Context；不合法即拒绝或降级。

### Novelty and Elegance Argument

该方案当前首先是一个工程型毕设，而不是追求顶会算法新颖性的论文。它的价值在于给通用 Agent Harness 一个领域内真实且有状态的任务：在异构、不完整、可能失败的 EEG-ASD 个案上编排工具和知识，并把师兄 RAG 放在有合法输入证据的位置。相比“EEG 分类后让 LLM 写报告”，它避免了从信号特征到能力缺陷的无依据跨越；相比固定流程，它通过 Observation 驱动的跳过、停止、补充输入和 RAG 路由体现 Agent 必要性。

## Claim-Driven Validation Sketch

### Claim 1：Agent 能根据不同个案状态正确编排同一套 EEG-ASD 工具

- **Minimal experiment：** raw EEG、feature table、existing model result、quality-only、bad-quality EEG、补充能力报告后二次继续，共 6 个固定场景。
- **Baselines / ablations：** 固定 Pipeline 与确定性规则 Planner；真实 LLM Planner 只作为功能展示，不先主张效果优越。
- **Metric：** 必需工具覆盖率、禁止调用率、正确跳过率、正确停止率、最终任务完成率。
- **Expected evidence：** 所有场景都只调用当前状态允许且任务需要的工具；至少一条运行因 Observation 改变后续动作。

### Claim 2：证据路由能让师兄 RAG 在不越过 EEG 证据边界的前提下进入报告闭环

- **Minimal experiment：** 同一个案分别提供“仅 EEG/model result”和“EEG + ability report”，比较检索通道和最终报告。
- **Baselines / ablations：** 不分证据类型、直接用所有结果构造 AbilityKG Query 的 naive RAG。
- **Metric：** Evidence ID 覆盖率、无依据能力映射次数、检索通道正确率、报告中个案发现与外部知识混淆次数。
- **Expected evidence：** 仅 EEG 时不产生个性化能力缺陷或训练任务；提供能力证据后，报告能追踪到对应 Ability/Dimension/Task Evidence ID。

## Experiment Handoff Inputs

- **Must-prove claims：** Agent 的状态适应性；师兄 RAG 的合法证据路由与可追踪生成。
- **Must-run ablations：** 固定 Pipeline；无 Evidence Router 的 naive RAG。
- **Critical datasets / metrics：** 现有 synthetic/fixture 场景用于软件正确性，至少补一个真实去标识 EEG 或真实特征案例用于演示；结果不用于临床性能主张。
- **Highest-risk assumptions：** 师兄 ability profile 与现有 EEG/model 输出之间是否有可验证映射；真实 Provider 是否稳定遵守动作和报告 Schema。

## Compute & Timeline Estimate

- **Estimated GPU-hours：** 当前系统搭建 0 GPU 小时；真实 LLM 通过 API；基线模型本地 CPU 推理。
- **Data / annotation cost：** 主要成本是确认师兄各能力域字段的来源与语义，并准备 6–10 个可人工核对的个案场景。
- **Timeline：** 1 周收敛系统任务和证据接口，1 周补 Memory/继续运行与真实 Provider，1 周做演示、错误场景和文档；后续再决定是否发展成研究创新。
