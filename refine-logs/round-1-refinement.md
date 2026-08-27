# Round 1 Refinement

## Problem Anchor

- **Bottom-line problem：** 用户希望先做出一个类似 paicli、具备 LLM、Context、RAG、ReAct 循环、Plan-and-Execute、工具系统和 Memory 的完整 Agent，并把它落到 EEG 的 ASD 评估场景；当前最需要解决的不是算法创新，而是明确这个 Agent 到底替谁完成什么真实任务，以及 EEG 分析、师兄 RAG 和 LLM Agent 如何形成一条不虚假的闭环。
- **Must-solve bottleneck：** “EEG 分类器 + RAG + 报告”只是模块拼接，不能说明为什么需要 Agent；同时师兄 AbilityKG 的入口是能力域证据，而原始 EEG 特征不能未经验证直接映射成社交、沟通、情绪等能力缺陷。系统必须给出明确的个案任务契约，并在输入不完整、质量失败、已有中间结果和证据不够时能够跳过、停止、补充检索或请求人工输入。
- **Non-goals：** 当前阶段不追求新的 EEG 算法、模型性能创新、临床诊断、自动处方、Multi-Agent、大模型微调或把 EEG 频带特征强行解释成 ASD 行为能力。
- **Constraints：** 本机没有真实受试者 EEG；现有师兄模型存在数据划分和特征契约风险；师兄 Neo4j/Spark 服务当前不可用但图谱 CSV 与论文语料可读；开发应优先复用现有 V1 架构和师兄资产；先完成可运行系统，再谈论文创新。
- **Success condition：** 给定自然语言任务和一个包含原始 EEG、特征表、已有模型结果、评估报告中任意组合的个案包，Agent 能根据执行态决定下一步，调用封闭 EEG/模型/RAG 工具，正确跳过或停止，并输出一个带执行记录、模型边界和 Evidence ID 的非诊断性 ASD 辅助评估档案；用户能清楚说明“为什么这里必须是 Agent，而不是一条固定脚本”。

## Anchor Check

- **Original bottleneck：** 不知道 Agent 在 EEG-ASD 中替谁完成什么闭环，现有模块容易退化成固定流水线或无依据的“自动诊断”。
- **Why the revised method still addresses it：** 修订后把用户、输入、输出和控制权划分写死：系统服务 EEG/ASD 研究人员，管理异构个案资产并形成可继续的评估档案；LLM 处理自然语言和非结构化材料，确定性层控制工具资格、状态更新和安全边界。
- **Reviewer suggestions rejected as drift：** 不接受把 Claim–Evidence Ledger 重新提升为系统主线。用户已明确不希望以 Claim–Evidence 为选题；本系统只保留轻量来源字段来防止混淆，不研究 claim verification。也不为追求顶会评分引入新模型或 Multi-Agent。

## Simplicity Check

- **Dominant contribution after revision：** 一个 evidence-typed case control plane（按证据类型约束动作的个案控制平面），让同一个 Agent 能从不同个案起点继续、跳过、停止和检索。
- **Components removed or merged：** Evidence Router 合并进确定性 Eligibility Policy；三层 Memory 合并为 `CaseStore + EvidenceStore`；不再同时宣称 ReAct 和 Plan-and-Execute 两套机制，只保留一个状态—动作—观察循环。
- **Reviewer suggestions rejected as unnecessary complexity：** 不新建独立 Claim Ledger，不在 V1 自动解析所有临床量表，不建设新向量库。
- **Why the remaining mechanism is the smallest adequate route：** 当前 V1 已具备大多数运行模块，新增重点只是一组证据类型、资格规则和最小持久化，不重写架构。

## Changes Made

### 1. 明确 LLM 和确定性层的控制权

- **Reviewer said：** 现有规则已经可处理大部分场景，尚未说明 LLM 为什么必要。
- **Action：** 把 Agent 分成确定性 Eligibility Policy、LLM Semantic Planner、确定性 Executor 三段。LLM 不能创造动作，只能解析目标或在多个合法动作之间作语义选择。
- **Reasoning：** EEG 数值路径大多应该确定性执行；LLM 真正适合的是自然语言目标、非结构化报告和歧义处理。
- **Impact：** 不再主张“所有下一步都必须由 LLM 决定”，而是主张“状态化 Agent 必要，LLM 在语义决策点有价值”。

### 2. 给师兄 AbilityKG 增加合法输入契约

- **Reviewer said：** 能力证据抽取缺少字段级契约。
- **Action：** 定义 `AbilityEvidence`，V1 优先接收人工确认的结构化能力档案；LLM 从非结构化报告抽取只作为可选输入，未确认前不能解锁个性化图谱路径。
- **Reasoning：** 这是避免 EEG 频带被臆测成社交/沟通缺陷的最小门控。
- **Impact：** 师兄 RAG 被放回它真实的数据语义中，并成为可继续的闭环，而不是报告装饰。

### 3. 简化 Memory 和报告来源机制

- **Reviewer said：** 三类 Memory 可以合并；报告需要更细来源类型。
- **Action：** Memory 只保留 `CaseStore + EvidenceStore`；每条 finding 携带 `finding_type`、`source_refs` 和 `assertion_strength`，但不引入 Claim–Evidence 研究主线。
- **Reasoning：** 足以支持继续运行、缓存和来源区分，同时避免旧选题回潮。
- **Impact：** 系统更像可运行产品，论文主线仍是工作流 Agent。

## Revised Proposal

# 研究方案：EEG-ASD 个案工作流助手

## Problem Anchor

- **Bottom-line problem：** 用户希望先做出一个类似 paicli、具备 LLM、Context、RAG、ReAct 循环、Plan-and-Execute、工具系统和 Memory 的完整 Agent，并把它落到 EEG 的 ASD 评估场景；当前最需要解决的不是算法创新，而是明确这个 Agent 到底替谁完成什么真实任务，以及 EEG 分析、师兄 RAG 和 LLM Agent 如何形成一条不虚假的闭环。
- **Must-solve bottleneck：** “EEG 分类器 + RAG + 报告”只是模块拼接，不能说明为什么需要 Agent；同时师兄 AbilityKG 的入口是能力域证据，而原始 EEG 特征不能未经验证直接映射成社交、沟通、情绪等能力缺陷。系统必须给出明确的个案任务契约，并在输入不完整、质量失败、已有中间结果和证据不够时能够跳过、停止、补充检索或请求人工输入。
- **Non-goals：** 当前阶段不追求新的 EEG 算法、模型性能创新、临床诊断、自动处方、Multi-Agent、大模型微调或把 EEG 频带特征强行解释成 ASD 行为能力。
- **Constraints：** 本机没有真实受试者 EEG；现有师兄模型存在数据划分和特征契约风险；师兄 Neo4j/Spark 服务当前不可用但图谱 CSV 与论文语料可读；开发应优先复用现有 V1 架构和师兄资产；先完成可运行系统，再谈论文创新。
- **Success condition：** 给定自然语言任务和一个包含原始 EEG、特征表、已有模型结果、评估报告中任意组合的个案包，Agent 能根据执行态决定下一步，调用封闭 EEG/模型/RAG 工具，正确跳过或停止，并输出一个带执行记录、模型边界和 Evidence ID 的非诊断性 ASD 辅助评估档案；用户能清楚说明“为什么这里必须是 Agent，而不是一条固定脚本”。

## Technical Gap

单次 EEG 分类解决的是“模型对一张特征表输出什么”，而真实软件任务是“这个个案目前拥有什么、还缺什么、下一步能做什么、执行失败后怎么办、哪些结果能进入知识解释”。固定 Pipeline 无法自然覆盖 raw EEG、已有特征、已有模型结果、既有评估报告和只做质量检查等不同起点。

另一方面，师兄 AbilityKG 由个体能力短板进入 `Ability → Dimension → Task` 图谱。EEG 功率、连接性或一个受限模型分数不能直接证明某个儿童的社交、沟通或情绪能力弱。因此系统必须把 EEG evidence、model evidence、ability evidence 和 external knowledge 分开，并用证据类型决定工具资格。

## Method Thesis

- **One-sentence thesis：** 面向 EEG/ASD 研究人员构建一个按证据类型控制动作的个案工作流 Agent，使 LLM 只负责自然语言目标和非结构化材料的语义决策，确定性层负责 EEG 计算、合法动作、状态更新和边界，师兄 RAG 只在相应 evidence type 已建立时被调用。
- **Why this is the smallest adequate intervention：** 现有 V1 的 State、Registry、Manifest、Observation、Allowlist、RAG 和 ReportSpec 全部保留，只补齐 `EvidenceProfile`、资格规则和持久化 CaseStore。
- **Why this route is timely：** LLM 的优势用于处理语言歧义与异构资料，外部工具负责可验证计算，RAG 负责显式知识来源；各自处在最适合的位置。

## Contribution Focus

- **Dominant contribution：** 一个 evidence-typed case control plane，用统一的 Case State 驱动 EEG-ASD 个案的继续、跳过、停止、补输入和知识检索。
- **Supporting contribution：** 将师兄 AbilityKG 的真实输入契约嵌入控制平面，避免把 EEG 特征直接写成能力缺陷。
- **Explicit non-contributions：** 不发明 EEG 算法、分类器、RAG 算法或通用 Agent 框架，不做临床诊断。

## Proposed Method

### Product Contract

- **Target user：** EEG/ASD 研究人员、数据分析人员或评估助理，不直接面向患者自主使用。
- **Input：** 自然语言目标 + Case Package；Case Package 可包含 raw EEG、feature table、model result、结构化 ability profile 或既有评估报告。
- **Primary job：** 把异构个案资产组织成一条可执行、可停止、可继续的辅助评估工作流。
- **Output：** `Assessment Dossier`，包含本次实际执行发现、受限模型结果、检索证据、缺失信息、人工复核项、工具版本和 Trace。

### Complexity Budget

- **Reused：** 当前 V1 的 Schema、Planner 接口、Registry、Manifests、EEG Harness、简单模型工具、师兄 KG/论文 Adapter、Allowlist、ReportSpec 和 Renderer。
- **New trainable components：** 0。
- **New minimal data structures：** `EvidenceProfile`、`AbilityEvidence`、`CaseStore`。
- **Excluded：** Multi-Agent、自动代码执行、通用向量数据库、新 EEG 模型、自动干预生成、大规模 benchmark。

### System Overview

```text
User Goal + Case Package
          ↓
LLM/Rule Intent Parser：把目标变成受控 goal enum
          ↓
Case State + EvidenceProfile
          ↓
Deterministic Eligibility Policy：根据 Manifest 与状态算合法动作集合
          ↓
若唯一合法动作 → 直接执行
若多个合法动作且涉及用户语义 → LLM Semantic Planner 选择一个
          ↓
Closed Tool Registry → ToolResult → Observation → deterministic StateUpdate
          ↺
RAG gate
  ├─ EEG/model evidence → senior paper retrieval
  └─ confirmed ability evidence → Ability→Dimension→Task traversal
          ↓
Allowlisted Context → strict ReportSpec → deterministic Renderer
          ↓
CaseStore + EvidenceStore 保存脱敏状态，支持下次继续
```

### EvidenceProfile

```text
eeg_signal_evidence:
  artifact_id, quality_status, preprocessing_version, feature_schema_version

model_evidence:
  artifact_id, model_version, readiness_status, output_label, score_semantics, limitations

ability_evidence[]:
  domain_label
  source_artifact_id
  source_span
  instrument_name
  observed_score
  extraction_mode: manual | llm_extracted
  human_confirmation_status

knowledge_evidence[]:
  evidence_id, source, retrieval_mode, relevance, limitations
```

规则是：`ability_evidence.human_confirmation_status=confirmed` 才能解锁个性化 AbilityKG；未确认的 LLM 抽取只能触发 `request_human_review`。V1 可先只支持人工录入的结构化 ability profile，从而避免新增复杂文档抽取。

### Control Semantics

1. **Eligibility Policy：** 只用结构化 State 与 Manifest 前置条件，计算全部合法动作；未知工具、非法参数和不满足前置条件的动作根本不进入 LLM 候选集合。
2. **LLM Semantic Planner：** 读取脱敏目标、Case 摘要、上一 Observation 和合法候选；只有在多个合法动作都可执行且用户语义会影响优先级时选择。数值处理顺序无歧义时直接走确定性动作。
3. **Executor/StateUpdate：** ToolResult 规范化、失败传播、缓存失效、状态更新、停止条件完全确定性。
4. **Reporter：** LLM 只把 Allowlist Context 转成严格 ReportSpec；校验和 Renderer 确定性。

### RAG Integration

- **论文通道：** EEG 质量、特征类别、模型模式和限制构造 Query，检索师兄 paper corpus；只用于解释研究背景和限制。
- **AbilityKG 通道：** confirmed ability domain 作为节点名/别名，遍历 `Ability → Dimension → Task`；Task 输出标记为“外部知识候选项，需专业人员复核”，不写成处方。
- **Fallback：** Neo4j 可用时走同一 Cypher；不可用时遍历只读 CSV graph export；论文通道使用本地 BM25。运行模式写入 Evidence。
- **Traceability：** 每个外部知识项必须有 Evidence ID；报告不得引用未检索 ID。

### Memory

- **CaseStore：** 持久化脱敏 Case State、用户目标、已执行工具、参数版本、Observation 摘要和待办；它同时承担 working/episodic memory。
- **EvidenceStore：** 持久化 artifact hash、schema/tool version、来源关系与 Evidence ID；师兄 KG/论文库继续作为 semantic knowledge source。
- **Resume：** 用户补充 ability profile 后，Agent 从原 Case State 继续，只使依赖该输入的旧 RAG/report 结果失效，不重复 EEG 处理。

### Report Contract

最终报告仍由 `FinalAssessmentReportSpec` 表达。每条 finding 增加最小来源类型，而不建立独立 Claim–Evidence 子系统：

```text
finding_type: case_observation | model_output | external_knowledge
source_refs: [tool_run_id or evidence_id]
assertion_strength: observed | model_estimated | literature_reported
text: bounded explanation
```

报告固定分成：本次执行发现、模型结果与含义、外部知识、证据缺口、限制与人工复核。LLM 不能看到 raw EEG、完整 feature table、绝对路径和内部 ToolResult。

### Inference Examples

**A. raw EEG 全评估**

`inspect → quality → preprocess → epochs/features → readiness → baseline → paper RAG → report`。如果没有能力档案，报告明确说明不能执行个性化 AbilityKG。

**B. feature table**

跳过全部 EEG 信号工具，从 feature quality/readiness 开始。

**C. existing model result**

跳过 EEG 与模型，直接构造论文 Query；如果同时有 confirmed ability profile，再走 AbilityKG。

**D. quality only**

质量 Observation 后停止，不调用模型、RAG 和报告。

**E. bad-quality EEG**

质量失败后合法动作变为请求重新采集/人工复核/停止；模型与 RAG 被门控。

**F. resume after new evidence**

第一次只有 EEG，生成有限报告；第二次补充结构化能力档案后，从 CaseStore 恢复，新增 AbilityKG 检索并重生成报告，不重复 EEG 工具链。

### Failure Modes and Diagnostics

- EEG 质量失败：停止模型路径，记录重新采集要求。
- 模型契约不可信：`readiness=unsupported_legacy_model`，模型工具仅作为可替换占位，不强行输出。
- 能力证据缺失/未确认：禁止 AbilityKG 个性化路径。
- RAG 服务不可用：CSV traversal + BM25 fallback，mode 显式记录。
- LLM Planner 非法：动作候选校验失败后由 deterministic policy 接管。
- LLM ReportSpec 非法：一次修复，仍失败生成有限模板报告。
- 上游 artifact 改变：按 hash 和依赖关系使下游结果失效。

### Novelty and Elegance Argument

本阶段目标是工程型毕设系统，不包装成顶会新方法。其完整性来自一个明确且真实的领域任务：研究人员给出不同完整度的 EEG-ASD 个案包，系统负责合法地继续分析、停止、补证据和生成档案。LLM 的价值被限制在目标理解、非结构化证据和歧义选择；确定性层控制数值、动作资格和安全。师兄 RAG 不再是报告末尾的文本装饰，而是由证据类型门控的领域工具。

## Claim-Driven Validation Sketch

### Claim 1：状态化 Agent 能正确处理不同起点与 Observation 分支

- **Minimal experiment：** 上述 A–F 六场景。
- **Baselines：** fixed Pipeline、deterministic policy、LLM semantic planner。
- **Metric：** required coverage、forbidden invocation、skip/stop/resume accuracy、task completion。
- **Decisive check：** 同一个案不同用户目标；同一目标在质量 pass/fail Observation 下产生不同动作。

### Claim 2：证据类型门控能让师兄 RAG 形成合法闭环

- **Minimal experiment：** 仅 EEG/model、EEG+未确认能力抽取、EEG+confirmed ability profile 三种状态。
- **Ablation：** 无门控、直接从 EEG/model 拼 Ability Query。
- **Metric：** RAG route accuracy、unsupported ability mappings、Evidence ID coverage、external/case finding confusion。
- **Decisive check：** 只有 confirmed ability evidence 才产生 AbilityKG 路径，其他状态只检索论文或请求复核。

## Experiment Handoff Inputs

- **Must-prove：** 状态/Observation 导致的真实动作变化；AbilityKG 的证据入口没有被 EEG 结果偷换。
- **Must-run ablations：** fixed Pipeline、无 evidence typing 的 naive RAG。
- **Critical assets：** synthetic/raw NPZ 只验证软件路径；师兄 graph export 与 paper corpus 验证 RAG；至少一个去标识真实案例用于最终演示。
- **Highest-risk assumptions：** 能否取得语义明确的真实 ability profile；真实 Provider 在多候选动作上的稳定性。

## Compute & Timeline Estimate

- **GPU：** 0 GPU 小时；模型工具和检索 CPU 可运行；LLM 使用配置化 API。
- **Data cost：** 主要是核实师兄 ability profile 的字段语义并准备少量人工确认案例。
- **Timeline：** 先用 1 周完成 EvidenceProfile/资格规则/CaseStore，1 周打通真实 Provider 与 resume，1 周完成真实案例、演示和文档。模型性能和创新后置。
