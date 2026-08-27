# 研究方案：EEG-ASD 个案工作流助手

## 一句话定位

本系统不是“用 EEG 自动诊断 ASD”，而是一个服务 EEG/ASD 研究人员的个案工作流 Agent：它接收自然语言目标和不同完整度的个案资料，判断当前能做什么，执行 EEG/模型/RAG 工具，在证据不足时暂停或有限完成，并在补充资料后从原状态继续生成可追踪的非诊断性评估档案。

## 为什么这个任务需要 Agent

单次分类器只能回答“一张特征表得到什么输出”，固定线性 Pipeline 只能从头顺序运行。实际个案可能从 raw EEG、已有特征、已有模型结果或既有报告任意阶段开始；质量可能失败；用户可能只要求质量检查；师兄 AbilityKG 又只有在能力域证据存在时才能合法调用。

条件工作流可以处理结构化分支，但自然语言目标解释、资料缺失澄清和非结构化证据交互仍需要语言模型。Agent 的任务因此是管理“目标—状态—合法动作—Observation—继续/停止”，而不是让 LLM 计算 EEG 或自主诊断。

## 用户、输入与输出

- 目标用户：EEG/ASD 研究人员、数据分析人员、评估助理。
- 输入：自然语言请求，以及 raw EEG、feature table、model result、ability profile、assessment report 的任意组合。
- 核心任务：将异构个案资料组织成一条可执行、可停止、可恢复的辅助评估工作流。
- 输出：完整或有限的 Assessment Dossier、单项任务结果，或等待补充资料的 resumable checkpoint。

## 核心控制结构

~~~text
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
~~~

控制权分配如下：

1. 确定性策略根据 Manifest、Case State 和前置条件计算合法动作、固定优先级、完成谓词和终态。
2. LLM Semantic Controller 只把自然语言转成受控 GoalSpec，或在多个合法动作且选择确实取决于用户语义时选一个 action ID。
3. 确定性工具和 StateUpdate 负责 EEG 数值、模型调用、错误传播、缓存失效和停止。
4. LLM Reporter 只把 Allowlist Context 转成严格 ReportSpec；校验和 Markdown/HTML 渲染确定性。

## Goal 与终态

Goal 使用受控枚举：quality_check、feature_extraction、asd_research_assessment、explain_existing_result、knowledge_lookup、resume_case。

终态包括 completed、completed_with_gaps、blocked_waiting_input、stopped_quality_failure、stopped_model_unsupported、user_cancelled。

语义分支失败时不任意回退：schema 非法、action 不在候选集、与 GoalSpec 冲突或 Provider 返回 needs_clarification 时，Agent 请求澄清或人工复核。

## EEG 与模型的位置

EEG 工具链保持确定性：

~~~text
inspect → quality → preprocess → events → epochs → features
~~~

简单模型只是可替换工具：readiness 通过才执行；旧模型契约不可信时明确返回 unsupported 或生成不含当前个案推理的有限档案。当前项目不以模型准确率、新 EEG 算法或临床有效性为主线。

## 师兄 RAG 如何真正进入闭环

师兄 RAG 的真实语义是由能力域进入 Ability → Dimension → Task 图谱，并联合论文语料。系统因此区分 eeg_signal_evidence、model_evidence、ability_evidence、knowledge_evidence。

路由规则：

- EEG/model evidence 只构造不含个体数值的论文 Query，例如特征族、研究任务、模型类型和限制。
- 只有人工确认且 KG alias 已匹配的 ability evidence 才能进入 AbilityKG。
- EEG 频带或模型分数不得直接推出“社交能力弱”“沟通能力弱”等结论。
- AbilityKG 的 Task 是外部知识候选项，需专业人员复核，不是个性化处方。
- Neo4j 可用时走师兄原 Cypher；不可用时遍历同一只读 CSV 图谱导出，论文通道使用本地 BM25；运行模式和 Evidence ID 必须记录。

V1 可以先只支持人工录入的结构化 ability profile，非结构化报告自动抽取后置。

## Memory

不建设复杂 Memory 系统。一个 SQLite 文件即可包含：

- cases：原始 GoalSpec、Case State、terminal status、pending actions、resume scope；
- artifacts：hash、schema/tool version、derived_from、stale；
- observations：白名单执行摘要；
- evidence：ability/knowledge 来源和确认状态。

补充新 ability evidence 后，只让 AbilityKG 与旧报告变为 stale；未改变的 EEG/模型 artifact 复用。这就是 Agent 的 working/episodic memory；师兄 KG 和论文库承担 semantic knowledge source。

## 最值得展示的主场景

第一次用户只提供 raw EEG。Agent 完成质量、预处理、特征、可用的简单模型和论文 RAG，因为缺少能力档案而输出 completed_with_gaps + limited_dossier。

随后用户补充一份人工确认的能力档案并说“继续这个个案”。Agent 从 CaseStore 恢复，只执行新解锁的 AbilityKG 检索和 stale report，不重复 EEG/模型。

这条路径同时让 Context、工具、Observation 循环、RAG、Memory、暂停恢复和报告边界都具有真实作用。

## 毕设贡献与边界

建议固定答辩表述：

> 本工作设计并实现了一个面向不完整 EEG-ASD 个案、按证据类型门控工具和知识路径、支持暂停与增量恢复的工作流 Agent 系统；它不是新的 ASD 分类算法，也不是临床诊断系统。

Claim–Evidence 不作为论文主线。报告只保留最小来源字段：finding type、source refs 和 assertion strength，用于区分本次执行结果、模型估计与外部知识。

## 最小验证

1. 状态控制：fixed linear Pipeline 对比 deterministic eligibility，验证 raw EEG、features、existing result、quality-only、bad quality 和 resume 六个场景的跳过、停止和恢复。
2. 语义控制：static priority 对比 Semantic Controller，只测试真正有歧义的用户请求；若 LLM 无增益，就不主张 LLM Planner 必要。
3. RAG 门控：无 evidence typing 对比完整 gate，验证无依据 AbilityKG 调用、alias 确认和 Evidence ID 覆盖。

## 实施顺序

1. 在现有 V1 上补 EvidenceProfile、Goal completion、terminal status 与 goal × action 静态优先级。
2. 实现 SQLite CaseStore、artifact hash/idempotency 和 resume scope。
3. 将师兄 RAG 拆成 paper route 与 confirmed AbilityKG route，跑通中心 resume 演示。
4. 用真实 Provider 验证两个语义歧义场景，并保留 deterministic/offline fallback。
5. 最后补一个去标识真实案例用于系统演示，不作临床性能主张。

