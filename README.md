# EEG-ASD Full-domain Agent

这是一个面向 EEG-ASD 研究评估的专用 Agent Harness。它的核心不是把 EEG、模型、RAG、报告
预先排成固定流水线，而是让 LLM 根据用户目标、当前个案状态和刚得到的 Observation，逐轮决定：

- 当前任务应该做什么；
- 下一步调用哪个 EEG-ASD Tool；
- 已有资产应复用、检查、比较、重算还是忽略；
- 是否需要 RAG、报告或继续分析；
- 什么时候用户目标已经满足。

用户只提供一份 EEG 也可以启动完整任务。特征表、模型结果、能力评估和既有报告都是可选资产，
不是运行前提，也不是自动跳过某一步的控制信号。当前已接通 EEG 读取、质量检查、预处理、
特征提取、非诊断性 ASD 相关 EEG Profile、知识检索和报告生成；尚未实现的能力仍以
`enabled=false` 的显式 Stub 出现在能力目录中，不会伪装成功。

本系统仅用于研究与软件验证，不是临床诊断系统，不输出 ASD 确诊、排除诊断或治疗处方。

## 现在搭好的 Agent 闭环

```text
User Request + Optional Assets
  → AssessmentCaseState
  → Agent Context（脱敏状态、Observation、Memory、完整能力目录、当前合法动作）
  → LLM AgentTurnDecision（修订 Plan + 只选择一个 Next Action）
  → Harness 校验（Schema、Closed Registry、参数、硬前置条件、预算）
  → Tool 执行
  → ToolResult
  → ObservationNormalizer + StateUpdate
  → StateReducer + SQLite Memory
  → 回到 LLM 重新决策
  → Goal Satisfied / Waiting / Blocked
```

完整架构图见 [agentic-runtime-architecture.png](figures/agentic-runtime-architecture.png)，实现说明见
[Agent 架构交接](docs/research/17_agentic_architecture.md)。

## LLM 与 Harness 的职责边界

| 组件 | 负责 | 不负责 |
|---|---|---|
| LLM Agent Controller | 理解用户目标、修订计划、选择下一工具、决定是否结束 | 直接读本地文件、绕过 Registry、修改状态 |
| Agent Context Builder | 提供脱敏 Case 摘要、白名单 Observation、Memory 摘要、Manifest 和合法动作 | 给出确定性预选答案、暴露原始 EEG/特征值/绝对路径 |
| Closed Tool Registry | 注册能力、校验参数和硬前置条件、拒绝未知/禁用工具 | 根据输入类型或固定顺序替 LLM 选择工具 |
| Tool | 执行一个封闭能力并返回 `ToolResult + StateUpdate` | 直接修改 Case State、决定下一步 |
| StateReducer | 以唯一通道应用状态更新 | 推断用户意图或工作流顺序 |
| SQLite Memory | 本地保存完整 Case 与跨运行 Turn；只向 LLM 投影安全摘要 | 把原始数据或完整 ToolResult 放入 Prompt |
| Report Provider | 只基于 Allowlist Context 输出严格 ReportSpec JSON | 查看原始 EEG、完整特征矩阵、绝对路径或编造 Evidence ID |
| Renderer | 将合法 ReportSpec 确定性渲染为 Markdown/HTML | 调用 LLM 或添加新事实 |

Harness 计算的是“现在硬条件上能执行哪些动作”，不是“现在应该执行哪个动作”。正常情况下 LLM 会
同时看到多个候选，例如检查 EEG、检查已有结果、先构造 RAG Query 或生成有限报告；顺序由当前请求决定。

## AgentTurnDecision 协议

真实 LLM 每轮只能输出一个严格 JSON 对象：

```json
{
  "plan": {
    "revision": 3,
    "objective": "验证已有结果并补充知识解释",
    "strategy": "先检查既有结果，再根据 Observation 决定重算或检索",
    "steps": [],
    "completion_criteria": ["用户要求已由执行结果支持"]
  },
  "action": {
    "tool_name": "inspect_existing_model_result",
    "arguments": {},
    "reason": "用户要求先验证既有结果",
    "terminal": false,
    "based_on_observations": []
  },
  "goal_satisfied": false,
  "assistant_message": null,
  "replan_reason": "根据用户目标建立首轮计划"
}
```

它不是一次性 DAG：`plan` 是可修订的高层任务计划，`action` 每轮最多一个。工具执行后，新
Observation 会进入下一轮 Context。若 LLM 输出未知工具、非法参数、虚构 Observation ID 或违反硬前置
条件，Agent Runtime 会拒绝并阻断本次运行；不会暗中切换成固定 Pipeline。

## EEG-ASD 能力面

当前 Registry 同时表达“已实现能力”和“目标能力”。这让 LLM 能理解完整 Agent 边界，又不会调用空壳。

| 能力族 | 当前可执行能力 | 已声明但禁用的扩展点 |
|---|---|---|
| EEG 数据 | `inspect_eeg`、`check_eeg_quality`、`preprocess_eeg`、`extract_events`、`create_epochs`、`extract_eeg_features` | `analyze_eeg_patterns`、`visualize_eeg_analysis` |
| 评估与模型 | `build_eeg_asd_profile`、`check_data_quality`、`check_model_readiness`、`run_asd_baseline_model`、`inspect_existing_model_result`、`research_replay_classifier` | `estimate_asd_severity`、`estimate_ability_dimensions`、`compare_assessment_results`、`conduct_structured_assessment`、`train_asd_research_model` |
| RAG | `build_asd_rag_query`、`retrieve_asd_knowledge` | 后续可增加干预知识和更多检索器 |
| 报告 | `generate_report_spec`、`render_report` | `parse_assessment_report` |
| 控制 | `inspect_case`、`request_human_review`、受限停止工具 | — |

已实现工具仍保留 `REAL`、`ADAPTER`、`RESTRICTED_REAL`、`REPLAY`、`OFFLINE` 等真实性标签。
禁用 Stub 标记为 `STUB`，Registry 不允许执行。

## RAG 如何进入 Agent

RAG 是 Agent 可选择的能力，不是固定在模型推理之后的流水线节点：

1. LLM 可根据请求在信号分析前、分析中或分析后选择 `build_asd_rag_query`；
2. Query Tool 根据当前已执行 Observation 构造查询，不读取完整特征值；
3. `retrieve_asd_knowledge` 调用师兄 AbilityKG 导出图谱遍历和论文 BM25；若配置 live Neo4j，
   可切换到同一图谱路径的在线检索；
4. Retriever 返回带稳定 Evidence ID 的结构化证据；
5. 报告 Provider 只能引用 Allowlist Context 中真实存在的 Evidence ID。

本机当前模式是 `senior_graph_export_traversal+paper_bm25`。它读取师兄 Ability、Dimension、Task
节点与关系 CSV，并沿 `Ability → Dimension → Task` 检索；论文通道读取师兄导出的 PubMed 语料。
详细来源与降级边界见 [RAG integration](docs/research/14_senior_rag_integration.md)。

## 同一资产，不同任务，不同动作

已有结果不会自动导致“跳过模型”，有特征也不会自动决定“直接推理”。例如同一个 Case 可以提出：

```bash
# 只解释已有结果：Agent 可以先 inspect_existing_model_result
--request "解释已有模型结果，并检索相关知识"

# 验证已有结果：Agent 可以选择 check_model_readiness + run_asd_baseline_model
--request "重新运行当前研究模型，验证并比较已有结果"

# 先查知识：Agent 可以在 EEG/模型之前选择 RAG
--request "先检索与当前问题相关的 ASD 证据，再决定怎样分析 EEG"

# 单项任务：质量 Observation 后结束
--request "只检查这个 EEG 的质量，不运行模型，不检索，不生成报告"
```

这些语义由真实 LLM 决策。默认 Offline 模式只是可复现的 Manifest-driven Agent 协议模拟，不能作为
“LLM 会正确规划”的实验结论。

## 运行

```bash
cd D:\Profession\Programs\eeg-asd-agent

# 正式入口：使用 DeepSeek Responses API（模型与地址默认读取 config/llm.json）
$env:OPENAI_API_KEY="<your-key>"
D:\Profession\anaconda3\python.exe -m eeg_asd_agent.cli assess `
  --eeg "D:\path\to\recording.edf" `
  --age 8 --sex male `
  --output-dir artifacts\assessment-deepseek

# 默认：完整 Agent Runtime + Offline Agent simulation
D:\Profession\anaconda3\python.exe -m eeg_asd_agent.cli run `
  --case demo_cases\01_raw_eeg_full_assessment.json `
  --request "分析这个 EEG，完成 ASD 相关研究评估；需要时检索知识并生成报告" `
  --offline --output-dir artifacts\agent\raw

# 同一套 Runtime，使用其他 OpenAI-compatible Provider
export OPENAI_API_KEY="..."
export OPENAI_MODEL="<pinned-model-id>"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # optional
export OPENAI_TEMPERATURE="0"                       # optional
.venv/bin/python -m eeg_asd_agent.cli run \
  --case demo_cases/core_b_existing_model.json \
  --request "验证已有结果；由你决定复用、重算和检索顺序" \
  --no-offline --output-dir artifacts/agent/provider

# 固定规划器只作为测试/对照基线，不是 Agent Runtime 的隐藏回退
.venv/bin/python -m eeg_asd_agent.cli run \
  --case demo_cases/04_feature_table_full_assessment.json \
  --request "检查数据质量" \
  --planner deterministic --offline --output-dir artifacts/baseline
```

`assess` 是正式 LLM Agent 入口：未配置模型服务时会直接报错，不会降级成规则或离线模式。
`run --offline` 仅保留给自动化测试和无网络情况下的 Harness 回归验证，不是正式评估入口。

## 本地网页

DeepSeek 的非敏感参数已保存在 `config/llm.json`。为避免凭据进入 Git、日志或浏览器，网页服务在
启动时从终端安全读取 API Key，并只在该后端进程内保存：

```powershell
D:\Profession\anaconda3\python.exe -m eeg_asd_agent.web
```

看到 `DeepSeek API Key:` 后粘贴 Key（终端不会回显），随后访问 `http://127.0.0.1:8765`。页面支持
EDF、NPZ、MAT 上传，可展示 Agent 已完成的工具步骤、运行状态和最终 HTML/Markdown 报告。每次运行会在
`logs/<时间>_<run_id>/` 下保存 `trace.jsonl`、可视化 `trace.html`、逐工具结果、Agent 上下文、报告和
运行摘要；页面完成或失败后均可查看 Trace 并下载完整日志 ZIP。日志目录可在服务重启后继续按 run ID 访问。

知识检索会优先使用配置好的 ASD 专用知识导出；若这些资产不存在，则直接复用所给 EEGAgent 工程的
`RAG/chunks.pkl`。原工程的 FAISS 索引依赖未附带完整权重的 BGE-M3 模型，因此当前在相同知识块上使用
可审计的 BM25 排序，不伪装成向量检索。可通过 `EEG_AGENT_REFERENCE_ROOT` 修改参考工程位置。

## Memory 与恢复

每次 Agent 运行会生成本地 SQLite Memory。完整 Case State 只保存在本机；LLM 只看到最近 Turn 的
安全摘要。恢复后 Turn index 单调递增，每个新请求获得独立 step budget，旧记录不会被覆盖。

```bash
.venv/bin/python -m eeg_asd_agent.cli resume \
  --memory artifacts/agent/raw/agent_memory.sqlite \
  --case-id demo-01-raw-synthetic \
  --request "重新检索最新相关知识并更新报告" \
  --offline --output-dir artifacts/agent/resumed
```

## 每次运行的可审计产物

```text
plan.json                 最终 Planner 类型、全部单步决策、最终高层 Plan
agent_plan.json           最后一次 Plan 修订
agent_turns.jsonl         每轮 Plan + Action + Stop 判断
agent_contexts.json       每轮实际交给 Agent Controller 的脱敏 Context
agent_memory.sqlite       可恢复的本地 Case / Turn Memory
memory_summary.json       可公开查看的 Memory 摘要
tool_results/*.json       完整本地工具结果（不直接进入 LLM）
observations.jsonl        规范化 Observation
llm_payload.json          报告 Provider 的 Allowlist Context
report_spec.json          严格 FinalAssessmentReportSpec
report.md / report.html   确定性 Renderer 输出
trace.jsonl / trace.html  执行 Trace
run_stats.json            Provider、运行模式与耗时
```

## 安装与验收

```bash
cd /Users/nick/programs/eeg-system/eeg-asd-workflow-agent
/Users/nick/.cache/codex-runtimes/codex-primary-runtime/dependencies/python -m venv .venv
.venv/bin/python -m pip install -e '.[dev,eeg]'

.venv/bin/python -m eeg_asd_agent.cli tools list
.venv/bin/python -m eeg_asd_agent.cli tools describe build_asd_rag_query
.venv/bin/python -m eeg_asd_agent.cli profile show

.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/mypy eeg_asd_agent
.venv/bin/python -m compileall -q eeg_asd_agent
```

当前架构测试覆盖：多候选动作 Context、同资产不同目标、Observation 驱动重规划、非法动作拒绝、
raw EEG 单输入闭环、严格 Agent JSON Schema、Memory 跨运行恢复与隐私过滤。

## 当前边界

- 真实 LLM Provider 接口已接入，但本轮没有凭据运行；Offline 模式不是 LLM 效果证据。
- 合成 EEG 只验证代码路径；没有真实受试者数据与临床验证。
- 当前模型只是受限占位能力，模型质量不是本轮架构目标。
- 完整能力目录中的 Profile、能力维度、比较、结构化评估等工具仍是禁用 Stub，下一阶段逐个实现。
- 不包含 Multi-Agent、Web 前端、大模型微调或临床性能主张。

师兄工程、旧 `bishe` 和 `paicli-main` 均保持只读。paicli 只借鉴 Harness、工具注册、预算、停滞检测、
Provider 分层和 Memory 思路，不调用其 Java Runtime、Shell、MCP 或代码生成能力。
