# EEG-ASD Workflow Agent

这是一个用于硕士论文研究的、非临床的增量状态驱动 Agent。当前核心路径接收自然语言
请求以及 EEG 特征表、既有模型结果或既有报告；它每轮读取最新
`AssessmentCaseState`，从封闭 Registry 选择一个下一步工具，执行后用显式
`StateUpdate` 写回 Observation，再决定继续或停止。

```text
User request + Case State
  → NextActionPlanner.decide(one action)
  → closed ToolRegistry + Manifest preconditions
  → ToolExecutionOutcome(ToolResult, StateUpdate)
  → ObservationNormalizer + StateReducer
  → decide again / stop
  → recursive deny-by-default Allowlist
  → execution-grounded report Context
  → strict FinalAssessmentReportSpec JSON
  → deterministic Markdown/HTML Renderer
```

LLM/离线报告 Provider 只能看到：用户请求、已执行工作流、白名单结果、对应限制、缺失
信息、ReportSpec JSON Schema 和非诊断规则。完整特征表、原始 EEG、绝对路径、错误栈、
密钥、完整 ToolResult 和未声明字段不会进入 Payload。Provider 只能输出合法
`FinalAssessmentReportSpec` JSON；Renderer 不调用 Provider，也不能添加新事实。

## 当前核心 Registry

| Tool | 作用 | 真实性 |
|---|---|---|
| `inspect_case` | 检查输入类型与可用资产 | REAL |
| `check_data_quality` | 确定性特征有限值检查 | REAL |
| `check_model_readiness` | 检查 replay/模型是否可执行 | REAL |
| `inspect_existing_model_result` | 校验并投影既有结果，不重跑模型 | ADAPTER |
| `research_replay_classifier` | 师兄历史 OOF 行回放 | REPLAY |
| `retrieve_asd_knowledge` | 师兄导出 KG/论文的本地词法索引 | OFFLINE |
| `generate_report_spec` | 严格 JSON 报告；默认离线结构生成 | OFFLINE |
| `render_report` | 确定性 Markdown Renderer | REAL |
| `stop_with_insufficient_input` | 输入不足时显式停止 | REAL |
| `stop_with_unsupported_request` | 临床确诊/未注册能力请求停止 | REAL |

没有真实 Provider 时使用明确命名的 `OfflineReportGenerator`。它只验证编排、白名单、
Schema 和降级流程，不能证明真实 LLM 效果。师兄 live Neo4j/Spark RAG 已实际探测但当前
不可运行；`SeniorKnowledgeAdapter` 始终返回 `mode=offline_local_index`。核心 Registry
没有 FAKE 模型，也没有把 OOF replay 写成新个体推理。

## 三个核心 Case

```bash
cd /Users/nick/programs/eeg-system/eeg-asd-workflow-agent

.venv/bin/python -m eeg_asd_agent.cli run \
  --case demo_cases/core_a_feature_assessment.json \
  --request "根据现有特征完成 ASD 辅助评估并生成报告。" \
  --offline --output-dir artifacts/core-a

.venv/bin/python -m eeg_asd_agent.cli run \
  --case demo_cases/core_b_existing_model.json \
  --request "只解释当前模型结果并检索相关知识。" \
  --offline --output-dir artifacts/core-b

.venv/bin/python -m eeg_asd_agent.cli run \
  --case demo_cases/core_c_readiness_only.json \
  --request "只检查数据和模型是否具备执行条件，不生成完整报告。" \
  --offline --output-dir artifacts/core-c
```

- Case A：quality → readiness → OOF replay → knowledge → ReportSpec → render。
- Case B：投影既有模型结果 → knowledge → ReportSpec → render；不重跑模型。
- Case C：quality → readiness → stop；不调用模型、RAG 或报告工具。

每次完整报告运行会生成 `plan.json`（逐轮 Action 历史）、`case_state.json`、
`tool_results/`、`observations.jsonl`、`llm_payload.json`、`report_spec.json`、
`report.md`、`report.html`、`trace.jsonl` 与运行统计。Case C 因未请求报告，不生成
ReportSpec/Renderer 产物。

## 安装

```bash
cd /Users/nick/programs/eeg-system/eeg-asd-workflow-agent
/Users/nick/programs/eeg-system/bishe/.venv/bin/python -m venv .venv
.venv/bin/python -m pip install -e '.[dev,eeg]'
```

旧 `bishe`、师兄工程和底层 EEG 工程保持只读。paicli 只提供 Registry、预算和停滞检测
的设计参考；本项目不调用其 Java Runtime、Shell、MCP 或编码工具。

## 检查工具与 Trace

```bash
.venv/bin/python -m eeg_asd_agent.cli tools list
.venv/bin/python -m eeg_asd_agent.cli tools describe research_replay_classifier
.venv/bin/python -m eeg_asd_agent.cli profile show
.venv/bin/python -m eeg_asd_agent.cli trace show artifacts/core-a/trace.jsonl
```

`plan` 子命令保留为 legacy 一次性规划预览；证明增量 Agent 行为应使用 `run` 的
`action_history` 和 `next_action_decided` Trace。

## 小规模独立规划 Benchmark

```bash
.venv/bin/python -m eeg_asd_agent.cli benchmark core-planning \
  --offline \
  --output-dir artifacts/benchmarks/core-planning
```

18 条案例比较 `fixed_pipeline`、`deterministic_state_planner` 和
`deterministic_without_manifest`。expected 字段只供独立 Evaluator 使用。当前环境无
Provider 凭据，因此 `llm_planner_with_manifest=NOT RUN`，真实 LLM 调用数为 0，也没有
Fixture 替代。离线数字只能验证实现与评测协议，不能直接支持 LLM 优越性论文结论。

## 可选真实报告 Provider

OpenAI-compatible Provider 只从环境变量读取凭据。当前核心 `run` 的下一步 Planner 仍是
确定性的；关闭 offline 只会替换结构化报告 Provider，不能写成“真实 LLM Planner 已
评测”。

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="<pinned-model-id>"

.venv/bin/python -m eeg_asd_agent.cli run \
  --case demo_cases/core_a_feature_assessment.json \
  --request "根据现有特征完成 ASD 辅助评估并生成报告。" \
  --no-offline \
  --output-dir artifacts/provider-report
```

## 质量门

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/mypy eeg_asd_agent
.venv/bin/python -m compileall -q eeg_asd_agent
```

研究边界与交接见：

- `docs/research/12_core_truth_audit.md`
- `docs/research/13_overnight_handoff.md`
- `docs/research/14_senior_rag_integration.md`
- `docs/research/15_core_planning_benchmark.md`
- `docs/research/known_limitations.md`
