# EEG-ASD Workflow Agent

一个面向硕士论文实验的能力感知 EEG-ASD 工作流 Agent。它不是临床诊断系统。

核心路径：

```text
自然语言请求
  → AssessmentCaseState
  → Deterministic/LLM Planner
  → Closed Tool Registry
  → Workflow Executor
  → ObservationNormalizer
  → deny-by-default AllowlistProjector
  → execution-grounded ContextBuilder
  → FinalAssessmentReportSpec
  → Markdown/HTML Renderer
```

## 它接收什么

- 合成或外部 EEG 引用；
- 已有特征表；
- 既有模型结果；
- 既有科研报告；
- 仅知识检索或干预知识请求。

系统先读取 Case State，再决定调用哪些注册工具。它不会让 LLM 临时编写 EEG
分析代码，也不会开放 Shell、文件写入或任意工具调用。

## Tool、Allowlist 与 Context

- Tool 是具有版本化 `ToolCapabilityManifest` 的确定性或受限 Adapter。
- ToolResult 保存完整执行结果；Observation 是规范化事实；LLMVisibleToolResult
  是按工作流白名单投影后的结果，三者类型严格分离。
- Allowlist 采用 deny-by-default；新增字段不会自动进入 LLM。
- Context 分为 Planning、Tool Selection 和 Report Generation 三类。
- 原始信号、样本数组、本地路径、密钥和完整内部模型输出永不进入报告 Payload。

## 安装

本机系统 `python3` 为 3.9；项目使用 Codex Runtime 提供的 Python 3.12 创建独立环境：

```bash
cd /Users/nick/programs/eeg-system/eeg-asd-workflow-agent
/Users/nick/programs/eeg-system/bishe/.venv/bin/python -m venv .venv
.venv/bin/python -m pip install -e '.[dev,eeg]'
```

`bishe/.venv/bin/python` 只用于定位 Python 3.12 Runtime，新环境不复用旧依赖。

## 快速运行

```bash
.venv/bin/python -m eeg_asd_agent.cli tools list
.venv/bin/python -m eeg_asd_agent.cli tools describe run_asd_classifier
.venv/bin/python -m eeg_asd_agent.cli profile show

.venv/bin/python -m eeg_asd_agent.cli plan \
  --case demo_cases/04_feature_table_full_assessment.json \
  --request "根据现有特征完成 ASD 辅助评估并生成报告" \
  --offline

.venv/bin/python -m eeg_asd_agent.cli run \
  --case demo_cases/01_raw_eeg_full_assessment.json \
  --request "分析这个 EEG，完成 ASD 辅助评估，并检索相关知识生成报告" \
  --offline \
  --output-dir artifacts/demo

.venv/bin/python -m eeg_asd_agent.cli benchmark all \
  --offline \
  --output-dir artifacts/benchmarks

.venv/bin/python -m eeg_asd_agent.cli trace show artifacts/demo/trace.jsonl
.venv/bin/python -m eeg_asd_agent.cli trace html artifacts/demo/trace.jsonl \
  --output artifacts/demo/trace-cli.html
.venv/bin/python -m eeg_asd_agent.cli dashboard build \
  --artifact-root artifacts --output artifacts/dashboard.html
.venv/bin/python -m eeg_asd_agent.cli audit-integrity
```

## 可选真实 Provider

默认永远使用 OfflineProvider。只有显式关闭 offline 并同时提供环境变量时才会调用
OpenAI-compatible Responses API：

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="<pinned-model-id>"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # 可选

.venv/bin/python -m eeg_asd_agent.cli run \
  --case demo_cases/04_feature_table_full_assessment.json \
  --request "根据现有特征完成 ASD 辅助评估并生成报告" \
  --no-offline --planner llm \
  --output-dir artifacts/provider-run
```

Provider 使用严格 JSON Schema Structured Outputs；密钥只从环境读取，不写入 Trace、
Artifact 或仓库。未经真实 Provider 运行的结果不会被标成真实 LLM 实验。

## Demo

1. `01_raw_eeg_full_assessment.json`：合成 EEG → Welch 特征 → fixture 模型 → KG → 报告。
2. `04_feature_table_full_assessment.json`：师兄特征 → legacy OOF replay → 离线 KG。
3. `06_existing_medical_report.json`：已有报告 → 跳过 EEG/模型 → 只做知识检索。

合成、fixture、legacy 和 restricted 状态都会写入 Manifest、ToolResult 和报告。

## 与旧项目的区别

- 固定 Pipeline 改为 Case State 驱动的 Planner；
- 工具调用改为封闭 Registry，不允许发明工具；
- LLM 不再读取完整 ToolResult；
- 报告先生成严格 JSON，再由确定性 Renderer 输出；
- 完整性审计保留为独立命令，不再作为研究主线。

师兄项目只提供受限 OOF 历史回放与离线 KG 数据；paicli 只贡献 Registry、DAG、
预算和停滞检测的设计思想，没有 Java Runtime 依赖。

## 质量门

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/mypy eeg_asd_agent
.venv/bin/python -m compileall -q eeg_asd_agent
```

Offline Benchmark 必须显示 `evaluation_mode=fixture` 和
`scientific_claim_allowed=false`，不能被写成真实 LLM 效果结论。
