# Progress

## 2026-08-04 — senior RAG correction

- [x] corrected the earlier BM25-only integration: it is no longer described as the senior RAG
- [x] dynamic Query selects senior Ability nodes and traverses Ability→Dimension→Task relationships
- [x] live Neo4j path uses environment-only configuration; current closed port falls back to the same CSV graph export
- [x] senior PubMed export remains a second BM25 evidence channel
- [x] stable `SENIOR-KG-*` / `SENIOR-PAPER-*` Evidence IDs reach Allowlist report Context
- [x] report validation rejects invented Evidence IDs after one repair attempt
- [x] actual existing-model E2E produced Query → 2 graph evidences + 3 paper evidences → ReportSpec citations
- [x] LLM Planner candidates now come from Manifest preconditions, not a deterministic singleton
- [x] true Provider payload no longer contains the deterministic fallback action
- [x] scripted test demonstrates a legal non-fallback action can be selected
- [x] targeted and full gates: 77 pytest, ruff, mypy, compileall passed

## 2026-08-04 — V1 runnable workflow completion

- [x] synthetic / NPZ / MAT / equal-rate EDF inspection and deterministic EEG chain
- [x] `check_eeg_quality` observation-driven stop gate
- [x] current-case NumPy inference from senior `best_mlp.pt`, with hash/version and hard limitations
- [x] open-source model audit; no incompatible foundation model was misused as an ASD classifier
- [x] dynamic `build_asd_rag_query` and BM25 Evidence contract
- [x] per-observation LLM `ActionDecision` Provider with deterministic fallback
- [x] allowlisted ReportSpec generation, one Provider repair, deterministic limited fallback
- [x] four required E2E paths plus blocked-quality branch
- [x] README, V1 profile and implementation handoff updated
- [ ] real Provider run (credentials unavailable; Offline mode is not efficacy evidence)
- [ ] real participant EEG / subject-level clinical validation (data unavailable)

## 2026-07-29 — Phase 4: final audit and handoff

- [x] README 第一屏改为当前 incremental state-driven 核心路径
- [x] README 列出核心工具、真实性、Allowlist、Context、ReportSpec、Renderer 与三个 Case
- [x] 生成 `docs/research/13_overnight_handoff.md`，覆盖要求的 15 项
- [x] 重跑 Case A/B/C；调用顺序与停止行为符合请求
- [x] 重跑 18-case core planning Benchmark；真实 LLM 保持 `NOT RUN`
- [x] Payload 扫描未发现绝对路径、API Key、raw signal 或完整特征矩阵
- [x] Case C 输出目录清除旧报告产物，不产生 ReportSpec/Markdown
- [x] 最终 62 pytest、ruff、mypy、compileall、`git diff --check` 全部通过
- [x] Phase 4 硬门槛通过

最终验收没有新增前端、Multi-Agent、模型训练、真实临床主张或旧服务耦合。

## 2026-07-29 — Phase 3: senior knowledge adapter + core planning benchmark

### Phase 3A

- [x] 定位师兄 Neo4j/Cypher 检索入口、外部生成入口、KG CSV 与论文 CSV
- [x] 实际探测 live runtime：环境缺少 `neo4j`，且 `127.0.0.1:7687` 未监听
- [x] 未复制/使用硬编码凭据，未修改师兄只读工程
- [x] 实现 `SeniorKnowledgeAdapter.retrieve()` → `StructuredKnowledgeResult`
- [x] fallback 明确标记 `mode=offline_local_index`
- [x] 来源、无结果降级、核心 Agent E2E 测试通过

### Phase 3C

- [x] 建立 18 条独立规划案例
- [x] expected 字段只由 `PlanningEvaluator` 读取，不进入 Planner 输入
- [x] 比较 `fixed_pipeline`、`deterministic_state_planner`、
  `deterministic_without_manifest`
- [x] 真实 LLM Planner 标记 `NOT RUN`，调用数 0，未使用 Fixture 替代
- [x] 输出结果、逐案例 JSONL、错误切片 CSV、配置快照和 Markdown 摘要
- [x] Phase 3 硬门槛通过

核心离线结果：

| Variant | Required coverage | Forbidden rate | Order | Stop | Precision |
|---|---:|---:|---:|---:|---:|
| fixed_pipeline | 0.8611 | 0.8727 | 0.9444 | 0.6667 | 0.4722 |
| deterministic_state_planner | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| deterministic_without_manifest | 1.0000 | 0.0182 | 1.0000 | 1.0000 | 0.9921 |

执行命令：

```bash
.venv/bin/python -m eeg_asd_agent.cli benchmark core-planning --offline --output-dir artifacts/benchmarks/core-planning
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/mypy eeg_asd_agent
.venv/bin/python -m compileall -q eeg_asd_agent
```

失败与修复：

- live RAG 不能运行：不仅缺少 Python driver，本机 Neo4j 服务也不存在；没有把“安装依赖”
  当作接入成功，而是记录准确阻断并建立同接口本地索引。
- Adapter 首次单测暴露 `_read` 只存在于旧类；补为 Adapter 自己的只读 CSV 方法后通过。
- 首轮 Benchmark 的 knowledge-only 请求因“只 + 报告”被误判为 report-only；IntentParser
  增加检索否定判断，并保持多目标完整评估优先级，修复后 state planner 18/18 全覆盖。
- 上述数字来自人工设计的确定性离线案例，只能验证实现与评测协议，不能直接写成真实 LLM
  或临床效果结论。

## 2026-07-29 — Phase 2: incremental core workflow

- [x] `AssessmentCaseState` 统一为 `input_type`、`failed_workflows`、`pending_requirements`、`current_action`、`step_count`、`execution_status`
- [x] 核心工具只能返回 `ToolExecutionOutcome(result, state_update)`；`StateReducer` 是唯一状态写入点
- [x] `NextActionPlanner` 每轮只选择一个工具，并读取当前状态、Observation、Manifest 与 precondition
- [x] 建立 10 个核心工具的封闭 Registry；模型回放标记 `REPLAY`，知识检索与离线报告标记 `OFFLINE`
- [x] 实现递归字段级 deny-by-default Allowlist，知识条目的嵌套字段逐项声明
- [x] 报告 Context 只含请求、已执行工作流、白名单结果、工具限制、缺失信息、ReportSpec Schema 和非诊断规则
- [x] `generate_report_spec` 与 `render_report` 成为 Agent 可选择工具；Renderer 不调用 Provider
- [x] Case A/B/C 全部运行，且 Case C 不调用 RAG 或报告工具
- [x] 57 个 pytest、ruff、mypy、compileall 全部通过
- [x] Phase 2 硬门槛通过

动态决策证据：

- Case A 的第三个 Observation 将 `model_readiness` 从 `unknown` 更新为 `restricted`。
- Planner 在下一轮才选择 `research_replay_classifier`；该 Action 的
  `based_on_observations` 包含前三个 Observation ID。
- Case C 的 readiness Observation 为 `blocked`，Planner 随后直接停止，不选择模型、
  RAG 或报告工具。

执行命令：

```bash
.venv/bin/python -m eeg_asd_agent.cli run --case demo_cases/core_a_feature_assessment.json --request "根据现有特征完成 ASD 辅助评估并生成报告。" --offline --output-dir artifacts/core-a
.venv/bin/python -m eeg_asd_agent.cli run --case demo_cases/core_b_existing_model.json --request "只解释当前模型结果并检索相关知识。" --offline --output-dir artifacts/core-b
.venv/bin/python -m eeg_asd_agent.cli run --case demo_cases/core_c_readiness_only.json --request "只检查数据和模型是否具备执行条件，不生成完整报告。" --offline --output-dir artifacts/core-c
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/mypy eeg_asd_agent
.venv/bin/python -m compileall -q eeg_asd_agent
```

失败与修复：

- 首轮旧测试有 10 项失败：旧测试仍要求旧 ReportSpec、质量检查也生成报告、旧 raw EEG
  E2E。按本轮核心语义更新为 feature/existing-result/readiness 三条路径；质量/就绪检查不再
  生成未请求报告。
- 重复运行报告仅因随机 Observation ID 不同而失败；保留真实 Trace ID，稳定性测试改为
  比较同一 ReportSpec 的 Renderer 输出和忽略 ID 后的跨运行语义内容。
- 初次静态检查发现旧 Manifest 构造仍使用兼容别名、3 处 import 顺序问题；已改为规范字段
  并格式化，最终 mypy/ruff 均通过。
- Phase 2 的知识检索仍是师兄 CSV 的离线词法读取，不称为真实 RAG；真实入口探测留到
  Phase 3A。

## 2026-07-29 — Core truth audit

- [x] 逐段追踪 CLI → Planner → Executor → Tool → Observation → Report
- [x] 确认当前系统属于 plan-then-execute，不是 incremental state-driven agent
- [x] 完成工具真实性矩阵和 Offline Benchmark 能力边界
- [x] 将核心实现缺口收敛到五项
- [x] Phase 1 硬门槛通过

执行命令：

```bash
rg -n "planner.plan|WorkflowExecutor|report_context|StructuredReportGenerator" eeg_asd_agent
rg -n "OfflineProvider|expected_|_simulate_report" eeg_asd_agent benchmark_cases
```

失败/偏差记录：

- README 描述的“根据新状态继续决策”与源码不一致。
- `restricted_real` 被用于本地 CSV 检索，真实性标签过强。
- Offline `llm_*` Benchmark 不是真实 LLM 实验。

详细证据：`docs/research/12_core_truth_audit.md`。

## 2026-07-29

- [x] 新建独立 Git 仓库与 Python 3.12 环境
- [x] AssessmentCaseState、Manifest、Plan、ToolResult、Observation、ReportSpec
- [x] 规则式 IntentParser、DeterministicPlanner、LLMPlanner 与 Offline fallback
- [x] Closed Tool Registry、DAG 校验、Executor、重试、跳过、max_steps、停滞检测
- [x] 17 个工具与 real/restricted_real/fixture 标记
- [x] 合成 EEG Welch/频带功率、师兄 OOF replay、师兄离线 KG
- [x] Normalizer、五类 Allowlist、三类 Context
- [x] 严格报告、结构修复、Markdown/HTML Renderer、Trace 与 Artifact
- [x] 8 个 Demo、30 个规划案例、25 个边界案例、20 个报告案例、8 个消融
- [x] 三种输入状态 E2E
- [x] 47 个自动测试
- [x] 完整 Benchmark 与完整性审计命令
- [x] 最终验收：pytest/ruff/mypy/compileall、三条 E2E、Benchmark、隐私扫描全部通过
- [x] OpenAI-compatible Responses Provider 接口、严格 Schema 与 Token 统计
- [x] 本地文献导出索引、Plan Mermaid、Trace HTML 与静态 Dashboard
- [ ] 真实 Provider 实验（无凭据时不阻塞，当前未运行）
- [ ] 大型交互 Web Dashboard（当前已提供静态 Artifact Dashboard）
- [ ] Multi-Agent 对照实验（P2，不影响主张）

## 实际成功命令

```bash
.venv/bin/pytest -q
.venv/bin/python -m eeg_asd_agent.cli tools list
.venv/bin/python -m eeg_asd_agent.cli run --case demo_cases/01_raw_eeg_full_assessment.json --request "分析这个 EEG，完成 ASD 辅助评估，并检索相关知识生成报告" --offline --output-dir artifacts/demo
.venv/bin/python -m eeg_asd_agent.cli benchmark all --offline --output-dir artifacts/benchmarks
```

## 已遇到并修复

- Codex Runtime 基础 Python 文件不可直接执行：改用旧 venv 的 Python 3.12 入口创建新 venv。
- pytest 8.4 保留 `request` fixture 名：参数化测试变量改名为 `query`。
- MyPy 3.10 配置无法解析 NumPy 3.12 stub：项目类型目标锁定为 Python 3.12。
