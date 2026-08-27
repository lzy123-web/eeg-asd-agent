# EEG-ASD Full-domain Agent 架构交接

日期：2026-08-06  
状态：Agent 流程骨架已实现并通过离线结构验收；领域能力继续按 Tool 逐项填充。

## 1. 本轮完成的目标

本轮没有把“原始 EEG → 模型 → RAG → 报告”固化成新 Pipeline，而是把现有组件接入统一的
Agent Runtime：

```text
plan → choose one action → validate → act → observe → update state → remember → replan
```

LLM 是流程控制器，Harness 是能力与安全边界。用户只提供 EEG 时，Agent 可以利用内部工具逐步产生
后续所需资产；用户额外提供特征、模型结果、量表或报告时，LLM 根据具体目标决定复用、检查、比较、
重算或忽略。输入资产本身不再触发确定性“自动跳过”。

系统定位为 EEG-ASD 全领域专用 Agent，而不是只接受一种固定完整评估请求。数据检查、信号处理、
特征分析、模型运行、既有结果解释、知识检索、报告、比较和后续评估都属于能力面；当前尚未实现的
部分作为禁用 Capability Stub 明确存在。

架构图：

- [PNG](../../figures/agentic-runtime-architecture.png)
- [Mermaid source](../../figures/agentic-runtime-architecture.mmd)
- [Markdown preview](../../figures/agentic-runtime-architecture.md)

## 2. 一次 Agent Turn 的真实执行顺序

1. `AgentContextBuilder` 从当前 `AssessmentCaseState` 读取状态。
2. `AllowlistProjector` 将最近 Observation 投影为 LLM 可见字段。
3. SQLite Memory 只返回最近 Turn 的安全摘要，不返回完整本地 State。
4. Registry 枚举全部 Manifest，并只根据硬前置条件计算 `allowed_next_actions`。
5. Provider 读取请求、状态摘要、Plan、Observation、Memory、完整能力目录和合法动作。
6. Provider 输出一个严格 `AgentTurnDecision`：新 Plan 修订、最多一个 Tool Action、停止判断。
7. Controller 校验 Plan 中的 Tool 引用、所选 Tool、参数、前置条件和 Observation ID。
8. Runtime 通过 Closed Registry 执行工具；工具只返回 `ToolResult + StateUpdate`。
9. `ObservationNormalizer` 产生标准 Observation；`StateReducer` 是唯一状态写入通道。
10. Runtime 将完整 State 与本轮决策写入本地 SQLite，再进入下一轮 LLM 决策。
11. LLM 明确返回无 Tool 的 terminal turn 后完成；非法决策、预算耗尽或停滞会进入受控阻断状态。

这意味着 Planner 不是一次性生成静态 DAG。`AgentPlan` 是每轮可修改的任务假设，`ActionDecision`
才是当前唯一可执行动作。

## 3. 控制权边界

### LLM 能决定

- 根据自然语言请求确定任务范围；
- 选择先处理 EEG、先检查已有资产、先检索知识或先做其他可用任务；
- 在 Observation 出现后改变原计划；
- 对已有特征/结果选择复用、验证、比较、重算或忽略；
- 选择是否生成报告、是否请求人工复核、何时完成。

### Harness 只强制

- Tool 必须在 Closed Registry 中且 `enabled=true`；
- 参数必须符合 Manifest；
- 物理或数据依赖等硬前置条件必须成立；
- Tool 不得直接修改 Case State；
- LLM Context 必须经过 Allowlist 和脱敏；
- 单轮一个动作、最大 24 个工具步骤、失败重试上限和停滞检测；
- 报告必须通过严格 JSON Schema，Renderer 不得添加事实。

Harness 不根据 `input_type` 生成唯一下一步，也不因为发现已有结果就自动决定跳过。Manifest 的
`use_when` 和 `avoid_when` 给 LLM语义提示；`preconditions` 只表达不可绕过的执行条件。

## 4. 关键实现文件

| 文件 | 作用 |
|---|---|
| `eeg_asd_agent/agent/runtime.py` | Agent 主循环、单步执行、重试、预算、停滞和结束状态 |
| `eeg_asd_agent/agent/controller.py` | 调用 Provider，解析并校验 `AgentTurnDecision`，拒绝非法动作 |
| `eeg_asd_agent/agent/context.py` | 构造唯一允许交给 Agent Controller 的脱敏 Context |
| `eeg_asd_agent/agent/memory.py` | SQLite Case/Turn Memory、恢复、LLM 安全摘要 |
| `eeg_asd_agent/schemas/models.py` | `AgentPlan`、`AgentTurnDecision`、State、Manifest 等严格 Schema |
| `eeg_asd_agent/tools/registry.py` | Closed Registry 与硬前置条件；只计算可执行集合，不排序 |
| `eeg_asd_agent/tools/domain_catalog.py` | 完整 EEG-ASD 能力目录及禁用 Stub |
| `eeg_asd_agent/tools/core_workflow.py` | 已实现 EEG、模型、RAG、报告 Tool 与 Manifest |
| `eeg_asd_agent/execution/core.py` | Observation Normalizer、StateReducer、旧确定性基线执行器 |
| `eeg_asd_agent/providers/openai_compatible.py` | 真实 Provider 的 Agent Turn / Report 严格 Schema 与 Prompt |
| `eeg_asd_agent/providers/offline.py` | 离线 Agent 协议模拟；用于结构测试，不是 LLM 效果证据 |
| `eeg_asd_agent/workflow.py` | 组装 Runtime、Registry、Memory、Trace 和运行产物 |
| `eeg_asd_agent/cli.py` | `run`、`resume`、工具和 Profile 命令 |

旧 `NextActionPlanner`、`IncrementalWorkflowExecutor` 和 `build_plan()` 保留为测试/消融基线。默认
`cli run` 已进入新 Agent Runtime；确定性 Planner 不作为非法 LLM 输出后的隐藏回退。

## 5. Context 与隐私模型

Agent Controller 的 Context 包含：

- 脱敏用户请求；
- Case ID、输入类型提示和资产清单；
- 特征是否存在、特征数量和字段名，不含特征值；
- 数据质量、模型就绪、已完成/失败动作；
- 最近 8 条 Allowlist Observation；
- 当前可修订 Plan 和当前运行的 Action history；
- SQLite Memory 最近 Turn 的安全摘要；
- 所有 Manifest，包括禁用 Stub；
- 当前满足硬条件的全部 `allowed_next_actions`；
- 剩余步骤预算和非临床规则。

Agent Controller 看不到：

- 原始 EEG 数组；
- 完整特征矩阵或特征值；
- 绝对文件路径；
- 内部模型对象；
- 完整 `ToolResult`；
- API Key 或本地服务凭据。

完整 State 可以包含上述本地资产，但只保存在 Python 运行时、Artifact 和 SQLite 中。Memory 的
`snapshot_for_llm()` 是独立的安全投影，不等于从数据库直接读取 JSON 进入 Prompt。

报告 Provider 使用另一套 Context。它只能看到按 Workflow Manifest 投影后的执行事实、Evidence、
限制和 ReportSpec Schema，不能复用 Agent Controller 的内部控制 Context。

## 6. Memory 语义

Memory 当前是单机 SQLite，包含：

- `cases`：每个 Case 最新完整状态；
- `turns`：每轮 `AgentTurnDecision` 和对应 Observation。

`resume` 加载完整 State，但会为新用户请求清空运行内 Action window；历史 Turn 仍通过 Memory 摘要
提供。Turn index 跨恢复运行单调递增，不覆盖旧记录。`step_count` 保留累计审计值，但每次新请求有
独立的 24-step budget。

当前 Memory 是 Case/episodic memory，不做向量化长期偏好学习，也不会自动跨 Case 迁移患者信息。

## 7. 能力目录策略

`build_agent_registry()` 由两部分组成：

1. `build_core_registry()`：现有可执行能力；
2. `register_domain_capability_stubs()`：完整 Agent 目标能力的禁用 Stub。

禁用 Stub 仍向 LLM 说明能力名称、用途、限制和未来输入输出，但不会进入 `allowed_next_actions`，直接
执行也会得到受控 `BLOCKED/UNAVAILABLE`。这样可以先冻结 Agent 的领域边界，再逐个替换实现，而不需要
继续增加 Planner 或 Executor 抽象。

目前声明的扩展点包括 EEG 模式分析、EEG-ASD Profile、严重程度研究估计、能力维度估计、结果比较、
报告解析、结构化评估、可视化和研究模型训练。

## 8. RAG 在循环中的位置

`build_asd_rag_query` 的硬前置条件只要求 Case 已检查，不要求模型阶段已经完成。因此 RAG 可以在
EEG 分析前、中、后被选择。Query 会随着 Observation 改变；检索结果以结构化 Evidence 返回，再由
Allowlist 投影给报告 Provider。

当前已接入师兄：

- Ability、Dimension、Task 节点及关系导出；
- `Ability-[:REFLECTED_BY]->Dimension-[:INCLUDE]->Task` 路径；
- 论文/PubMed 导出语料；
- 可选 live Neo4j；不可用时使用同图谱 CSV 遍历 + paper BM25。

RAG 不是“模型做完后的解释模板”，而是 Agent 可按目标随时调用的外部知识能力。

## 9. Provider 与失败语义

### 真实 Provider

`OpenAIResponsesProvider` 使用环境变量配置模型、Base URL、温度和密钥。`agent_turn` 使用严格
`AgentTurnDecision` JSON Schema；`report` 使用严格 `FinalAssessmentReportSpec` Schema。

### Offline Provider

`OfflineProvider` 模拟相同的 Plan–Act–Observe–Replan 协议，可用于：

- 无凭据时跑通系统；
- 测试 Manifest 和 State 更新；
- 生成稳定 Artifact；
- 验证不同请求会触发不同动作顺序。

它是规则化 Fixture，不代表真实 LLM 的规划准确性。

### 失败原则

- 非法 Agent JSON：本次 Agent 运行阻断并写 Trace；
- 未注册/禁用 Tool：Registry 拒绝；
- Tool failure：按 Manifest 最多重试；下一轮可由 LLM改计划；
- 重复相同 Tool + 参数 + 状态签名：触发停滞保护；
- 超过预算：受控阻断；
- 报告 JSON 非法：报告 Provider 允许一次安全修复，仍失败则生成确定性有限报告。

“报告降级”不等于“规划回退”：前者只保证输出安全；后者如果偷偷启用固定 Pipeline，会破坏 Agent
控制权，因此新 Runtime 不这样做。

## 10. 已验证的架构行为

`tests/test_agent_architecture.py` 目前验证：

1. 完整领域能力目录存在，未实现能力均禁用；
2. Controller Context 不包含路径和特征值，并可同时提供多个合法动作；
3. 同一资产在“解释已有结果”和“重新运行验证”请求下可选择不同动作；
4. LLM 在真实质量 Observation 后改变下一动作，并引用 Observation ID；
5. 非法 `shell` 动作被拒绝，且没有固定 Pipeline 回退；
6. 只提供 raw EEG 也可在 Offline Agent 协议下完成闭环，无需用户先给量表或模型结果；
7. 真实 Provider 使用严格 Agent Turn Schema；
8. SQLite Memory 恢复不会覆盖旧 Turn，新请求可重复调用可重复能力；
9. `request_human_review` 通过显式 StateUpdate 将 Case 置为 `waiting`；
10. 调用 Tool 的 Turn 不能同时终止，必须先观察工具结果再决定结束。

当前全量验收命令：

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/mypy eeg_asd_agent
.venv/bin/python -m compileall -q eeg_asd_agent
```

## 11. 下一阶段按什么顺序填能力

架构已足够支撑继续开发，不需要新增 Planner 层或 Multi-Agent。建议顺序：

1. 优先把 `build_eeg_asd_profile`、`estimate_ability_dimensions`、`compare_assessment_results` 从 Stub
   替换为可执行 Tool，让“完整评估”不再主要依赖单个占位模型；
2. 继续恢复师兄 RAG 的真实查询与生成细节，用动态 Agent Query 验证图谱/论文 Evidence 闭环；
3. 配置一个固定真实 LLM，跑同资产多任务测试，记录每轮 Plan 修订、工具选择与停止行为。

在这三项完成前，不需要做 Web 前端、Multi-Agent、大规模 Benchmark、模型微调或临床性能主张。
