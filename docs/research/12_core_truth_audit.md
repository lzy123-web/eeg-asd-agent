# Core Truth Audit

**Date:** 2026-07-29  
**Audited commit:** `5325f01`  
**Question:** 当前代码是否真的实现了所描述的 EEG-ASD Workflow Agent？

## 结论

当前实现真实属于：

> **B. plan-then-execute（一次规划后固定执行）**

它不是固定到完全忽略输入的 Pipeline：`DeterministicPlanner` 会读取目标、输入类型和
`completed_workflows`，因此优于 A。但它也不是 C：Planner 只在执行前调用一次；
每个 ToolResult 虽然会更新 State，新的 Observation 不会触发下一次 Planner 决策。

当前模块名称中没有把 Executor 直接命名为 ReAct，但 README/架构图中描述的
“根据新状态继续决策”尚未由实际控制流实现。

## 1. 当前完整调用链

| Stage | File | Symbol | Input | Output |
|---|---|---|---|---|
| CLI | `eeg_asd_agent/cli.py` | `run_command` | case JSON、request、profile | Artifact 路径 |
| Workflow | `eeg_asd_agent/workflow.py` | `run_workflow` | `AssessmentCaseState`、request | 多个本地 Artifact |
| Intent | `intent/parser.py` | `RuleBasedIntentParser.parse` | request string | `RequestedGoal` |
| Planner | `planning/planners.py` | `DeterministicPlanner.plan` | State、Goal、Registry | 完整 `PlanSpec` |
| Executor | `execution/core.py` | `WorkflowExecutor.execute` | State、完整 Plan | `ExecutionOutcome` |
| Registry | `tools/registry.py` | `ToolRegistry.execute` | tool name、State、arguments | `ToolResult` |
| Normalize | `execution/core.py` | `ObservationNormalizer.normalize` | `ToolResult` | `Observation` |
| State update | `execution/core.py` | `_apply_state_transition` | State、ToolResult | 原地修改 State |
| Projection | `execution/core.py` | `AllowlistProjector.project` | Observation、Goal | `LLMVisibleToolResult` |
| Context | `execution/core.py` | `ContextBuilder.report_context` | 最终 State、Goal | report payload dict |
| Generation | `reporting/core.py` | `StructuredReportGenerator.generate` | report payload | `FinalAssessmentReportSpec` |
| Validation | `tools/builtin.py` | `validate_report_spec` | State metadata 中的 spec | `ToolResult` |
| Rendering | `tools/builtin.py` | `render_report` | 已解析 ReportSpec | Markdown ToolResult |

实际控制流是：

```text
request
→ parse goal
→ build complete PlanSpec once
→ for step in plan.steps
    execute
    normalize
    mutate selected State fields
→ build report context once
→ generate ReportSpec outside Executor
→ validate and render
```

缺失的目标控制流是：

```text
Observe State
→ Decide one next Action
→ Execute
→ Normalize
→ apply explicit StateUpdate
→ Decide again
```

## 2. Planner 真实性

### 确实读取

- 用户请求：先由 `RuleBasedIntentParser` 转为 Goal；Planner 只在多目标干预关键词处直接读取
  `state.user_request`。
- 当前 State：读取 `input_kind`、`completed_workflows`，部分分支读取 Goal。
- 已完成步骤：从初始完整 Plan 中删除已完成工具。
- Registry：过滤未注册名称；读取 Manifest 的 `retry_limit`。
- LLMPlanner：额外把 `completed_workflows`、`model_readiness` 和全部 Manifest 放入一次性 Prompt。

### 没有真实读取或执行

- 没有逐轮读取新 Observation。
- DeterministicPlanner 没有通过 Manifest precondition 推导下一步。
- 没有读取 failed workflows、pending requirements、current tool availability。
- Registry 不可用工具会在执行期 blocked，而不是在下一轮决策时改道。
- `PlanStep.condition` 存在 Schema，但 Executor 不解释该字段。

## 3. Executor 真实性

| Requirement | Current evidence | Result |
|---|---|---|
| 每步更新状态 | `_apply_state_transition` 更新质量、特征、readiness、模型结果 | PARTIAL |
| 根据新 Observation 决定下一步 | 固定 `for step in plan.steps`，无 Planner 回调 | FAIL |
| 跳过已完成 | 执行前及执行中检查 `completed_workflows` | PASS |
| 输入不足停止 | 仅依赖初始 Plan 和关键工具 `on_failure=stop` | PARTIAL |
| 失败重试/降级 | 同一工具重试；之后按预写 `on_failure` 继续或停止 | PARTIAL |
| blocked 不调用正式模型 | classifier 前硬编码 readiness blocked 检查 | PASS，但不是 Planner 决策 |
| 质量检查提前结束 | 初始 Goal 生成短 Plan | PASS，但不是增量停止 |

State 由 Executor 私有方法直接修改，部分工具还会直接修改
`state.metadata_summary`。当前不存在独立、可验证的 `StateUpdate`。

## 4. 核心模块存在性

| Module | Exists | Actually called | Truth |
|---|---:|---:|---|
| AssessmentCaseState | yes | yes | 字段不完整 |
| Capability Manifest | yes | yes | 主要用于展示、重试和顶层白名单 |
| Closed Tool Registry | yes | yes | unknown/duplicate 会拒绝 |
| Planner | yes | once | 一次性 Planner |
| Executor | yes | yes | 固定 Plan Executor |
| ObservationNormalizer | yes | yes | 会递归移除敏感 key |
| AllowlistProjector | yes | yes | 顶层 deny-by-default；嵌套不严格 |
| Execution-grounded Context | yes | yes | 来自已执行 Observation，但包含不符合新契约的 state 摘要 |
| ReportSpec Validator | yes | yes | 在生成之后执行 |
| Deterministic Renderer | yes | yes | Markdown/HTML 不调用 LLM |

## 5. 关键工具真实性矩阵

| Capability | Implementation | Honest type | Reason |
|---|---|---|---|
| `inspect_case` | State 字段汇总 | REAL | 确定性读取当前 State |
| `inspect_eeg` synthetic | 从 metadata 返回固定信息 | FIXTURE | 不读取实际信号 |
| `inspect_eeg` external | 扩展名＋State 中预填 metadata | STUB | 没有实际读取 EDF/MAT |
| `check_data_quality` | 路径/特征有限性规则 | REAL | 规则真实，但质量范围有限 |
| `preprocess_eeg` | 只返回步骤名称 | STUB | 没有对信号进行变换 |
| `extract_eeg_features` | 固定随机种子合成信号＋SciPy Welch | REAL algorithm / SYNTHETIC input | 算法执行真实，输入不是真实被试 |
| `check_model_readiness` | Profile/row/result 规则 | REAL | 真实规则状态工具 |
| `run_asd_classifier` fixture | 根据 synthetic feature 返回演示 score | FIXTURE | 不是训练模型 |
| `run_asd_classifier` senior | 读取 5 个 OOF CSV 指定行 | REPLAY | 历史输出回放，不是 prospective inference |
| `generate_medical_analysis` | 固定模板文字 | FIXTURE | 不运行师兄 LoRA 或真实 LLM |
| `predict_severity` | 固定 insufficient label | FIXTURE | 不运行真实 Stage2 |
| `retrieve_asd_knowledge` | ability/paper CSV 词面排序 | OFFLINE | 不是 Neo4j、embedding 或师兄在线 RAG |
| `generate_intervention_candidates` | KG task 节点模板 | OFFLINE/ADAPTER | 候选汇总，不是治疗生成 |
| `validate_report_spec` | Pydantic validate | REAL | 确定性 |
| `render_report` | Markdown template | REAL | 确定性 |
| Offline report provider | 根据 visible facts 写固定 JSON | OFFLINE/FIXTURE | 只验证 Schema 和控制流 |

当前 `retrieve_asd_knowledge` Manifest 标记为 `restricted_real`，这比真实能力更强，
需要改成 `offline`。Classifier Manifest 同时覆盖 fixture 和 replay，也需要拆分。

## 6. Allowlist 与 Context 真实性

- 新顶层字段默认不可见，`new_unreviewed_field` 测试可以证明这一点。
- Normalizer 会递归删除 key 中含 raw/signal/path/file/secret/token 的内容。
- 但是 `items` 一旦作为顶层允许字段进入，内部 dict 没有逐字段投影。
- 模型输出没有目标契约要求的 model name/version/output meaning。
- 知识结果没有统一 `evidence_id/title/normalized_summary/source/relevance/limitations`。
- Context 来自已执行 Observation，而非未执行工具，这一点成立。
- Context 仍缺用户原文、pending requirements、ReportSpec Schema 和统一非诊断规则。
- 当前 Context 不含完整 Trace 或原始 EEG。

## 7. Offline Benchmark 真实性

| Question | Answer |
|---|---|
| 是否调用真实 LLM | 否 |
| 是否处理真实 EEG | 否；规划数据是 JSON，E2E 原始链为 synthetic URI |
| 是否调用师兄真实 RAG | 否；调用导出 CSV 的本地词面检索 |
| 是否使用 expected answer | Evaluator 使用 expected；Planner 没收到 expected 字段 |
| 是否针对具体句子写死 | IntentParser 使用关键词规则；不是逐 case ID，但与冻结请求措辞高度耦合 |
| fixture 是否预含正确 limitations | 是；OfflineProvider 和 ToolResult 已预置限制 |
| LLM variants 是否真实 LLM | 否；`llm_*` variants 实际使用 OfflineProvider |
| Report variants 是否真实生成实验 | 否；`_simulate_report` 主动注入预设错误用于 evaluator self-test |

当前 Benchmark 能证明：

- CLI、Evaluator、Artifact 和指标管道可以运行；
- Deterministic Planner 在当前冻结规则集上满足预设约束；
- Allowlist/Schema evaluator 能检出人工注入的已知错误。

当前 Benchmark 不能证明：

- 真实 LLM 使用 Manifest 后更准确；
- 真实 EEG/模型/RAG 的有效性；
- LLM 边界行为得到改善；
- 方法可泛化到未冻结语言表达；
- 临床或研究结论成立。

## 8. 五个最关键实现缺口

1. **State/StateUpdate 不完整**：缺少 failed workflows、pending requirements、current action、
   step count、execution status；工具没有显式 StateUpdate 边界。
2. **没有增量决策**：Planner 只运行一次，Executor 是完整 Plan for-loop。
3. **报告和停止不是 Agent Action**：report generation 位于 Executor 外；缺少 generate/stop 工具。
4. **Allowlist 与 Schema 粒度不足**：嵌套 items 整体放行；Context/ReportSpec 不符合新契约。
5. **真实性与评测标签不准确**：Offline RAG、Replay classifier、Fixture LLM 与 Benchmark
   variants 的命名可能造成超额解读。

## Phase 1 Gate

- [x] 找到完整调用链
- [x] 确认核心模块实际位置
- [x] 明确 Planner 当前真实行为
- [x] 明确关键工具真实类型
- [x] 缺口收敛到五项

**Gate result: PASS。可以进入 Phase 2，但只允许修正核心闭环。**

