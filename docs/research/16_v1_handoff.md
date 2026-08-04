# V1 implementation handoff — 2026-08-04

## 1. 本轮交付结论

V1 主链已实现并本地验证：

```text
raw EEG or feature table
→ deterministic EEG chain (when required)
→ current-case pretrained research baseline
→ execution-grounded RAG query
→ traceable local knowledge retrieval
→ incremental LLM/Deterministic Planner
→ strict structured report
→ deterministic Markdown/HTML renderer
```

本轮没有重写已有架构，没有增加 Multi-Agent、前端、训练任务或临床结论。旧工程全程只读。

## 2. 实际实现文件

- `eeg_asd_agent/adapters/eeg.py`：synthetic / NPZ / MAT / EDF 读取，质量、预处理、事件、epoch、特征。
- `eeg_asd_agent/adapters/baseline.py`：直接读取师兄 `best_mlp.pt` 的 NumPy 当前个案推理。
- `eeg_asd_agent/adapters/senior.py`：师兄 AbilityKG 图谱遍历、可选 Neo4j 与论文 BM25 Evidence。
- `eeg_asd_agent/planning/incremental.py`：Deterministic 与逐轮 Provider `ActionDecision` Planner。
- `eeg_asd_agent/tools/core_workflow.py`：EEG、baseline、query、RAG、report 闭环工具与 Manifest。
- `eeg_asd_agent/reporting/core.py`：一次 Provider repair、确定性有限回退、Evidence ID 渲染。
- `profiles/v1.json`：EEG、模型、RAG、Planner 与 Provider 的版本化配置快照。
- `tests/test_v1_workflow.py`：四条 E2E、真实质量分支、模型前向、LLM Context 和安全边界。

## 3. EEG 工具链

固定 profile：`eeg-v1.0`，128 Hz、1–40 Hz、可用时 50 Hz notch、2 秒 epoch。

调用顺序：

```text
inspect_eeg
→ check_eeg_quality
→ preprocess_eeg
→ extract_events
→ create_epochs
→ extract_eeg_features
```

特征包括五频带相对功率均值/标准差、parietal alpha、beta+gamma、Higuchi FD、可用通道下的
FAA 与 theta coherence。原始信号不进入 ToolResult、Observation、LLM Payload 或 Trace。

本机没有发现真实受试者 EEG，因此主 Demo 使用固定 seed synthetic；另有自动测试使用真实落盘
NPZ 读取与处理。MAT 与等采样率 EDF 已实现，混合采样率 EDF 和 BDF/SET/FIF 会明确拒绝。

## 4. 模型接入与开源搜索结论

优先审计了公开已训练模型：

- [EEGPT](https://github.com/BINE022/EEGPT) 是通用 EEG 表征检查点，官方流程仍要求 downstream linear probe / fine-tune。
- [EEG-DINO pretrained](https://huggingface.co/braindecode/eegdino-small-pretrained) 的分类 head 加载时重新初始化，模型卡明确要求下游 fine-tune / linear probe。
- [Archicava autism-detector](https://huggingface.co/Archicava/autism-detector) 虽是已训练 ASD screening 模型，但输入是 8 个临床结构字段，不是 EEG。

因此没有发现可以在当前 4–8 通道 synthetic/本地 EEG 契约下直接执行、且输出 ASD 个体结果的可信
开源检查点。强行接入 foundation encoder 会违反“不重新训练”以及输入契约。

V1 最终直接接入师兄现成 `best_mlp.pt`：7→32→16→3，目标 `ability_score`。使用 checkpoint
SHA-256 版本标识，NumPy 直接加载六个 tensor buffer，实际执行当前 case 前向传播；不需要 PyTorch，
也没有使用远程服务器。

限制：训练脚本为行级随机切分；训练 scaler 丢失；当前用非增强源表代理 scaler；缺失模态按参考
均值插补；raw EEG 同名特征的语义和量纲未证明与训练管线一致；输出不是 ASD 概率或临床结论。
历史 `research_replay_classifier` 只作 baseline 失败时的
历史 OOF fallback。

## 5. RAG 闭环

先执行 `build_asd_rag_query`，查询会根据模型 pattern、可用 EEG 特征名和已有报告状态变化；不包含
完整数值。`retrieve_asd_knowledge` 返回严格 Evidence：`evidence_id`、title、source、
retrieved_chunk、summary、relevance、metadata、limitations。

师兄 live Neo4j runtime 当前不可用，所以没有伪装成在线服务。当前 Adapter 先用 Query 匹配师兄
Ability 节点，再按师兄原实现遍历 `Ability→Dimension→Task` 的导出关系，随后联合师兄 PubMed
语料的 BM25 结果，运行模式为 `senior_graph_export_traversal+paper_bm25`。配置 Neo4j 环境变量后会
优先运行同一条 Cypher。ReportSpec 只能引用检索实际返回的 `SENIOR-KG-*` / `SENIOR-PAPER-*` ID；
不存在的 ID 在报告校验阶段被拒绝。

## 6. LLM Planner 与报告 Provider

真实 Provider 接口使用 OpenAI-compatible Responses API，model/base URL/temperature/key 全由环境变量
配置。每轮只请求一个 `ActionDecision`；Provider 收到脱敏 Case 摘要、先前白名单 Observation、
Capability Manifests、完成状态，以及从 Manifest 前置条件独立计算的合法候选集合。真实 Provider
Payload 不含 DeterministicPlanner 的预选动作；确定性 Planner 仅在 Provider 失败时接管。

Schema 错误、未知工具、非法参数、提前停止、网络/Provider 异常会回退到 DeterministicPlanner。
报告 Provider 只看到执行态 Allowlist Context；ReportSpec 第一次失败后允许一次 Provider repair，仍失败
则生成 deterministic limited report。未配置凭据时只验证 OfflineProvider 接口，不声称 LLM 效果。

## 7. 四条 E2E 验证

1. Raw synthetic EEG：完整 EEG chain → baseline → dynamic query → RAG → ReportSpec → render。
2. Feature table：跳过 `inspect_eeg`/preprocess/events/epochs/features，直接 quality → baseline → RAG → report。
3. Existing model result：跳过 EEG 和模型推理，inspect existing result → RAG → report。
4. Quality only：`check_eeg_quality` 后停止，不运行模型、RAG 或报告。

额外观察分支：高幅值且平直的 NPZ 在质量 Observation 后选择
`stop_with_insufficient_input`；下游三类工具均未调用。

## 8. 复现命令

```bash
cd /Users/nick/programs/eeg-system/eeg-asd-workflow-agent
.venv/bin/python -m eeg_asd_agent.cli run \
  --case demo_cases/01_raw_eeg_full_assessment.json \
  --request "分析这个 EEG，完成 ASD 辅助评估，并检索相关知识生成报告" \
  --planner llm --offline --output-dir artifacts/v1/raw-llm-interface

.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/mypy eeg_asd_agent
.venv/bin/python -m compileall -q eeg_asd_agent
```

## 9. 失败、降级与未执行项

- 没有兼容的即用型开源 ASD-EEG checkpoint：保留搜索记录，使用师兄已训练受限模型。
- 师兄 checkpoint 缺 scaler：使用代理 scaler并作为硬限制，不宣称性能。
- live Neo4j/Spark RAG 不可用：遍历师兄同一图谱的只读 CSV 导出并联合论文 BM25，mode 明示。
- 本地没有真实 EEG：使用 synthetic + 测试 NPZ，只证明软件链路。
- 没有 Provider 凭据：真实 LLM 调用未运行；离线 NextAction 只证明接口和安全门。
- 未使用远程 GPU：NumPy 可直接完成 checkpoint 前向，避免无必要的凭据和环境风险。

## 10. 科学与临床边界

当前结果可以支持“工作流是否按状态选择、跳过和停止”以及“LLM Payload 是否受控”的软件研究，
不能支持模型准确率、ASD 个体识别有效性、LLM 方法优于基线或临床效用。任何论文主结果仍需真实受试者
数据、subject-level protocol、冻结模型契约和真实 Provider 重复实验。
