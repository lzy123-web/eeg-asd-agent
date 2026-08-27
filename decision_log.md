# Decision Log

## D001 — 新建独立仓库

旧 `bishe` 保持只读。原因是旧硬编码 Workflow 与新 Case-State/Registry 架构的核心抽象冲突。

## D002 — paicli 只借鉴设计

移植 Closed Registry、ExecutionPlan DAG、预算与稳定签名；不引入 Java、Shell、MCP 或编码工具。

## D003 — 两条论文主张

只研究能力感知规划和执行态安全上下文。Claim–Evidence 不再作为系统主线。

## D004 — 师兄模型为 restricted_real

OOF 输出只允许历史回放。新受试者推理默认阻断，不报告泛化性能或临床诊断。

## D005 — Offline 不是效果证据

OfflineProvider 用于可复现的系统/Schema/Benchmark 基础设施验证。所有输出设置
`evaluation_mode=fixture`、`scientific_claim_allowed=false`。

## D006 — 原始 EEG Demo 使用 synthetic URI

本机未发现真实受试者 EEG。`synthetic://` 输入通过固定随机种子生成，仅验证预处理和特征链路。

## D007 — Responses Provider 默认关闭

真实 Provider 必须同时显式使用 `--no-offline`、固定 `OPENAI_MODEL` 并提供环境密钥。
默认路径不访问网络，Provider 结果与 fixture Benchmark 分开保存。

## D008 — 不强接不兼容开源 EEG foundation model

EEGPT、EEG-DINO 等公开 checkpoint 是表征模型，ASD head 仍需下游训练；公开 ASD screening
checkpoint 使用临床结构字段而不是 EEG。V1 使用师兄现成 `best_mlp.pt` 做受限当前个案前向，
不重新训练，也不把其 `ability_score` 输出改写成 ASD 概率。

## D009 — 师兄 MLP 使用 NumPy checkpoint reader

checkpoint 只有六个小型 float tensor，NumPy 可在本机直接加载并前向。没有安装 PyTorch或使用远程
GPU。由于 scaler 未保存，`dynamic_features.csv` 只能提供代理 scaler，此差异作为硬限制输出。

## D010 — live RAG 不可用时使用 BM25 Evidence Adapter

查询由实际执行结果动态构造；师兄导出 CSV 通过只读 BM25 检索，模式固定为
`offline_bm25_local_export`。每条结果必须携带 Evidence ID、来源、摘要、相关度、metadata 和限制。

## D011 — LLM 拥有流程控制权

默认 Runtime 改为 `AgentTurnDecision` 驱动的 Plan–Act–Observe–Replan。Registry 只返回满足硬前置条件
的候选集合，不根据输入类型生成唯一动作，不替 LLM 排序，也不因已有资产自动跳过。

## D012 — 确定性 Planner 是测试基线，不是隐藏回退

非法 LLM Plan/Action 在新 Agent Runtime 中会被拒绝并写入 Trace；不会切换到固定 Pipeline。
OfflineProvider 可以模拟同一协议，但必须标记为 fixture，不能用于支持 LLM 规划效果主张。

## D013 — 完整领域能力先用禁用 Stub 冻结边界

EEG 模式分析、Profile、能力维度、结果比较等尚未实现能力以 `enabled=false`、`real_or_fake=stub`
注册。LLM 能理解未来能力面，但 Registry 永不执行 Stub；后续直接替换 Tool 实现，不新增架构层。

## D014 — Memory 本地持久化、对 LLM 最小投影

完整 Case State 与 Turn 保存到 SQLite；LLM 只看到安全摘要。恢复运行使用单调递增 Turn index，旧记录
不覆盖；累计 step_count 用于审计，每个新用户请求获得独立预算。
