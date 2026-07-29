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
