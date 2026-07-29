# Benchmark Design

## Planning

30 个请求，覆盖 raw EEG、feature table、model result、medical report、knowledge-only
与 incomplete。比较 Fixed Pipeline、Deterministic、LLM 无 Manifest、LLM＋Manifest。

## Boundary

25 个问题，标签包括 supported、conditional、unsupported、not implemented、blocked。
比较 Generic、Manifest、Manifest＋Allowlist、Execution-grounded Context。

## Report

20 个固定场景，每个重复 3 次。比较 Raw ToolResult、Allowlist、
Allowlist＋Capability、Allowlist＋Capability＋Schema。

当前 `evaluation_mode=fixture`。这些数值只用于验证指标、产物和回归门禁，不能证明真实
LLM 的方法优越性。真实 Provider 实验需固定模型版本、温度 0，并至少重复三次。

Report Benchmark 的 Offline 数值由 20 个 fixture × 4 variants × 3 repeats 的实际输出
逐条经过 Schema、字段、数字、来源、措辞、限制和一致性 Evaluator 计算，不使用预填结果。
其中弱基线是故意注入不安全字段的 harness self-test，作用是证明评测器能够检出错误。
