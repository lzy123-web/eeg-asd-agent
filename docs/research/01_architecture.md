# Architecture

系统只允许两类生成模型行为：

1. Planner 输出严格 `PlanSpec`；
2. Reporter 输出严格 `FinalAssessmentReportSpec`。

数值处理全部由确定性工具完成。Plan 中的工具名必须来自 Registry；每一步执行后生成
ToolResult，再经 Normalizer 与工作流白名单形成 LLMVisibleToolResult。

关键不变量：

- 未注册工具不能执行；
- Plan 是无环依赖图；
- 新 ToolResult 字段默认不可见；
- 原始 EEG 和本地路径不进入 Context；
- Readiness blocked 时分类器不执行；
- 非法报告只修复一次，仍失败则生成有限报告；
- Renderer 不调用 LLM。

八张经 Mermaid CLI 验证并人工复核的图表位于 `figures/`。
