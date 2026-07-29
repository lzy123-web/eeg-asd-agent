# Progress

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
- [x] 46 个自动测试
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
