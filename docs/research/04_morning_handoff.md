# Morning Handoff

## 一分钟验收

```bash
cd /Users/nick/programs/eeg-system/eeg-asd-workflow-agent
.venv/bin/python -m eeg_asd_agent.cli tools list
.venv/bin/python -m eeg_asd_agent.cli run \
  --case demo_cases/01_raw_eeg_full_assessment.json \
  --request "分析这个 EEG，完成 ASD 辅助评估，并检索相关知识生成报告" \
  --offline --output-dir artifacts/demo
.venv/bin/python -m eeg_asd_agent.cli benchmark all \
  --offline --output-dir artifacts/benchmarks
.venv/bin/pytest -q
```

## 看哪里

- `artifacts/demo/report.html`：最终可见报告。
- `artifacts/demo/plan.json`：状态感知工具计划。
- `artifacts/demo/llm_payload.json`：确认没有原始 EEG、本地路径或内部模型输出。
- `artifacts/demo/trace.jsonl`：逐步执行与受限状态。
- `artifacts/benchmarks/benchmark_summary.md`：四类实验汇总。

## 不要误读

`fixture` 是流程验证，`legacy_oof_replay` 是历史输出重放；两者都不是临床模型结果。

