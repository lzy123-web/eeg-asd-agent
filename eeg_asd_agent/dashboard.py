from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any


def build_dashboard(artifact_root: Path, output: Path) -> Path:
    benchmark_path = artifact_root / "benchmarks" / "benchmark_results.json"
    report_path = artifact_root / "demo" / "report_spec.json"
    benchmark = _load(benchmark_path)
    report = _load(report_path)
    cards = []
    for name, suite in benchmark.get("suites", {}).items():
        count = suite.get("case_count", suite.get("count", 0))
        cards.append(f"<article><h2>{escape(name)}</h2><strong>{count}</strong><p>frozen cases/configs</p></article>")
    findings = "".join(
        f"<li><strong>{escape(str(item.get('title', '')))}</strong> — "
        f"{escape(str(item.get('statement', '')))}</li>"
        for item in report.get("findings", [])
    )
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EEG-ASD Workflow Dashboard</title>
<style>
body{{font-family:system-ui;margin:0;background:#f5f7fb;color:#172033}}
header,main{{max-width:1100px;margin:auto;padding:28px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:16px}}
article,section{{background:white;border:1px solid #dfe5ee;border-radius:14px;padding:20px}}
strong{{font-size:2rem;color:#3156d3}}code{{background:#eef2f7;padding:2px 5px}}
.notice{{border-left:5px solid #e36a2e}}
</style></head><body>
<header><h1>EEG-ASD Workflow Agent</h1>
<p>Execution-grounded planning and report safety dashboard</p></header><main>
<section class="notice"><strong>fixture</strong>
<p>当前 Benchmark 验证基础设施，不支持真实 LLM 或临床效果主张。</p></section>
<div class="grid">{''.join(cards)}</div>
<section><h2>Demo Findings</h2><ul>{findings}</ul></section>
<section><h2>Artifacts</h2>
<p><code>artifacts/demo/trace.html</code> · <code>artifacts/demo/report.html</code> ·
<code>artifacts/benchmarks/benchmark_summary.md</code></p></section>
</main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        return {}
    return value
