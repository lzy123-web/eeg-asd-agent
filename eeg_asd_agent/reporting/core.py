from __future__ import annotations

import json
import re
from typing import Protocol, cast

from jinja2 import BaseLoader, Environment, select_autoescape

from eeg_asd_agent.schemas import FinalAssessmentReportSpec


class ReportProvider(Protocol):
    def generate_json(self, purpose: str, payload: dict[str, object]) -> str: ...


class StructuredReportGenerator:
    def __init__(self, provider: ReportProvider) -> None:
        self.provider = provider

    def generate(self, context: dict[str, object]) -> tuple[FinalAssessmentReportSpec, str, bool]:
        raw = self.provider.generate_json("report", context)
        try:
            return FinalAssessmentReportSpec.model_validate_json(raw), raw, False
        except Exception:
            try:
                repair_payload = dict(context)
                repair_payload["raw_output"] = raw
                repaired_raw = self.provider.generate_json("repair", repair_payload)
                repaired = self._safe_repair(repaired_raw)
                return FinalAssessmentReportSpec.model_validate_json(repaired), raw, True
            except Exception:
                return self._deterministic_limited_report(context), raw, True

    @staticmethod
    def _safe_repair(raw: str) -> str:
        stripped = raw.strip()
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
        first = stripped.find("{")
        last = stripped.rfind("}")
        if first >= 0 and last > first:
            candidate = stripped[first : last + 1]
            json.loads(candidate)
            return candidate
        return stripped

    @staticmethod
    def _deterministic_limited_report(
        context: dict[str, object],
    ) -> FinalAssessmentReportSpec:
        visible = list(cast(list[object], context.get("visible_results", [])))
        provenance = []
        for item in visible:
            if not isinstance(item, dict):
                continue
            provenance.append(
                {
                    "tool_name": str(item.get("tool_name", "unknown")),
                    "observation_id": str(item.get("observation_id", "unknown")),
                    "implementation_kind": str(item.get("implementation_kind", "unavailable")),
                }
            )
        return FinalAssessmentReportSpec.model_validate(
            {
                "case_id": str(context["case_id"]),
                "report_type": "deterministic_limited_report",
                "request_summary": str(context.get("user_request", "")),
                "executed_workflows": [
                    str(item)
                    for item in cast(
                        list[object], context.get("executed_workflows", [])
                    )
                ],
                "data_summary": {
                    "status": "limited",
                    "summary": "结构化报告生成失败，未扩展任何工具事实。",
                    "source_tools": [],
                },
                "model_assessment": {
                    "status": "unavailable",
                    "summary": "报告生成失败后不推断模型结论。",
                    "source_tools": [],
                },
                "knowledge_interpretation": {
                    "status": "unavailable",
                    "summary": "报告生成失败后不推断知识内容。",
                    "evidence_ids": [],
                },
                "unavailable_sections": ["model_assessment", "knowledge_interpretation"],
                "limitations": [
                    "Provider 输出在一次安全结构修复后仍不符合 ReportSpec。",
                    "本报告为确定性有限降级输出，不代表真实 LLM 效果。",
                ],
                "conclusion": "信息不足；需要人工复核。",
                "human_review_required": True,
                "provenance": provenance,
            }
        )


class MarkdownRenderer:
    def render(self, report: FinalAssessmentReportSpec) -> str:
        lines = [
            "# EEG-ASD 辅助工作流报告",
            "",
            f"**Case ID:** `{report.case_id}`",
            f"**Report type:** `{report.report_type}`",
            "",
            "## 请求摘要",
            "",
            report.request_summary,
            "",
            "## 数据摘要",
            "",
            report.data_summary.summary,
            "",
            "## 模型评估",
            "",
            report.model_assessment.summary,
            "",
            "## 知识解释",
            "",
            report.knowledge_interpretation.summary,
            "",
            "### Evidence IDs",
            "",
            *(
                [f"- `{item}`" for item in report.knowledge_interpretation.evidence_ids]
                or ["- 无可用 Evidence ID"]
            ),
            "",
            "## 结论",
            "",
            report.conclusion,
            "",
            "## 限制",
            "",
        ]
        lines.extend(f"- {item}" for item in report.limitations)
        lines.extend(["", "## Provenance", ""])
        lines.extend(
            f"- `{item.tool_name}` / `{item.observation_id}` / `{item.implementation_kind.value}`"
            for item in report.provenance
        )
        return "\n".join(lines).rstrip() + "\n"


class HTMLRenderer:
    _template = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EEG-ASD 辅助工作流报告</title>
<style>
body{font-family:system-ui,sans-serif;max-width:920px;margin:40px auto;padding:0 24px;line-height:1.65;color:#172033}
.meta,.notice{background:#f3f6fa;padding:12px 16px;border-radius:10px}
.finding{border-left:4px solid #4169e1;padding-left:16px;margin:20px 0}
code{background:#eef2f7;padding:2px 5px;border-radius:4px}
</style></head><body>
<h1>EEG-ASD 辅助工作流报告</h1>
<div class="meta">Case <code>{{ report.case_id }}</code> · type <code>{{ report.report_type }}</code></div>
<h2>请求摘要</h2><p>{{ report.request_summary }}</p>
<h2>数据摘要</h2><p>{{ report.data_summary.summary }}</p>
<h2>模型评估</h2><p>{{ report.model_assessment.summary }}</p>
<h2>知识解释</h2><p>{{ report.knowledge_interpretation.summary }}</p>
<h3>Evidence IDs</h3><ul>
{% for item in report.knowledge_interpretation.evidence_ids %}<li><code>{{ item }}</code></li>{% endfor %}
</ul>
<h2>结论</h2><p class="notice">{{ report.conclusion }}</p>
<h2>限制</h2><ul>{% for item in report.limitations %}<li>{{ item }}</li>{% endfor %}</ul>
</body></html>"""

    def __init__(self) -> None:
        self.environment = Environment(
            loader=BaseLoader(),
            autoescape=select_autoescape(default=True),
        )

    def render(self, report: FinalAssessmentReportSpec) -> str:
        template = self.environment.from_string(self._template)
        return template.render(report=report)
