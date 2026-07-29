from __future__ import annotations

import json
import re
from typing import Protocol

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
            repaired = self._safe_repair(raw)
            try:
                return FinalAssessmentReportSpec.model_validate_json(repaired), raw, True
            except Exception:
                try:
                    provider_repaired = self.provider.generate_json(
                        "repair",
                        {
                            "case_id": context["case_id"],
                            "raw_output": raw,
                            "visible_results": context.get("visible_results", []),
                            "capability_context": context.get("capability_context", []),
                        },
                    )
                    return (
                        FinalAssessmentReportSpec.model_validate_json(provider_repaired),
                        raw,
                        True,
                    )
                except Exception:
                    pass
                fallback_raw = self.provider.generate_json(
                    "report",
                    {
                        "case_id": context["case_id"],
                        "visible_results": [],
                    },
                )
                spec = FinalAssessmentReportSpec.model_validate_json(fallback_raw)
                return spec, raw, True

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


class MarkdownRenderer:
    def render(self, report: FinalAssessmentReportSpec) -> str:
        lines = [
            f"# {report.title}",
            "",
            f"**Case ID:** `{report.case_id}`",
            f"**Evaluation mode:** `{report.evaluation_mode}`",
            "",
            "## 摘要",
            "",
            report.summary,
            "",
        ]
        findings = {finding.finding_id: finding for finding in report.findings}
        for section in report.sections:
            lines.extend([f"## {section.heading}", ""])
            for paragraph in section.paragraphs:
                lines.extend([paragraph, ""])
            for finding_id in section.finding_ids:
                finding = findings.get(finding_id)
                if finding:
                    lines.extend(
                        [
                            f"### {finding.title}",
                            "",
                            finding.statement,
                            "",
                            f"- 支持状态：`{finding.support_status}`",
                            f"- 来源工具：{', '.join(finding.source_tools) or '无'}",
                            "",
                        ]
                    )
        lines.extend(["## 安全说明", "", report.safety_notice, "", "## Provenance", ""])
        lines.extend(f"- `{item}`" for item in report.provenance)
        return "\n".join(lines).rstrip() + "\n"


class HTMLRenderer:
    _template = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ report.title }}</title>
<style>
body{font-family:system-ui,sans-serif;max-width:920px;margin:40px auto;padding:0 24px;line-height:1.65;color:#172033}
.meta,.notice{background:#f3f6fa;padding:12px 16px;border-radius:10px}
.finding{border-left:4px solid #4169e1;padding-left:16px;margin:20px 0}
code{background:#eef2f7;padding:2px 5px;border-radius:4px}
</style></head><body>
<h1>{{ report.title }}</h1>
<div class="meta">Case <code>{{ report.case_id }}</code> · mode <code>{{ report.evaluation_mode }}</code></div>
<h2>摘要</h2><p>{{ report.summary }}</p>
{% for section in report.sections %}<section><h2>{{ section.heading }}</h2>
{% for paragraph in section.paragraphs %}<p>{{ paragraph }}</p>{% endfor %}
{% for finding_id in section.finding_ids %}{% set finding = findings[finding_id] %}
<div class="finding"><h3>{{ finding.title }}</h3><p>{{ finding.statement }}</p>
<small>{{ finding.support_status }} · {{ finding.source_tools|join(', ') }}</small></div>
{% endfor %}</section>{% endfor %}
<h2>安全说明</h2><p class="notice">{{ report.safety_notice }}</p>
</body></html>"""

    def __init__(self) -> None:
        self.environment = Environment(
            loader=BaseLoader(),
            autoescape=select_autoescape(default=True),
        )

    def render(self, report: FinalAssessmentReportSpec) -> str:
        template = self.environment.from_string(self._template)
        findings = {finding.finding_id: finding for finding in report.findings}
        return template.render(report=report, findings=findings)
