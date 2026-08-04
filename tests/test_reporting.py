from eeg_asd_agent.providers import OfflineProvider
from eeg_asd_agent.reporting import HTMLRenderer, MarkdownRenderer, StructuredReportGenerator


def _context() -> dict[str, object]:
    return {
        "case_id": "report-case",
        "user_request": "生成受限报告",
        "executed_workflows": ["check_data_quality"],
        "visible_results": [
            {
                "observation_id": "obs-quality",
                "tool_name": "check_data_quality",
                "status": "success",
                "visible_facts": {"quality_status": "pass"},
                "limitations": [],
                "provenance_labels": ["tool:check_data_quality@1.0"],
                "implementation_kind": "real",
            }
        ],
    }


def test_report_generator_returns_strict_spec() -> None:
    report, raw, repaired = StructuredReportGenerator(OfflineProvider()).generate(_context())
    assert report.case_id == "report-case"
    assert report.report_type == "non_diagnostic_research_assessment"
    assert raw.startswith("{")
    assert repaired is False


def test_report_safe_repair_strips_code_fence() -> None:
    raw = "```json\n" + OfflineProvider().generate_json("report", _context()) + "\n```"
    repaired = StructuredReportGenerator._safe_repair(raw)
    assert repaired.startswith("{")
    assert repaired.endswith("}")


def test_renderers_are_stable() -> None:
    report, _, _ = StructuredReportGenerator(OfflineProvider()).generate(_context())
    first = MarkdownRenderer().render(report)
    second = MarkdownRenderer().render(report)
    assert first == second
    html = HTMLRenderer().render(report)
    assert "<!doctype html>" in html
    assert "不支持临床确诊" in html


def test_report_contains_sources_and_limitations() -> None:
    report, _, _ = StructuredReportGenerator(OfflineProvider()).generate(_context())
    assert [item.tool_name for item in report.provenance] == ["check_data_quality"]
    assert report.limitations
    assert report.data_summary.source_tools == ["check_data_quality"]


def test_invalid_provider_output_attempts_one_repair_then_uses_fallback() -> None:
    class RepairProvider:
        calls: list[str] = []

        def generate_json(self, purpose: str, payload: dict[str, object]) -> str:
            self.calls.append(purpose)
            if purpose == "report":
                return "invalid output"
            raise AssertionError(purpose)

    provider = RepairProvider()
    report, _, repaired = StructuredReportGenerator(provider).generate(_context())
    assert report.case_id == "report-case"
    assert report.report_type == "deterministic_limited_report"
    assert repaired is True
    assert provider.calls == ["report", "repair"]
