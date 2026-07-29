from eeg_asd_agent.providers import OfflineProvider
from eeg_asd_agent.reporting import HTMLRenderer, MarkdownRenderer, StructuredReportGenerator


def _context() -> dict[str, object]:
    return {
        "case_id": "report-case",
        "visible_results": [
            {
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
    assert report.evaluation_mode == "fixture"
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
    assert "不得将本输出用于临床确诊" in html


def test_report_contains_sources_and_limitations() -> None:
    report, _, _ = StructuredReportGenerator(OfflineProvider()).generate(_context())
    assert "check_data_quality" in report.provenance
    assert report.limitations
    assert report.findings[0].source_tools == ["check_data_quality"]


def test_provider_repair_is_attempted_once() -> None:
    valid = OfflineProvider().generate_json("report", {"case_id": "report-case", "visible_results": []})

    class RepairProvider:
        calls: list[str] = []

        def generate_json(self, purpose: str, payload: dict[str, object]) -> str:
            self.calls.append(purpose)
            if purpose == "report":
                return "invalid output"
            if purpose == "repair":
                return valid
            raise AssertionError(purpose)

    provider = RepairProvider()
    report, _, repaired = StructuredReportGenerator(provider).generate(_context())
    assert report.case_id == "report-case"
    assert repaired is True
    assert provider.calls == ["report", "repair"]
