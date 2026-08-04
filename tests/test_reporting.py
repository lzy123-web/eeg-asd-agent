import json

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


def _rag_context() -> dict[str, object]:
    context = _context()
    context["executed_workflows"] = [
        "check_data_quality",
        "retrieve_asd_knowledge",
    ]
    context["visible_results"] = [
        *context["visible_results"],  # type: ignore[misc]
        {
            "observation_id": "obs-rag",
            "tool_name": "retrieve_asd_knowledge",
            "status": "success",
            "visible_facts": {
                "mode": "senior_graph_export_traversal+paper_bm25",
                "items": [
                    {
                        "evidence_id": "SENIOR-KG-ability_1",
                        "title": "师兄 AbilityKG：social_interaction",
                        "source": "senior_rag:Ability-REFLECTED_BY-Dimension-INCLUDE-Task",
                        "summary": "social_interaction 关联图谱维度和任务。",
                    }
                ],
            },
            "limitations": [],
            "provenance_labels": ["tool:retrieve_asd_knowledge@2.0"],
            "implementation_kind": "adapter",
        },
    ]
    return context


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


def test_report_cites_only_ids_returned_by_senior_rag() -> None:
    report, _, repaired = StructuredReportGenerator(OfflineProvider()).generate(
        _rag_context()
    )
    assert repaired is False
    assert report.knowledge_interpretation.evidence_ids == [
        "SENIOR-KG-ability_1"
    ]
    assert "AbilityKG" in report.knowledge_interpretation.summary


def test_invented_evidence_id_is_rejected_after_one_repair() -> None:
    class InventedEvidenceProvider:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.offline = OfflineProvider()

        def generate_json(self, purpose: str, payload: dict[str, object]) -> str:
            self.calls.append(purpose)
            raw = self.offline.generate_json("report", payload)
            value = json.loads(raw)
            value["knowledge_interpretation"]["evidence_ids"] = ["MADE-UP-ID"]
            return json.dumps(value, ensure_ascii=False)

    provider = InventedEvidenceProvider()
    report, _, repaired = StructuredReportGenerator(provider).generate(_rag_context())
    assert repaired is True
    assert report.report_type == "deterministic_limited_report"
    assert report.knowledge_interpretation.evidence_ids == []
    assert provider.calls == ["report", "repair"]


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
