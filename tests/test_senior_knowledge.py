from __future__ import annotations

from pathlib import Path

from eeg_asd_agent.adapters import SeniorKnowledgeAdapter
from eeg_asd_agent.schemas import StructuredKnowledgeResult


def _senior_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "senior"
    entity = root / "KG" / "knowledgegraph_rag" / "entity"
    (entity / "evidence").mkdir(parents=True)
    (entity / "paper").mkdir(parents=True)
    (entity / "assessment").mkdir(parents=True)
    (entity / "evidence" / "ability_nodes.csv").write_text(
        "id,name,description\n"
        "ability_1,social_interaction,autism social interaction ability\n",
        encoding="utf-8",
    )
    (entity / "evidence" / "ability_dimension_relationships.csv").write_text(
        "START_ID,END_ID,TYPE\n"
        "ability_1,dim_1,REFLECTED_BY\n",
        encoding="utf-8",
    )
    (entity / "assessment" / "dimension_nodes.csv").write_text(
        "id,name,description,scaleId,category\n"
        "dim_1,Social dimension,observed social communication,scale_1,social\n",
        encoding="utf-8",
    )
    (entity / "assessment" / "dimension_task_relationships.csv").write_text(
        "START_ID,END_ID,TYPE\n"
        "dim_1,task_1,INCLUDE\n",
        encoding="utf-8",
    )
    (entity / "assessment" / "task_nodes.csv").write_text(
        "id,name,description,scaleId,dimensionId,assessmentTarget,scoringRule,ageRange\n"
        "task_1,Joint attention task,,scale_1,dim_1,social interaction,1,child\n",
        encoding="utf-8",
    )
    (entity / "paper" / "paper.csv").write_text(
        "id,title,abstract,authors,journal,year,doi,url,query_tag,source\n"
        "p1,EEG markers in autism,EEG research background,,Journal,2025,10.test/p1,,EEG_ASD,PubMed\n",
        encoding="utf-8",
    )
    return root


def test_senior_knowledge_adapter_traverses_senior_graph_and_searches_papers(
    tmp_path: Path,
) -> None:
    result = SeniorKnowledgeAdapter(_senior_fixture(tmp_path)).retrieve(
        "EEG autism social communication",
        limit=2,
    )
    assert isinstance(result, StructuredKnowledgeResult)
    assert result.mode == "senior_graph_export_traversal+paper_bm25"
    assert result.query == "EEG autism social communication"
    assert result.items
    assert all(item.title and item.source and item.chunk for item in result.items)
    assert all(item.score > 0 for item in result.items)
    assert all(
        item.metadata["mode"] == "senior_graph_export_traversal+paper_bm25"
        for item in result.items
    )
    assert all(item.evidence_id and item.summary and item.limitations for item in result.items)
    graph = next(item for item in result.items if item.evidence_id.startswith("SENIOR-KG-"))
    paper = next(item for item in result.items if item.evidence_id.startswith("SENIOR-PAPER-"))
    assert graph.metadata["graph_path"] == "Ability->Dimension->Task"
    assert graph.metadata["dimension_count"] == 1
    assert graph.metadata["task_count"] == 1
    assert "Joint attention task" in graph.retrieved_chunk
    assert paper.metadata["doi"] == "10.test/p1"


def test_senior_knowledge_adapter_empty_result_degrades_cleanly(
    tmp_path: Path,
) -> None:
    result = SeniorKnowledgeAdapter(_senior_fixture(tmp_path)).retrieve(
        "term_that_does_not_exist",
        limit=3,
    )
    assert result.mode == "senior_graph_export_traversal+paper_bm25"
    assert result.items == []
