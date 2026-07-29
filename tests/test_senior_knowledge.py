from __future__ import annotations

from pathlib import Path

from eeg_asd_agent.adapters import SeniorKnowledgeAdapter
from eeg_asd_agent.schemas import StructuredKnowledgeResult


def _senior_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "senior"
    entity = root / "KG" / "knowledgegraph_rag" / "entity"
    (entity / "evidence").mkdir(parents=True)
    (entity / "paper").mkdir(parents=True)
    (entity / "evidence" / "ability_nodes.csv").write_text(
        "id,name,description\n"
        "ability_1,social_interaction,autism social interaction ability\n",
        encoding="utf-8",
    )
    (entity / "paper" / "paper.csv").write_text(
        "id,title,abstract,authors,journal,year,doi,url,query_tag,source\n"
        "p1,EEG markers in autism,EEG research background,,Journal,2025,10.test/p1,,EEG_ASD,PubMed\n",
        encoding="utf-8",
    )
    return root


def test_senior_knowledge_adapter_returns_structured_local_index(
    tmp_path: Path,
) -> None:
    result = SeniorKnowledgeAdapter(_senior_fixture(tmp_path)).retrieve(
        "EEG autism",
        limit=2,
    )
    assert isinstance(result, StructuredKnowledgeResult)
    assert result.mode == "offline_local_index"
    assert result.query == "EEG autism"
    assert result.items
    assert all(item.title and item.source and item.chunk for item in result.items)
    assert all(item.score > 0 for item in result.items)
    assert all(item.metadata["mode"] == "offline_local_index" for item in result.items)
    assert all(item.metadata["source_file"] for item in result.items)


def test_senior_knowledge_adapter_empty_result_degrades_cleanly(
    tmp_path: Path,
) -> None:
    result = SeniorKnowledgeAdapter(_senior_fixture(tmp_path)).retrieve(
        "term_that_does_not_exist",
        limit=3,
    )
    assert result.mode == "offline_local_index"
    assert result.items == []
