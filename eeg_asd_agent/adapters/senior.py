from __future__ import annotations

import csv
import hashlib
import math
import os
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any

from eeg_asd_agent.schemas import StructuredKnowledgeItem, StructuredKnowledgeResult

DEFAULT_SENIOR_ROOT = Path("/Users/nick/eeg/迁移/daima/autodl_latest_data")
DEFAULT_EEGAGENT_ROOT = Path(
    r"D:\Profession\Programs\CANKAO\EEGAgent-main\EEGAgent-main"
)


class _DataOnlyUnpickler(pickle.Unpickler):
    """Load EEGAgent's list-of-string chunk cache without allowing globals."""

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(
            f"global objects are forbidden in the EEGAgent chunk cache: {module}.{name}"
        )


class EEGAgentKnowledgeAdapter:
    """Reuse EEGAgent's prepared knowledge chunks with deterministic BM25 ranking.

    The reference repository's FAISS index requires an omitted BGE-M3 base-model
    checkpoint plus optional torch/faiss dependencies. Its already prepared chunk
    corpus is reusable, so this adapter preserves that knowledge asset and adds a
    dependency-light lexical fallback with stable evidence identifiers.
    """

    mode = "eegagent_prepared_chunks_bm25"

    def __init__(self, root: Path | None = None) -> None:
        configured = os.environ.get("EEG_AGENT_REFERENCE_ROOT")
        self.root = root or (Path(configured) if configured else DEFAULT_EEGAGENT_ROOT)
        self.chunk_path = self.root / "RAG" / "chunks.pkl"
        self._chunks: list[str] | None = None

    def retrieve(self, query: str, limit: int = 5) -> StructuredKnowledgeResult:
        chunks = self._load_chunks()
        tokens = SeniorKnowledgeAdapter._tokens(query)
        if not chunks or not tokens or limit <= 0:
            return StructuredKnowledgeResult(query=query, items=[], mode=self.mode)
        documents = [self._tokenize(chunk) for chunk in chunks]
        average_length = sum(map(len, documents)) / max(1, len(documents))
        document_frequency = {
            token: sum(token in set(document) for document in documents)
            for token in tokens
        }
        ranked: list[tuple[float, int, str]] = []
        for index, (chunk, document) in enumerate(zip(chunks, documents, strict=True)):
            counts = Counter(document)
            score = 0.0
            for token in tokens:
                frequency = counts[token]
                if not frequency:
                    continue
                inverse_document_frequency = math.log(
                    1
                    + (len(documents) - document_frequency[token] + 0.5)
                    / (document_frequency[token] + 0.5)
                )
                denominator = frequency + 1.5 * (
                    0.25 + 0.75 * len(document) / max(1.0, average_length)
                )
                score += inverse_document_frequency * frequency * 2.5 / denominator
            if score > 0:
                ranked.append((score, index, chunk))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        items = []
        for score, index, chunk in ranked[:limit]:
            digest = hashlib.sha256(chunk.encode("utf-8", errors="replace")).hexdigest()[:12]
            title = " ".join(chunk.replace("\n", " ").split())[:120]
            items.append(
                StructuredKnowledgeItem(
                    evidence_id=f"EEGAGENT-KB-{digest}",
                    title=title or f"EEGAgent knowledge chunk {index}",
                    source="EEGAgent:RAG/chunks.pkl",
                    retrieved_chunk=chunk[:1200],
                    summary=" ".join(chunk.replace("\n", " ").split())[:360],
                    relevance=round(float(score), 6),
                    metadata={
                        "mode": self.mode,
                        "chunk_index": index,
                        "source_file": "RAG/chunks.pkl",
                    },
                    limitations=[
                        "该条目复用 EEGAgent 预处理知识块，并以 BM25 词法相关性检索。",
                        "背景知识不能替代当前个案的临床判读。",
                    ],
                )
            )
        return StructuredKnowledgeResult(query=query, items=items, mode=self.mode)

    def _load_chunks(self) -> list[str]:
        if self._chunks is not None:
            return self._chunks
        if not self.chunk_path.is_file():
            self._chunks = []
            return self._chunks
        with self.chunk_path.open("rb") as handle:
            raw = _DataOnlyUnpickler(handle).load()
        if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
            raise ValueError("EEGAgent RAG/chunks.pkl must contain only a list of strings")
        self._chunks = [item.strip() for item in raw if item.strip()]
        return self._chunks

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]+|[\u4e00-\u9fff]{2,}", text)
            if len(token) > 1
        ]


class CompositeKnowledgeAdapter:
    """Combine optional ASD-specific exports with the reusable EEGAgent corpus."""

    def __init__(
        self,
        asd_adapter: SeniorKnowledgeAdapter | None = None,
        eeg_adapter: EEGAgentKnowledgeAdapter | None = None,
    ) -> None:
        self.asd_adapter = asd_adapter or SeniorKnowledgeAdapter()
        self.eeg_adapter = eeg_adapter or EEGAgentKnowledgeAdapter()

    def retrieve(self, query: str, limit: int = 5) -> StructuredKnowledgeResult:
        asd_result = self.asd_adapter.retrieve(query, limit=limit)
        remaining = max(0, limit - len(asd_result.items))
        eeg_result = self.eeg_adapter.retrieve(query, limit=remaining)
        modes = [
            mode
            for mode, items in (
                (asd_result.mode, asd_result.items),
                (eeg_result.mode, eeg_result.items),
            )
            if items
        ]
        return StructuredKnowledgeResult(
            query=query,
            items=[*asd_result.items, *eeg_result.items][:limit],
            mode="+".join(modes) or "knowledge_unavailable",
        )


class SeniorOOFAdapter:
    """Read-only replay of stored out-of-fold outputs.

    This is not a deployable classifier and must never be presented as prospective
    inference on a new participant.
    """

    files = {
        "catboost": ("CatBoostPred_with_oof.csv", "ability_score_pred", "ability_score_pred_proba"),
        "random_forest": ("RandomForest_with_oof.csv", "rf_pred", "rf_pred_proba"),
        "xgboost": ("XGBoostPred_with_oof.csv", "xgb_pred", "xgb_pred_proba"),
        "lightgbm": ("LightGBMPred_with_oof.csv", "lgb_pred", "lgb_pred_proba"),
        "svm": ("SVMPred_with_oof.csv", "svm_pred", "svm_pred_proba"),
    }

    def __init__(self, root: Path | None = None) -> None:
        configured = os.environ.get("EEG_ASD_SENIOR_ROOT")
        self.root = root or (Path(configured) if configured else DEFAULT_SENIOR_ROOT)

    def replay(self, row_index: int) -> dict[str, Any]:
        outputs = []
        for model_name, (filename, prediction_key, probability_key) in self.files.items():
            path = self.root / filename
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if row_index < 0 or row_index >= len(rows):
                raise IndexError(f"row_index {row_index} outside {filename}")
            row = rows[row_index]
            outputs.append(
                {
                    "model": model_name,
                    "prediction": row[prediction_key],
                    "probability": round(float(row[probability_key]), 6),
                }
            )
        labels = [str(item["prediction"]) for item in outputs]
        consensus = max(set(labels), key=labels.count)
        return {
            "predicted_label": consensus,
            "model_outputs": outputs,
            "model_count": len(outputs),
            "source_type": "legacy_oof_replay",
        }


class SeniorKGAdapter:
    def __init__(self, root: Path | None = None) -> None:
        configured = os.environ.get("EEG_ASD_SENIOR_ROOT")
        base = root or (Path(configured) if configured else DEFAULT_SENIOR_ROOT)
        self.root = base / "KG" / "knowledgegraph_rag" / "entity"

    def retrieve_asd(self, query: str, limit: int = 5) -> list[dict[str, str]]:
        path = self.root / "evidence" / "ability_nodes.csv"
        rows = self._read(path)
        tokens = {token.lower() for token in query.replace("，", " ").split() if token}
        ranked = sorted(
            rows,
            key=lambda row: sum(
                token in f"{row.get('name', '')} {row.get('description', '')}".lower()
                for token in tokens
            ),
            reverse=True,
        )
        return [
            {
                "id": row.get("id", ""),
                "name": row.get("name", ""),
                "description": row.get("description", ""),
                "source": "senior_kg:ability_nodes",
            }
            for row in ranked[:limit]
        ]

    def retrieve_intervention(self, query: str, limit: int = 5) -> list[dict[str, str]]:
        dimensions = self._read(self.root / "assessment" / "dimension_nodes.csv")
        tasks = self._read(self.root / "assessment" / "task_nodes.csv")
        selected: list[dict[str, str]] = []
        for row in tasks:
            if query and query.lower() not in str(row).lower() and len(selected) >= limit:
                continue
            selected.append(
                {
                    "id": row.get("id", ""),
                    "name": row.get("name", ""),
                    "assessment_target": row.get("assessmentTarget", ""),
                    "source": "senior_kg:task_nodes",
                }
            )
            if len(selected) >= limit:
                break
        if not selected and dimensions:
            row = dimensions[0]
            selected.append(
                {
                    "id": row.get("id", ""),
                    "name": row.get("name", ""),
                    "assessment_target": row.get("description", ""),
                    "source": "senior_kg:dimension_nodes",
                }
            )
        return selected

    @staticmethod
    def _read(path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))


class SeniorKnowledgeAdapter:
    """Query the senior project's AbilityKG RAG and paper corpus.

    This keeps the senior retrieval semantics instead of treating every CSV as
    an unstructured document: query terms select Ability nodes, then the adapter
    traverses ``Ability -> Dimension -> Task`` using the same relationships as
    the senior Neo4j code. A configured Neo4j service is preferred; otherwise
    the reviewed CSV graph export is traversed locally. PubMed exports form a
    second lexical evidence channel.
    """

    export_mode = "senior_graph_export_traversal+paper_bm25"
    live_mode = "senior_neo4j_graph_traversal+paper_bm25"
    ability_aliases = {
        "social_interaction": {
            "social",
            "interaction",
            "social interaction",
            "social communication",
            "社交",
            "交往",
            "社会互动",
            "社交沟通",
        },
        "communication_expression": {
            "communication",
            "language",
            "expression",
            "social communication",
            "沟通",
            "语言",
            "表达",
            "社交沟通",
        },
        "emotion_recognition": {
            "emotion",
            "recognition",
            "情绪",
            "识别",
            "情绪识别",
        },
        "emotion_regulation": {
            "emotion",
            "regulation",
            "情绪",
            "调节",
            "情绪调节",
        },
        "cognitive_flexibility": {
            "cognitive",
            "flexibility",
            "认知",
            "灵活",
            "认知灵活性",
        },
        "behavior_control": {
            "behavior",
            "control",
            "行为",
            "控制",
            "行为控制",
        },
    }

    def __init__(self, root: Path | None = None) -> None:
        configured = os.environ.get("EEG_ASD_SENIOR_ROOT")
        base = root or (Path(configured) if configured else DEFAULT_SENIOR_ROOT)
        self.entity_root = base / "KG" / "knowledgegraph_rag" / "entity"
        self._papers: list[dict[str, str]] | None = None
        self.neo4j_uri = os.environ.get("EEG_ASD_NEO4J_URI")
        self.neo4j_user = os.environ.get("EEG_ASD_NEO4J_USER")
        self.neo4j_password = os.environ.get("EEG_ASD_NEO4J_PASSWORD")

    def retrieve(self, query: str, limit: int = 5) -> StructuredKnowledgeResult:
        tokens = self._tokens(query)
        if not tokens or limit <= 0:
            return StructuredKnowledgeResult(query=query, items=[], mode=self.export_mode)
        ranked_abilities = self._rank_abilities(query)
        graph_limit = min(2, limit)
        graph_items, graph_mode, graph_warning = self._retrieve_graph(
            ranked_abilities[:graph_limit]
        )
        remaining = max(0, limit - len(graph_items))
        paper_items = self._retrieve_papers(query, remaining, graph_mode)
        items = [*graph_items, *paper_items]
        if graph_warning:
            items = [
                item.model_copy(
                    update={
                        "limitations": list(
                            dict.fromkeys([*item.limitations, graph_warning])
                        )
                    }
                )
                for item in items
            ]
        return StructuredKnowledgeResult(query=query, items=items, mode=graph_mode)

    def _retrieve_graph(
        self,
        ranked_abilities: list[tuple[float, dict[str, str]]],
    ) -> tuple[list[StructuredKnowledgeItem], str, str | None]:
        if not ranked_abilities:
            return [], self.export_mode, None
        ability_names = [row["name"] for _score, row in ranked_abilities]
        if self.neo4j_uri and self.neo4j_user and self.neo4j_password:
            try:
                return self._retrieve_live_graph(ranked_abilities, ability_names), self.live_mode, None
            except Exception as exc:
                warning = (
                    "Configured senior Neo4j retrieval was unavailable "
                    f"({type(exc).__name__}); the reviewed graph export was used."
                )
                return self._retrieve_export_graph(ranked_abilities), self.export_mode, warning
        return (
            self._retrieve_export_graph(ranked_abilities),
            self.export_mode,
            "Senior Neo4j was not configured; the same Ability-Dimension-Task "
            "graph was traversed from its read-only export.",
        )

    def _retrieve_live_graph(
        self,
        ranked_abilities: list[tuple[float, dict[str, str]]],
        ability_names: list[str],
    ) -> list[StructuredKnowledgeItem]:
        from neo4j import GraphDatabase  # type: ignore[import-not-found]

        assert self.neo4j_uri is not None
        assert self.neo4j_user is not None
        assert self.neo4j_password is not None
        driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password),
            connection_timeout=3,
        )
        query = """
        MATCH (a:Ability)
        WHERE a.name IN $abilities
        OPTIONAL MATCH (a)-[:REFLECTED_BY]->(d:Dimension)
        OPTIONAL MATCH (d)-[:INCLUDE]->(t:Task)
        RETURN a.id AS ability_id,
               a.name AS ability,
               a.description AS ability_description,
               collect(DISTINCT {id:d.id, name:d.name, description:d.description}) AS dimensions,
               collect(DISTINCT {id:t.id, name:t.name, target:t.assessmentTarget})[0..5] AS tasks
        """
        try:
            driver.verify_connectivity()
            with driver.session() as session:
                records = [
                    record.data()
                    for record in session.run(query, abilities=ability_names)
                ]
        finally:
            driver.close()
        score_by_name = {row["name"]: score for score, row in ranked_abilities}
        return [
            self._graph_record_to_item(
                record,
                float(score_by_name.get(str(record.get("ability", "")), 0.0)),
                self.live_mode,
            )
            for record in records
        ]

    def _retrieve_export_graph(
        self,
        ranked_abilities: list[tuple[float, dict[str, str]]],
    ) -> list[StructuredKnowledgeItem]:
        dimensions = {
            row.get("id", ""): row
            for row in self._read_optional(
                self.entity_root / "assessment" / "dimension_nodes.csv"
            )
        }
        tasks = {
            row.get("id", ""): row
            for row in self._read_optional(
                self.entity_root / "assessment" / "task_nodes.csv"
            )
        }
        ability_dimensions: dict[str, list[str]] = {}
        for row in self._read_optional(
            self.entity_root / "evidence" / "ability_dimension_relationships.csv"
        ):
            if row.get("TYPE") == "REFLECTED_BY":
                ability_dimensions.setdefault(row.get("START_ID", ""), []).append(
                    row.get("END_ID", "")
                )
        dimension_tasks: dict[str, list[str]] = {}
        for row in self._read_optional(
            self.entity_root / "assessment" / "dimension_task_relationships.csv"
        ):
            if row.get("TYPE") == "INCLUDE":
                dimension_tasks.setdefault(row.get("START_ID", ""), []).append(
                    row.get("END_ID", "")
                )

        items: list[StructuredKnowledgeItem] = []
        for score, ability in ranked_abilities:
            dimension_rows = [
                dimensions[dimension_id]
                for dimension_id in ability_dimensions.get(ability.get("id", ""), [])
                if dimension_id in dimensions
            ]
            task_rows: list[dict[str, str]] = []
            for dimension in dimension_rows:
                for task_id in dimension_tasks.get(dimension.get("id", ""), []):
                    if task_id in tasks and tasks[task_id] not in task_rows:
                        task_rows.append(tasks[task_id])
            record: dict[str, Any] = {
                "ability_id": ability.get("id", ""),
                "ability": ability.get("name", ""),
                "ability_description": ability.get("description", ""),
                "dimensions": [
                    {
                        "id": row.get("id", ""),
                        "name": row.get("name", ""),
                        "description": row.get("description", ""),
                    }
                    for row in dimension_rows
                ],
                "tasks": [
                    {
                        "id": row.get("id", ""),
                        "name": row.get("name", ""),
                        "target": row.get("assessmentTarget", ""),
                    }
                    for row in task_rows[:5]
                ],
            }
            items.append(self._graph_record_to_item(record, score, self.export_mode))
        return items

    def _graph_record_to_item(
        self,
        record: dict[str, Any],
        score: float,
        mode: str,
    ) -> StructuredKnowledgeItem:
        ability_id = str(record.get("ability_id") or record.get("ability") or "unknown")
        ability_name = str(record.get("ability", ""))
        ability_description = str(record.get("ability_description", ""))
        dimensions = [
            dict(value)
            for value in record.get("dimensions", [])
            if isinstance(value, dict) and value.get("name")
        ]
        tasks = [
            dict(value)
            for value in record.get("tasks", [])
            if isinstance(value, dict) and value.get("name")
        ]
        dimension_text = "; ".join(
            f"{value.get('name', '')}: {value.get('description', '')}".strip()
            for value in dimensions[:4]
        )
        task_text = "; ".join(
            f"{value.get('name', '')}（目标: {value.get('target', '')}）".strip()
            for value in tasks[:5]
        )
        chunk = (
            f"能力节点 {ability_name}: {ability_description}。"
            f"图谱路径 Ability-REFLECTED_BY-Dimension: {dimension_text or '无导出维度'}。"
            f"Dimension-INCLUDE-Task: {task_text or '无导出任务'}。"
        )
        return StructuredKnowledgeItem(
            evidence_id=f"SENIOR-KG-{ability_id}",
            title=f"师兄 AbilityKG：{ability_name}",
            source="senior_rag:Ability-REFLECTED_BY-Dimension-INCLUDE-Task",
            retrieved_chunk=chunk[:1200],
            summary=(
                f"{ability_name} 关联 {len(dimensions)} 个评估维度和 {len(tasks)} 个任务。"
            ),
            relevance=round(float(score), 6),
            metadata={
                "mode": mode,
                "ability_id": ability_id,
                "dimension_ids": ",".join(
                    str(value.get("id", "")) for value in dimensions
                ),
                "task_ids": ",".join(str(value.get("id", "")) for value in tasks),
                "dimension_count": len(dimensions),
                "task_count": len(tasks),
                "graph_path": "Ability->Dimension->Task",
            },
            limitations=[
                "师兄 AbilityKG 提供背景知识与任务关联，不是当前个案的临床证据。",
                "能力节点由动态 Query 词法匹配选择，尚未进行临床相关性验证。",
            ],
        )

    def _rank_abilities(self, query: str) -> list[tuple[float, dict[str, str]]]:
        lowered = query.lower()
        query_tokens = self._tokens(query)
        ranked: list[tuple[float, dict[str, str]]] = []
        for row in self._read_optional(
            self.entity_root / "evidence" / "ability_nodes.csv"
        ):
            name = row.get("name", "")
            text_tokens = self._tokens(
                f"{name.replace('_', ' ')} {row.get('description', '')}"
            )
            score = float(len(query_tokens & text_tokens))
            score += 2.0 * sum(
                alias in lowered
                for alias in self.ability_aliases.get(name, set())
            )
            if score > 0:
                ranked.append((score, row))
        return sorted(ranked, key=lambda item: (-item[0], item[1].get("id", "")))

    def _retrieve_papers(
        self,
        query: str,
        limit: int,
        mode: str,
    ) -> list[StructuredKnowledgeItem]:
        if limit <= 0:
            return []
        documents = self._load_papers()
        tokens = self._tokens(query)
        tokenized = [self._document_tokens(document) for document in documents]
        average_length = sum(len(items) for items in tokenized) / max(1, len(tokenized))
        frequencies = {
            token: sum(token in set(items) for items in tokenized) for token in tokens
        }
        ranked: list[tuple[float, dict[str, str]]] = []
        for document, document_tokens in zip(documents, tokenized, strict=True):
            counts = Counter(document_tokens)
            length = len(document_tokens)
            score = 0.0
            for token in tokens:
                frequency = counts[token]
                if frequency == 0:
                    continue
                document_frequency = frequencies[token]
                inverse_document_frequency = math.log(
                    1 + (len(documents) - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * length / max(1, average_length))
                score += inverse_document_frequency * frequency * 2.5 / denominator
                if token in self._tokens(document["title"]):
                    score += inverse_document_frequency
            if score > 0:
                ranked.append((score, document))
        ranked.sort(key=lambda item: (-item[0], item[1]["id"]))
        return [
            StructuredKnowledgeItem(
                evidence_id=f"SENIOR-PAPER-{document['id'] or index}",
                title=document["title"],
                source=f"senior_rag:{document['source']}",
                retrieved_chunk=document["chunk"][:800],
                summary=document["chunk"][:360],
                relevance=round(score, 6),
                metadata={
                    "source_file": document["source_file"],
                    "year": document.get("year", ""),
                    "doi": document.get("doi", ""),
                    "mode": mode,
                },
                limitations=[
                    "师兄 PubMed 导出语料仅作为背景文献，不是当前个案证据。",
                    "论文通道使用 BM25 词法相关性，尚未进行临床相关性验证。",
                ],
            )
            for index, (score, document) in enumerate(ranked[:limit], start=1)
        ]

    def _load_papers(self) -> list[dict[str, str]]:
        if self._papers is not None:
            return self._papers
        documents: list[dict[str, str]] = []
        paper_path = self.entity_root / "paper" / "paper.csv"
        for row in self._read_optional(paper_path):
            documents.append(
                {
                    "id": row.get("id", ""),
                    "title": row.get("title", ""),
                    "chunk": row.get("abstract", ""),
                    "tags": row.get("query_tag", ""),
                    "source": f"senior_paper_export:{row.get('source', 'unknown')}",
                    "source_file": "paper/paper.csv",
                    "year": row.get("year", ""),
                    "doi": row.get("doi", ""),
                }
            )
        self._papers = documents
        return documents

    @staticmethod
    def _tokens(query: str) -> set[str]:
        tokens = {
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]+|[\u4e00-\u9fff]{2,}", query)
            if len(token) > 1
        }
        lowered = query.lower()
        if "asd" in lowered or "autism" in lowered or "孤独症" in query:
            tokens.update({"asd", "autism"})
        if "eeg" in lowered or "脑电" in query:
            tokens.update({"eeg", "electroencephal"})
        if "能力" in query or "ability" in lowered:
            tokens.update({"ability", "cognitive", "social"})
        if "communication" in lowered or "沟通" in query:
            tokens.update({"communication", "language", "social"})
        return tokens

    @classmethod
    def _document_tokens(cls, document: dict[str, str]) -> list[str]:
        text = f"{document['title']} {document['chunk']} {document['tags']}"
        tokens = [
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]+|[\u4e00-\u9fff]{2,}", text)
            if len(token) > 1
        ]
        tokens.extend(token for token in cls._tokens(text) if token not in tokens)
        return tokens

    @staticmethod
    def _read(path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    @classmethod
    def _read_optional(cls, path: Path) -> list[dict[str, str]]:
        return cls._read(path) if path.is_file() else []


class LocalLiteratureAdapter:
    """Small read-only lexical index over the senior project's exported paper table."""

    def __init__(self, root: Path | None = None) -> None:
        configured = os.environ.get("EEG_ASD_SENIOR_ROOT")
        base = root or (Path(configured) if configured else DEFAULT_SENIOR_ROOT)
        self.path = base / "KG" / "knowledgegraph_rag" / "entity" / "paper" / "paper.csv"

    def search(self, query: str, limit: int = 3) -> list[dict[str, str]]:
        tokens = {
            token.lower()
            for token in query.replace("，", " ").replace("-", " ").split()
            if len(token) > 2
        }
        with self.path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        ranked = sorted(
            rows,
            key=lambda row: sum(
                token
                in (
                    f"{row.get('title', '')} {row.get('abstract', '')} "
                    f"{row.get('query_tag', '')}"
                ).lower()
                for token in tokens
            ),
            reverse=True,
        )
        return [
            {
                "id": row.get("id", ""),
                "title": row.get("title", ""),
                "year": row.get("year", ""),
                "doi": row.get("doi", ""),
                "source": "senior_offline_literature_export",
            }
            for row in ranked[:limit]
        ]
