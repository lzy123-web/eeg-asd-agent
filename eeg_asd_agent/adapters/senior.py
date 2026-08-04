from __future__ import annotations

import csv
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from eeg_asd_agent.schemas import StructuredKnowledgeItem, StructuredKnowledgeResult

DEFAULT_SENIOR_ROOT = Path("/Users/nick/eeg/迁移/daima/autodl_latest_data")


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
    """Read-only local index over the senior project's exported KG and paper data.

    The live Neo4j retrieval code was inspected and probed, but cannot run in the
    current environment. This adapter therefore preserves a stable structured
    interface while reporting ``mode=offline_bm25_local_export``.
    """

    mode = "offline_bm25_local_export"

    def __init__(self, root: Path | None = None) -> None:
        configured = os.environ.get("EEG_ASD_SENIOR_ROOT")
        base = root or (Path(configured) if configured else DEFAULT_SENIOR_ROOT)
        self.entity_root = base / "KG" / "knowledgegraph_rag" / "entity"
        self._documents: list[dict[str, str]] | None = None

    def retrieve(self, query: str, limit: int = 5) -> StructuredKnowledgeResult:
        tokens = self._tokens(query)
        if not tokens or limit <= 0:
            return StructuredKnowledgeResult(query=query, items=[], mode=self.mode)
        documents = self._load_documents()
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
        items = [
            StructuredKnowledgeItem(
                evidence_id=document["id"] or f"local-{index}",
                title=document["title"],
                source=document["source"],
                retrieved_chunk=document["chunk"][:800],
                summary=document["chunk"][:360],
                relevance=round(score, 6),
                metadata={
                    "source_file": document["source_file"],
                    "year": document.get("year", ""),
                    "doi": document.get("doi", ""),
                    "mode": self.mode,
                },
                limitations=[
                    "Locally exported background knowledge; not participant-level evidence.",
                    "Retrieval relevance is lexical BM25 and has not been clinically validated.",
                ],
            )
            for index, (score, document) in enumerate(ranked[:limit], start=1)
        ]
        return StructuredKnowledgeResult(query=query, items=items, mode=self.mode)

    def _load_documents(self) -> list[dict[str, str]]:
        if self._documents is not None:
            return self._documents
        documents: list[dict[str, str]] = []
        ability_path = self.entity_root / "evidence" / "ability_nodes.csv"
        for row in self._read(ability_path):
            documents.append(
                {
                    "id": row.get("id", ""),
                    "title": row.get("name", ""),
                    "chunk": row.get("description", ""),
                    "tags": "ASD autism ability 能力 孤独症",
                    "source": "senior_kg_export:ability_nodes",
                    "source_file": "evidence/ability_nodes.csv",
                    "year": "",
                    "doi": "",
                }
            )
        paper_path = self.entity_root / "paper" / "paper.csv"
        for row in self._read(paper_path):
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
        self._documents = documents
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
