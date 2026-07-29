from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

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
