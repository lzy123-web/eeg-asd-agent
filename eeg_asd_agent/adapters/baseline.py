from __future__ import annotations

import csv
import hashlib
import math
import os
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from eeg_asd_agent.adapters.senior import DEFAULT_SENIOR_ROOT


class SeniorMLPBaselineAdapter:
    """NumPy inference adapter for the senior project's already-trained MLP.

    The original training script saved only a PyTorch state dict. The six tensor
    buffers are read directly from the documented zip checkpoint layout, so V1
    does not require PyTorch. The missing training scaler is approximated from
    the non-augmented source table and is surfaced as a hard limitation.
    """

    feature_names = [
        "higuchi_fd",
        "arousal_mean",
        "faa_when_avoid",
        "arousal_max",
        "parietal_alpha",
        "beta_gamma_mean",
        "arousal_std",
    ]
    tensor_shapes = [(32, 7), (32,), (16, 32), (16,), (3, 16), (3,)]
    output_labels = ["ability_level_1", "ability_level_2", "ability_level_3"]
    version = "senior-mlp-ability-v1"

    def __init__(self, root: Path | None = None) -> None:
        configured = os.environ.get("EEG_ASD_SENIOR_ROOT")
        self.root = root or (Path(configured) if configured else DEFAULT_SENIOR_ROOT)
        self.checkpoint_path = self.root / "best_mlp.pt"
        self.reference_table_path = self.root / "dynamic_features.csv"
        self._weights: list[np.ndarray] | None = None
        self._scaler: tuple[np.ndarray, np.ndarray] | None = None

    def readiness(
        self, features: Mapping[str, float | int | str | None]
    ) -> tuple[bool, str, list[str]]:
        missing_assets = [
            name
            for name, path in (
                ("best_mlp.pt", self.checkpoint_path),
                ("dynamic_features.csv", self.reference_table_path),
            )
            if not path.is_file()
        ]
        if missing_assets:
            return False, f"baseline assets unavailable: {', '.join(missing_assets)}", self.feature_names
        observed = []
        for name in self.feature_names:
            value = features.get(name)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                observed.append(name)
        missing = [name for name in self.feature_names if name not in observed]
        if len(observed) < 2:
            return False, "fewer than two compatible baseline features are present", missing
        contract = (
            "raw EEG features are name-mapped proxies with unverified training semantics"
            if "delta_relative_power" in features
            else "legacy feature names match but schema provenance is unverified"
        )
        return True, f"restricted baseline can run; {contract}", missing

    def predict(
        self, features: Mapping[str, float | int | str | None]
    ) -> dict[str, Any]:
        ready, reason, missing = self.readiness(features)
        if not ready:
            raise ValueError(reason)
        means, scales = self._load_scaler()
        values = []
        for index, name in enumerate(self.feature_names):
            value = features.get(name)
            values.append(
                float(value)
                if isinstance(value, (int, float)) and math.isfinite(float(value))
                else float(means[index])
            )
        vector = np.asarray(values, dtype=np.float32)
        standardized = (vector - means) / scales
        w1, b1, w2, b2, w3, b3 = self._load_weights()
        hidden1 = np.maximum(w1 @ standardized + b1, 0)
        hidden2 = np.maximum(w2 @ hidden1 + b2, 0)
        logits = w3 @ hidden2 + b3
        shifted = logits - np.max(logits)
        scores = np.exp(shifted) / np.exp(shifted).sum()
        selected = int(np.argmax(scores))
        checkpoint_hash = self._sha256(self.checkpoint_path)
        semantic_contract_status = (
            "proxy_raw_eeg_unverified"
            if "delta_relative_power" in features
            else "legacy_name_match_unverified"
        )
        return {
            "model_name": "senior_pretrained_mlp_ability_baseline",
            "model_version": f"{self.version}+sha256:{checkpoint_hash[:12]}",
            "readiness_status": "restricted",
            "predicted_pattern": self.output_labels[selected],
            "score": round(float(scores[selected]), 6),
            "class_scores": {
                label: round(float(score), 6)
                for label, score in zip(self.output_labels, scores, strict=True)
            },
            "score_meaning": "unvalidated softmax confidence, not an ASD probability",
            "output_meaning": (
                "Research-only three-level ability-pattern output from the senior project's "
                "already-trained checkpoint; it is not an ASD diagnosis."
            ),
            "observed_feature_count": len(self.feature_names) - len(missing),
            "imputed_features": missing,
            "feature_contract_version": "senior-top7-v1",
            "semantic_contract_status": semantic_contract_status,
            "scaler_mode": "proxy_from_non_augmented_source_table_with_mean_imputation",
            "scaler_version": f"proxy-sha256:{self._sha256(self.reference_table_path)[:12]}",
            "limitations": self.limitations(missing, semantic_contract_status),
        }

    @staticmethod
    def limitations(
        missing: list[str], semantic_contract_status: str = "unverified"
    ) -> list[str]:
        limitations = [
            "The checkpoint predicts a three-level ability_score research label, not ASD diagnosis.",
            "The original script used a row-level random split rather than a verified subject-level split.",
            "The training scaler was not saved; V1 reconstructs a proxy scaler from dynamic_features.csv.",
            "No prospective or clinical validation is claimed.",
            "Feature semantics and scales have not been proven identical to the model training pipeline "
            f"(contract={semantic_contract_status}).",
        ]
        if missing:
            limitations.append(
                "Missing model inputs were reference-mean-imputed: "
                + ", ".join(sorted(missing))
                + "."
            )
        return limitations

    def _load_weights(self) -> list[np.ndarray]:
        if self._weights is not None:
            return self._weights
        with zipfile.ZipFile(self.checkpoint_path) as archive:
            candidates = sorted(
                (
                    name
                    for name in archive.namelist()
                    if "/data/" in name and name.rsplit("/", 1)[-1].isdigit()
                ),
                key=lambda name: int(name.rsplit("/", 1)[-1]),
            )
            if len(candidates) != len(self.tensor_shapes):
                raise ValueError("unexpected MLP checkpoint tensor layout")
            weights = []
            for name, shape in zip(candidates, self.tensor_shapes, strict=True):
                values = np.frombuffer(archive.read(name), dtype="<f4").copy()
                if values.size != int(np.prod(shape)):
                    raise ValueError(f"unexpected tensor size for {name}")
                weights.append(values.reshape(shape))
        self._weights = weights
        return weights

    def _load_scaler(self) -> tuple[np.ndarray, np.ndarray]:
        if self._scaler is not None:
            return self._scaler
        with self.reference_table_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        matrix = np.asarray(
            [[float(row[name]) for name in self.feature_names] for row in rows],
            dtype=np.float32,
        )
        means = matrix.mean(axis=0)
        scales = matrix.std(axis=0)
        scales[scales == 0] = 1.0
        self._scaler = means, scales
        return self._scaler

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
