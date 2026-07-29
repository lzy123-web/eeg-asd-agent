from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def audit_integrity(root: Path) -> dict[str, Any]:
    train_path = root / "stage1_train_data.jsonl"
    test_path = root / "stage1_test_data.jsonl"
    if not train_path.exists() or not test_path.exists():
        return {
            "status": "blocked",
            "reason": "stage1 train/test files unavailable",
            "root_label": root.name,
        }
    train = _normalized_hashes(train_path)
    test = _normalized_hashes(test_path)
    overlap = train & test
    return {
        "status": "risk_detected" if overlap else "no_exact_overlap_detected",
        "train_rows": len(train),
        "test_rows": len(test),
        "exact_overlap_count": len(overlap),
        "model_readiness": "blocked" if overlap else "restricted",
        "limitations": [
            "Exact normalized overlap is only one integrity check.",
            "This command does not establish clinical validity or generalization.",
        ],
    }


def _normalized_hashes(path: Path) -> set[str]:
    hashes = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            canonical = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            hashes.add(hashlib.sha256(canonical.encode()).hexdigest())
    return hashes

