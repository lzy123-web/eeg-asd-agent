from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from pydantic import BaseModel

_SENSITIVE_KEYS = re.compile(r"(path|raw|signal|password|secret|token|api[_-]?key)", re.I)


def jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return value.name
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if _SENSITIVE_KEYS.search(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str) and (value.startswith("/") or re.match(r"^[A-Za-z]:\\", value)):
        return "[LOCAL_PATH_REDACTED]"
    return value


class TraceLogger:
    def __init__(self, path: Path, trace_id: str) -> None:
        self.path = path
        self.trace_id = trace_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_id": self.trace_id,
            "event_type": event_type,
            "payload": redact(jsonable(payload)),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(self, relative: str, value: Any) -> Path:
        target = self._target(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(jsonable(value), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target

    def write_text(self, relative: str, value: str) -> Path:
        target = self._target(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value, encoding="utf-8")
        return target

    def append_jsonl(self, relative: str, value: Any) -> Path:
        target = self._target(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(jsonable(value), ensure_ascii=False) + "\n")
        return target

    def _target(self, relative: str) -> Path:
        target = (self.root / relative).resolve()
        if self.root.resolve() not in target.parents and target != self.root.resolve():
            raise ValueError("artifact path escapes output directory")
        return target


def render_trace_html(path: Path) -> str:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        rows.append(
            "<tr>"
            f"<td>{escape(str(record.get('timestamp', '')))}</td>"
            f"<td>{escape(str(record.get('event_type', '')))}</td>"
            f"<td><pre>{escape(json.dumps(record.get('payload', {}), ensure_ascii=False, indent=2))}</pre></td>"
            "</tr>"
        )
    return (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<title>EEG-ASD Trace</title><style>"
        "body{font-family:system-ui;margin:24px;color:#172033}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #d8dee9;padding:8px;vertical-align:top}"
        "th{background:#eef2f7}pre{white-space:pre-wrap;margin:0}"
        "</style></head><body><h1>EEG-ASD Trace</h1><table>"
        "<thead><tr><th>Timestamp</th><th>Event</th><th>Payload</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></body></html>"
    )
