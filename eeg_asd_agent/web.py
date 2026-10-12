from __future__ import annotations

import argparse
import getpass
import json
import os
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

from eeg_asd_agent.providers import OpenAIResponsesProvider, ProviderConfigurationError
from eeg_asd_agent.schemas import AssessmentCaseState, InputKind
from eeg_asd_agent.workflow import run_workflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "web"
RUN_ROOT = PROJECT_ROOT / "artifacts" / "web-runs"
ALLOWED_SUFFIXES = {".edf", ".npz", ".mat"}
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024

app = FastAPI(title="EEG-ASD LLM Agent", docs_url=None, redoc_url=None)
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _provider_summary() -> dict[str, object]:
    try:
        provider = OpenAIResponsesProvider.from_env()
    except ProviderConfigurationError:
        return {"configured": False, "model": None, "provider": "deepseek"}
    return {
        "configured": True,
        "model": provider.model,
        "provider": "deepseek",
        "api_style": provider.api_style,
    }


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/api/health")
def health() -> dict[str, object]:
    return {"status": "ok", "llm": _provider_summary()}


@app.post("/api/runs")
async def create_run(
    eeg: UploadFile = File(...),
    age: float | None = Form(default=None),
    sex: str | None = Form(default=None),
    request: str = Form(default="分析 EEG，并生成 ASD 儿童研究性评估报告"),
) -> dict[str, str]:
    if not _provider_summary()["configured"]:
        raise HTTPException(status_code=503, detail="DeepSeek API Key 未配置")
    original_name = Path(eeg.filename or "recording.edf").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="仅支持 EDF、NPZ 和 MAT 文件")
    if age is not None and not 0 <= age <= 120:
        raise HTTPException(status_code=400, detail="年龄应在 0 至 120 岁之间")

    run_id = uuid4().hex
    run_dir = RUN_ROOT / run_id
    upload_dir = run_dir / "input"
    upload_dir.mkdir(parents=True, exist_ok=False)
    eeg_path = upload_dir / original_name
    total = 0
    with eeg_path.open("wb") as output:
        while chunk := await eeg.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                output.close()
                eeg_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="EEG 文件不能超过 1 GB")
            output.write(chunk)
    await eeg.close()

    with _jobs_lock:
        _jobs[run_id] = {
            "run_id": run_id,
            "status": "queued",
            "file_name": original_name,
            "error": None,
            "output_dir": str(run_dir / "output"),
        }
    worker = threading.Thread(
        target=_run_agent,
        args=(run_id, eeg_path, age, sex, request),
        daemon=True,
        name=f"eeg-agent-{run_id[:8]}",
    )
    worker.start()
    return {"run_id": run_id, "status_url": f"/api/runs/{run_id}"}


def _run_agent(
    run_id: str,
    eeg_path: Path,
    age: float | None,
    sex: str | None,
    request: str,
) -> None:
    output_dir = RUN_ROOT / run_id / "output"
    with _jobs_lock:
        _jobs[run_id]["status"] = "running"
    metadata: dict[str, object] = {}
    if age is not None:
        metadata["age"] = age
    if sex:
        metadata["sex"] = sex
    state = AssessmentCaseState(
        case_id=f"web-{run_id[:12]}",
        input_type=InputKind.RAW_EEG,
        raw_files=[str(eeg_path.resolve())],
        modalities=["EEG"],
        metadata_summary=metadata,
    )
    try:
        provider = OpenAIResponsesProvider.from_env()
        artifacts = run_workflow(
            state,
            request,
            output_dir,
            planner_kind="agent",
            provider=provider,
        )
        with _jobs_lock:
            _jobs[run_id].update(
                {
                    "status": "completed",
                    "artifacts": {key: str(value) for key, value in artifacts.items()},
                }
            )
    except Exception as exc:
        with _jobs_lock:
            _jobs[run_id].update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, object]:
    with _jobs_lock:
        job = dict(_jobs.get(run_id, {}))
    if not job:
        raise HTTPException(status_code=404, detail="运行记录不存在")
    output_dir = Path(str(job["output_dir"]))
    trace_path = output_dir / "trace.jsonl"
    steps: list[dict[str, object]] = []
    if trace_path.is_file():
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event_type") != "agent_tool_observed":
                continue
            payload = event.get("payload", {})
            result = payload.get("result", {}) if isinstance(payload, dict) else {}
            steps.append(
                {
                    "tool": result.get("tool_name", "unknown"),
                    "status": result.get("status", "unknown"),
                    "duration_ms": result.get("duration_ms"),
                }
            )
    return {
        "run_id": run_id,
        "status": job["status"],
        "file_name": job["file_name"],
        "error": job.get("error"),
        "steps": steps,
        "report_ready": (output_dir / "report.html").is_file(),
        "report_html_url": f"/api/runs/{run_id}/report.html",
        "report_markdown_url": f"/api/runs/{run_id}/report.md",
    }


def _report_path(run_id: str, name: str) -> Path:
    with _jobs_lock:
        job = dict(_jobs.get(run_id, {}))
    if not job:
        raise HTTPException(status_code=404, detail="运行记录不存在")
    path = Path(str(job["output_dir"])) / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="报告尚未生成")
    return path


@app.get("/api/runs/{run_id}/report.html", response_class=HTMLResponse)
def report_html(run_id: str) -> HTMLResponse:
    return HTMLResponse(_report_path(run_id, "report.html").read_text(encoding="utf-8"))


@app.get("/api/runs/{run_id}/report.md", response_class=PlainTextResponse)
def report_markdown(run_id: str) -> PlainTextResponse:
    return PlainTextResponse(
        _report_path(run_id, "report.md").read_text(encoding="utf-8"),
        media_type="text/markdown; charset=utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="EEG-ASD LLM Agent local web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("DeepSeek API Key: ").strip()
    if not os.environ["OPENAI_API_KEY"]:
        raise SystemExit("DeepSeek API Key is required")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
