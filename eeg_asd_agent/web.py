from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

from eeg_asd_agent.providers import OpenAIResponsesProvider, ProviderConfigurationError
from eeg_asd_agent.schemas import AssessmentCaseState, CaseStatus, InputKind, RequestedGoal
from eeg_asd_agent.workflow import run_workflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "web"
RUN_ROOT = PROJECT_ROOT / "artifacts" / "web-runs"
LOG_ROOT = PROJECT_ROOT / "logs"
ALLOWED_SUFFIXES = {".edf", ".npz", ".mat"}
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024

app = FastAPI(title="EEG-ASD LLM Agent", docs_url=None, redoc_url=None)
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _job_for(run_id: str) -> dict[str, Any]:
    with _jobs_lock:
        live = dict(_jobs.get(run_id, {}))
    if live:
        return live
    matches = sorted(LOG_ROOT.glob(f"*_{run_id}"), reverse=True)
    if not matches:
        return {}
    output_dir = matches[0]
    summary_path = output_dir / "run_summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        summary = {}
    return {
        "run_id": run_id,
        "status": summary.get("status", "completed"),
        "file_name": summary.get("file_name", "unknown"),
        "error": summary.get("error"),
        "output_dir": str(output_dir),
        "log_dir_name": output_dir.name,
        "bundle_path": str(RUN_ROOT / run_id / "trace-bundle.zip"),
    }


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
    log_dir = LOG_ROOT / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{run_id}"
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
            "output_dir": str(log_dir),
            "log_dir_name": log_dir.name,
            "bundle_path": str(run_dir / "trace-bundle.zip"),
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
    with _jobs_lock:
        job = dict(_jobs[run_id])
    output_dir = Path(str(job["output_dir"]))
    bundle_path = Path(str(job["bundle_path"]))
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
        report_ready = (output_dir / "report.html").is_file()
        completed = state.execution_status is CaseStatus.COMPLETED and (
            state.requested_goal is not RequestedGoal.FULL_ASSESSMENT or report_ready
        )
        with _jobs_lock:
            _jobs[run_id].update(
                {
                    "status": "completed" if completed else "failed",
                    "error": None
                    if completed
                    else "; ".join(state.warnings[-3:])
                    or "Agent 未满足完整评估的完成条件",
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
    finally:
        output_dir.mkdir(parents=True, exist_ok=True)
        with _jobs_lock:
            summary = {
                "run_id": run_id,
                "file_name": _jobs[run_id]["file_name"],
                "status": _jobs[run_id]["status"],
                "error": _jobs[run_id].get("error"),
                "log_directory": _jobs[run_id]["log_dir_name"],
            }
        (output_dir / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.make_archive(
            str(bundle_path.with_suffix("")),
            "zip",
            root_dir=output_dir,
        )


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, object]:
    job = _job_for(run_id)
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
        "trace_ready": (output_dir / "trace.html").is_file(),
        "trace_html_url": f"/api/runs/{run_id}/trace.html",
        "log_bundle_url": f"/api/runs/{run_id}/logs.zip",
        "log_directory": job.get("log_dir_name"),
    }


def _report_path(run_id: str, name: str) -> Path:
    job = _job_for(run_id)
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


@app.get("/api/runs/{run_id}/trace.html", response_class=HTMLResponse)
def trace_html(run_id: str) -> HTMLResponse:
    return HTMLResponse(_report_path(run_id, "trace.html").read_text(encoding="utf-8"))


@app.get("/api/runs/{run_id}/logs.zip")
def logs_bundle(run_id: str) -> FileResponse:
    job = _job_for(run_id)
    if not job:
        raise HTTPException(status_code=404, detail="运行记录不存在")
    path = Path(str(job["bundle_path"]))
    if not path.is_file():
        output_dir = Path(str(job["output_dir"]))
        if not output_dir.is_dir():
            raise HTTPException(status_code=404, detail="日志包尚未生成")
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.make_archive(str(path.with_suffix("")), "zip", root_dir=output_dir)
    return FileResponse(
        path,
        media_type="application/zip",
        filename=f"eeg-asd-trace-{run_id}.zip",
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
