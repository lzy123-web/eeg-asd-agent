from __future__ import annotations

import json

from fastapi.testclient import TestClient

from eeg_asd_agent.providers import OpenAIResponsesProvider
from eeg_asd_agent import web
from eeg_asd_agent.web import app


def test_deepseek_non_secret_config_is_the_provider_default(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_STYLE", raising=False)
    provider = OpenAIResponsesProvider.from_env()
    assert provider.model == "deepseek-flash"
    assert provider.base_url == "https://api.deepseek.com"
    assert provider.api_style == "responses"


def test_provider_schema_removes_formats_unsupported_by_deepseek() -> None:
    schema, _ = OpenAIResponsesProvider._schema("report")

    def formats(node: object) -> list[str]:
        if isinstance(node, dict):
            own = [str(node["format"])] if "format" in node else []
            return own + [item for value in node.values() for item in formats(value)]
        if isinstance(node, list):
            return [item for value in node for item in formats(value)]
        return []

    assert "date-time" not in formats(schema)


def test_web_home_and_health_do_not_expose_api_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "never-return-this-secret")
    client = TestClient(app)
    home = client.get("/")
    assert home.status_code == 200
    assert "EEG-ASD 评估 Agent" in home.text
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["llm"]["configured"] is True
    assert "never-return-this-secret" not in json.dumps(health.json())
    assert "下载完整日志包" in home.text


def test_web_serves_trace_bundle_after_process_state_is_lost(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(web, "LOG_ROOT", tmp_path / "logs")
    monkeypatch.setattr(web, "RUN_ROOT", tmp_path / "web-runs")
    output = web.LOG_ROOT / "20261012-120000_test-run"
    output.mkdir(parents=True)
    (output / "trace.html").write_text("<h1>trace</h1>", encoding="utf-8")
    (output / "run_summary.json").write_text(
        json.dumps(
            {
                "run_id": "test-run",
                "file_name": "sample.edf",
                "status": "completed",
                "error": None,
                "log_directory": output.name,
            }
        ),
        encoding="utf-8",
    )
    bundle = web.RUN_ROOT / "test-run" / "trace-bundle.zip"
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(b"PK-test")
    web._jobs.pop("test-run", None)
    try:
        client = TestClient(app)
        assert client.get("/api/runs/test-run").json()["log_directory"] == output.name
        assert client.get("/api/runs/test-run/trace.html").status_code == 200
        response = client.get("/api/runs/test-run/logs.zip")
        assert response.status_code == 200
        assert response.content == b"PK-test"
    finally:
        web._jobs.pop("test-run", None)


def test_web_rejects_runs_when_llm_key_is_missing(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(app)
    response = client.post(
        "/api/runs",
        files={"eeg": ("sample.edf", b"not-an-edf", "application/octet-stream")},
    )
    assert response.status_code == 503
