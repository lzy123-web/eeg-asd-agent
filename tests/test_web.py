from __future__ import annotations

import json

from fastapi.testclient import TestClient

from eeg_asd_agent.providers import OpenAIResponsesProvider
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


def test_web_rejects_runs_when_llm_key_is_missing(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(app)
    response = client.post(
        "/api/runs",
        files={"eeg": ("sample.edf", b"not-an-edf", "application/octet-stream")},
    )
    assert response.status_code == 503
