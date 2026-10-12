import json
from pathlib import Path
from typing import Any

import pytest

from eeg_asd_agent.dashboard import build_dashboard
from eeg_asd_agent.providers import OfflineProvider, OpenAIResponsesProvider
from eeg_asd_agent.providers.openai_compatible import ProviderConfigurationError
from eeg_asd_agent.schemas import PlanSpec, PlanStep, RequestedGoal
from eeg_asd_agent.tracing import TraceLogger, render_trace_html
from eeg_asd_agent.workflow import plan_to_mermaid


def test_openai_provider_requires_explicit_key_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with pytest.raises(ProviderConfigurationError):
        OpenAIResponsesProvider.from_env()


def test_openai_provider_parses_responses_output(monkeypatch: pytest.MonkeyPatch) -> None:
    report = OfflineProvider().generate_json(
        "report",
        {"case_id": "provider-test", "visible_results": []},
    )
    response_body = json.dumps(
        {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": report}],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        }
    ).encode()
    captured: dict[str, Any] = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return response_body

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = OpenAIResponsesProvider(api_key="test-secret", model="test-model")
    output = provider.generate_json(
        "report",
        {"case_id": "provider-test", "visible_results": []},
    )
    assert json.loads(output)["case_id"] == "provider-test"
    assert provider.last_usage["total_tokens"] == 30
    body = json.loads(captured["request"].data.decode())
    assert body["text"]["format"]["type"] == "json_schema"
    assert body["text"]["format"]["strict"] is True


def test_openai_provider_supports_chat_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    report = OfflineProvider().generate_json(
        "report",
        {"case_id": "chat-provider-test", "visible_results": []},
    )
    response_body = json.dumps(
        {
            "choices": [{"message": {"role": "assistant", "content": report}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10},
        }
    ).encode()
    captured: dict[str, Any] = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return response_body

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        captured["request"] = request
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = OpenAIResponsesProvider(
        api_key="test-secret",
        model="qwen-test",
        base_url="https://example.invalid/v1",
        api_style="chat_completions",
    )
    output = provider.generate_json(
        "report",
        {"case_id": "chat-provider-test", "visible_results": []},
    )
    assert json.loads(output)["case_id"] == "chat-provider-test"
    assert provider.last_usage["total_tokens"] == 10
    assert captured["request"].full_url.endswith("/chat/completions")
    body = json.loads(captured["request"].data.decode())
    assert body["response_format"] == {"type": "json_object"}
    assert "The JSON response must validate against this schema" in body["messages"][0]["content"]


def test_trace_html_and_plan_visualization(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    TraceLogger(trace_path, "t").event("started", {"safe": True})
    html = render_trace_html(trace_path)
    assert "<table>" in html
    assert "started" in html
    plan = PlanSpec(
        case_id="c",
        goal=RequestedGoal.QUALITY_CHECK,
        planner_kind="test",
        steps=[
            PlanStep(step_id="s1", tool_name="inspect_case"),
            PlanStep(step_id="s2", tool_name="check_data_quality", depends_on=["s1"]),
        ],
    )
    mermaid = plan_to_mermaid(plan)
    assert "s1 --> s2" in mermaid


def test_static_dashboard_builds_from_artifacts(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "benchmarks"
    demo_dir = tmp_path / "demo"
    benchmark_dir.mkdir()
    demo_dir.mkdir()
    (benchmark_dir / "benchmark_results.json").write_text(
        json.dumps({"suites": {"planning": {"case_count": 30}}}),
        encoding="utf-8",
    )
    (demo_dir / "report_spec.json").write_text(
        json.dumps(
            {
                "findings": [
                    {"title": "Quality", "statement": "Data quality passed."}
                ]
            }
        ),
        encoding="utf-8",
    )
    output = build_dashboard(tmp_path, tmp_path / "dashboard.html")
    content = output.read_text(encoding="utf-8")
    assert "EEG-ASD Workflow Agent" in content
    assert "planning" in content
    assert "fixture" in content
