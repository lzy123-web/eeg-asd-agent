from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from eeg_asd_agent.schemas import ActionDecision, FinalAssessmentReportSpec, PlanSpec


class ProviderConfigurationError(RuntimeError):
    pass


class ProviderResponseError(RuntimeError):
    pass


@dataclass
class OpenAIResponsesProvider:
    """Minimal OpenAI-compatible Responses API provider with strict JSON Schema output."""

    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: float = 120
    temperature: float = 0.0
    name: str = "openai_responses"
    last_usage: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> OpenAIResponsesProvider:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        model = os.environ.get("OPENAI_MODEL", "")
        if not api_key:
            raise ProviderConfigurationError("OPENAI_API_KEY is not configured")
        if not model:
            raise ProviderConfigurationError("OPENAI_MODEL is not configured")
        return cls(
            api_key=api_key,
            model=model,
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            temperature=float(os.environ.get("OPENAI_TEMPERATURE", "0")),
        )

    def generate_json(self, purpose: str, payload: dict[str, object]) -> str:
        schema, schema_name = self._schema(purpose)
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": self._system_prompt(purpose),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                        }
                    ],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "description": f"Strict structured output for {purpose}",
                    "strict": True,
                    "schema": schema,
                }
            },
        }
        request = urllib.request.Request(
            f"{self.base_url}/responses",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                parsed = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise ProviderResponseError(f"Responses API HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ProviderResponseError(f"Responses API unavailable: {exc.reason}") from exc
        usage = parsed.get("usage", {})
        self.last_usage = {
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        }
        text = self._extract_output_text(parsed)
        json.loads(text)
        return text

    @staticmethod
    def _schema(purpose: str) -> tuple[dict[str, Any], str]:
        if purpose == "next_action":
            return OpenAIResponsesProvider._strict_schema(
                ActionDecision.model_json_schema()
            ), "next_action"
        if purpose == "planning":
            return OpenAIResponsesProvider._strict_schema(
                PlanSpec.model_json_schema()
            ), "plan_spec"
        if purpose in {"report", "repair"}:
            return OpenAIResponsesProvider._strict_schema(
                FinalAssessmentReportSpec.model_json_schema()
            ), "final_assessment_report_spec"
        raise ValueError(f"unsupported provider purpose: {purpose}")

    @staticmethod
    def _strict_schema(raw: dict[str, Any]) -> dict[str, Any]:
        """Normalize Pydantic JSON Schema for strict Structured Outputs."""
        schema = deepcopy(raw)

        def visit(node: Any) -> None:
            if isinstance(node, dict):
                node.pop("default", None)
                if node.get("type") == "object" or "properties" in node:
                    properties = node.get("properties", {})
                    if isinstance(properties, dict):
                        node["additionalProperties"] = False
                        node["required"] = list(properties)
                for value in node.values():
                    visit(value)
            elif isinstance(node, list):
                for value in node:
                    visit(value)

        visit(schema)
        return schema

    @staticmethod
    def _system_prompt(purpose: str) -> str:
        if purpose == "next_action":
            return (
                "Return exactly one ActionDecision. Choose only allowed_next_actions and legal arguments. "
                "Base the decision on current_state and previous_observations. Never expose or request raw data."
            )
        if purpose == "planning":
            return (
                "Return only a PlanSpec. Use only allowed_tool_names. Respect completed workflows, "
                "tool preconditions, dependencies, max steps, and stop conditions."
            )
        if purpose == "repair":
            return (
                "Repair the supplied report into FinalAssessmentReportSpec. Do not add facts, "
                "numbers, sources, or capabilities absent from the payload."
            )
        return (
            "Return only FinalAssessmentReportSpec. Use only visible_results and capability_context. "
            "Do not diagnose, invent numbers, expose paths, or claim an unexecuted tool succeeded."
        )

    @staticmethod
    def _extract_output_text(response: dict[str, Any]) -> str:
        if isinstance(response.get("output_text"), str):
            return str(response["output_text"])
        for output in response.get("output", []):
            if not isinstance(output, dict):
                continue
            for content in output.get("content", []):
                if isinstance(content, dict) and content.get("type") == "output_text":
                    text = content.get("text")
                    if isinstance(text, str):
                        return text
        raise ProviderResponseError("Responses API returned no output_text content")
