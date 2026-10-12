from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eeg_asd_agent.schemas import (
    ActionDecision,
    AgentTurnDecision,
    FinalAssessmentReportSpec,
    PlanSpec,
)


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
    max_output_tokens: int = 8192
    api_style: str = "responses"
    name: str = "openai_responses"
    last_usage: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> OpenAIResponsesProvider:
        config_path = Path(
            os.environ.get(
                "EEG_ASD_LLM_CONFIG",
                str(Path(__file__).resolve().parents[2] / "config" / "llm.json"),
            )
        )
        config: dict[str, Any] = {}
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
        api_key = os.environ.get("OPENAI_API_KEY", "")
        model = os.environ.get("OPENAI_MODEL", str(config.get("model", "")))
        if not api_key:
            raise ProviderConfigurationError("OPENAI_API_KEY is not configured")
        if not model:
            raise ProviderConfigurationError("OPENAI_MODEL is not configured")
        api_style = os.environ.get(
            "OPENAI_API_STYLE", str(config.get("api_style", "responses"))
        ).strip().lower()
        if api_style not in {"responses", "chat_completions"}:
            raise ProviderConfigurationError(
                "OPENAI_API_STYLE must be 'responses' or 'chat_completions'"
            )
        return cls(
            api_key=api_key,
            model=model,
            base_url=os.environ.get(
                "OPENAI_BASE_URL",
                str(config.get("base_url", "https://api.openai.com/v1")),
            ).rstrip("/"),
            temperature=float(
                os.environ.get("OPENAI_TEMPERATURE", str(config.get("temperature", 0)))
            ),
            max_output_tokens=int(
                os.environ.get(
                    "OPENAI_MAX_OUTPUT_TOKENS",
                    str(config.get("max_output_tokens", 8192)),
                )
            ),
            api_style=api_style,
            name=(
                "openai_chat_completions"
                if api_style == "chat_completions"
                else "openai_responses"
            ),
        )

    def generate_json(self, purpose: str, payload: dict[str, object]) -> str:
        if self.api_style == "chat_completions":
            return self._generate_chat_completions(purpose, payload)
        return self._generate_responses(purpose, payload)

    def _generate_responses(self, purpose: str, payload: dict[str, object]) -> str:
        schema, schema_name = self._schema(purpose)
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
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

    def _generate_chat_completions(
        self,
        purpose: str,
        payload: dict[str, object],
    ) -> str:
        schema, _schema_name = self._schema(purpose)
        strict_schema = self._strict_schema(schema)
        system_prompt = (
            self._system_prompt(purpose)
            + "\nThe JSON response must validate against this schema:\n"
            + json.dumps(strict_schema, ensure_ascii=False, separators=(",", ":"))
        )
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                },
            ],
            "response_format": {"type": "json_object"},
        }
        thinking = os.environ.get("OPENAI_ENABLE_THINKING")
        if thinking is not None:
            body["enable_thinking"] = thinking.strip().lower() in {"1", "true", "yes"}
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
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
            raise ProviderResponseError(
                f"Chat Completions API HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ProviderResponseError(
                f"Chat Completions API unavailable: {exc.reason}"
            ) from exc
        usage = parsed.get("usage", {})
        self.last_usage = {
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        }
        try:
            content = parsed["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderResponseError(
                "Chat Completions API returned no assistant content"
            ) from exc
        if not isinstance(content, str):
            raise ProviderResponseError(
                "Chat Completions API returned non-text assistant content"
            )
        normalized = self._extract_json_object(content)
        json.loads(normalized)
        return normalized

    @staticmethod
    def _extract_json_object(raw: str) -> str:
        stripped = raw.strip()
        if stripped.startswith("```"):
            stripped = stripped.removeprefix("```json").removeprefix("```")
            stripped = stripped.removesuffix("```").strip()
        first = stripped.find("{")
        last = stripped.rfind("}")
        if first < 0 or last <= first:
            raise ProviderResponseError("provider returned no JSON object")
        return stripped[first : last + 1]

    @staticmethod
    def _schema(purpose: str) -> tuple[dict[str, Any], str]:
        if purpose == "agent_turn":
            return OpenAIResponsesProvider._strict_schema(
                AgentTurnDecision.model_json_schema()
            ), "agent_turn"
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
                # DeepSeek supports only a small subset of JSON Schema string
                # formats. Pydantic still validates these values after the
                # response, so unsupported annotations can be omitted here.
                if node.get("format") not in {
                    None,
                    "email",
                    "hostname",
                    "ipv4",
                    "ipv6",
                    "uuid",
                }:
                    node.pop("format", None)
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
        if purpose == "agent_turn":
            return (
                "You are the controller of a specialist EEG-ASD research Agent. "
                "On every turn, update a concise high-level plan and choose at most one tool from "
                "allowed_next_actions. A selected tool must appear in a current plan step and its action "
                "must be non-terminal; observe the result on the next turn before finishing. The harness "
                "validates and executes but never chooses workflow order. "
                "Use the user request, case state, observations, memory, and capability manifests to decide "
                "whether to inspect, reuse, compare, recompute, retrieve, report, or finish. Existing assets "
                "are optional evidence and never imply automatic skipping. RAG can be called whenever it helps "
                "the current goal. Never expose raw EEG, local paths, feature values, secrets, or internal "
                "ToolResult objects. Use retrieved evidence only by its exact Evidence ID. "
                "For a full assessment, do not finish until generate_report_spec and render_report have "
                "succeeded. A no-tool action requires goal_satisfied=true. Do not diagnose ASD or "
                "prescribe treatment. Return only AgentTurnDecision JSON."
            )
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
                "numbers, sources, or capabilities absent from the payload. Cite only exact Evidence IDs "
                "returned by retrieve_asd_knowledge."
            )
        return (
            "Return only FinalAssessmentReportSpec. Use only visible_results and capability_context. "
            "Do not diagnose, invent numbers, expose paths, or claim an unexecuted tool succeeded. "
            "Every knowledge statement must use exact retrieved Evidence IDs; never invent an ID. "
            "Organize data_summary.details using the EEGAgent report concepts when supported: "
            "patient_information, history, technical_description, eeg_description, impression, "
            "clinical_correlation, and conclusion. Leave unsupported content explicitly unavailable."
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
