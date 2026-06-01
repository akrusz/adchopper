"""LLM backends for ad classification.

Each backend is a small callable: given the system prompt and the user
content (a numbered transcript window), it returns the model's raw text
response, which is expected to be JSON of the form
``{"ads": [{"start_line": int, "end_line": int, "reason": str}]}``.

Three backends are supported:

* ``ollama``    -- a local model served by Ollama (default; fully offline).
* ``anthropic`` -- the Claude API via the official ``anthropic`` SDK.
* ``openai``    -- the OpenAI API via the official ``openai`` SDK.

The cloud backends are optional dependencies (see ``pyproject.toml`` extras);
they're only imported when actually selected.
"""

from __future__ import annotations

import json
from typing import Callable

import requests


# Sensible default model per backend. Override with --llm-model.
DEFAULT_MODELS = {
    "ollama": "llama3.1:8b",
    "anthropic": "claude-opus-4-8",
    "openai": "gpt-4o-mini",
}

# Shared JSON schema for the structured response. The cloud backends use this
# to *guarantee* schema-valid JSON; Ollama uses plain JSON mode + parsing.
ADS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "ads": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["start_line", "end_line", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["ads"],
    "additionalProperties": False,
}

# A backend is just: (system_prompt, user_content) -> raw_text_response.
Backend = Callable[[str, str], str]


class OllamaBackend:
    """Local model served by Ollama's HTTP API."""

    def __init__(self, model: str, host: str, timeout: float):
        self.model = model
        self.host = host
        self.timeout = timeout

    def __call__(self, system: str, user: str) -> str:
        url = self.host.rstrip("/") + "/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
        except requests.exceptions.ConnectionError as e:
            raise SystemExit(
                f"Could not reach Ollama at {self.host}. Is it running? "
                f"(start it with `ollama serve` and `ollama pull {self.model}`)"
            ) from e
        if resp.status_code == 404:
            raise SystemExit(
                f"Ollama model '{self.model}' not found. "
                f"Pull it with: ollama pull {self.model}"
            )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


class AnthropicBackend:
    """Claude API via the official anthropic SDK.

    The large system prompt is identical across every transcript window, so we
    mark it with ``cache_control`` -- after the first call it is served from
    Anthropic's prompt cache at ~0.1x cost. Structured outputs
    (``output_config.format``) guarantee the response is schema-valid JSON.
    """

    def __init__(self, model: str, timeout: float):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - dependency hint
            raise SystemExit(
                "The anthropic SDK is not installed. "
                "Run: pip install 'adchopper[anthropic]'  (and set ANTHROPIC_API_KEY)"
            ) from e
        self._anthropic = anthropic
        self.model = model
        # Resolves ANTHROPIC_API_KEY from the environment.
        self.client = anthropic.Anthropic(timeout=timeout)

    def __call__(self, system: str, user: str) -> str:
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=[
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user}],
                output_config={
                    "format": {"type": "json_schema", "schema": ADS_JSON_SCHEMA}
                },
            )
        except self._anthropic.AuthenticationError as e:
            raise SystemExit(
                "Anthropic auth failed -- is ANTHROPIC_API_KEY set correctly?"
            ) from e
        # output_config.format guarantees the first text block is valid JSON.
        return next((b.text for b in resp.content if b.type == "text"), "")


class OpenAIBackend:
    """OpenAI API via the official openai SDK, using JSON-schema structured
    outputs for guaranteed-valid JSON."""

    def __init__(self, model: str, timeout: float):
        try:
            import openai
        except ImportError as e:  # pragma: no cover - dependency hint
            raise SystemExit(
                "The openai SDK is not installed. "
                "Run: pip install 'adchopper[openai]'  (and set OPENAI_API_KEY)"
            ) from e
        self._openai = openai
        self.model = model
        self.client = openai.OpenAI(timeout=timeout)

    def __call__(self, system: str, user: str) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ad_spans",
                        "schema": ADS_JSON_SCHEMA,
                        "strict": True,
                    },
                },
            )
        except self._openai.AuthenticationError as e:
            raise SystemExit(
                "OpenAI auth failed -- is OPENAI_API_KEY set correctly?"
            ) from e
        return resp.choices[0].message.content or ""


def get_backend(
    name: str, model: str, host: str, timeout: float
) -> Backend:
    """Construct the selected backend callable."""
    if name == "ollama":
        return OllamaBackend(model, host, timeout)
    if name == "anthropic":
        return AnthropicBackend(model, timeout)
    if name == "openai":
        return OpenAIBackend(model, timeout)
    raise SystemExit(
        f"Unknown backend '{name}'. Choose: ollama, anthropic, openai."
    )
