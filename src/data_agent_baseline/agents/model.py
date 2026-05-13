from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from openai import APIError, OpenAI


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ModelStep:
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str


class ModelAdapter(Protocol):
    def complete(self, messages: list[ModelMessage]) -> str:
        raise NotImplementedError


class OpenAIModelAdapter:
    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float,
        timeout_seconds: float = 90.0,
        max_output_tokens: int | None = 2048,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens

    def complete(self, messages: list[ModelMessage]) -> str:
        if not self.api_base:
            raise RuntimeError("Missing model API base URL.")

        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key or "EMPTY",
            "base_url": self.api_base,
            "max_retries": 0,
        }
        if self.timeout_seconds > 0:
            client_kwargs["timeout"] = self.timeout_seconds
        client = OpenAI(**client_kwargs)

        try:
            request: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": message.role, "content": message.content} for message in messages],
                "temperature": self.temperature,
            }
            if self.max_output_tokens is not None and self.max_output_tokens > 0:
                request["max_tokens"] = self.max_output_tokens
            response = client.chat.completions.create(**request)
        except APIError as exc:
            raise RuntimeError(f"Model request failed: {exc}") from exc

        choices = response.choices or []
        if not choices:
            raise RuntimeError("Model response missing choices.")
        content = choices[0].message.content
        if not isinstance(content, str):
            raise RuntimeError("Model response missing text content.")
        return content


class ScriptedModelAdapter:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, messages: list[ModelMessage]) -> str:
        del messages
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)
