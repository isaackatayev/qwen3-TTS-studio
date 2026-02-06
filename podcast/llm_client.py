"""Unified LLM client abstraction supporting OpenAI, Ollama, and OpenRouter."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any


class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"


DEFAULT_MODELS = {
    LLMProvider.OPENAI: "gpt-5.2",
    LLMProvider.OLLAMA: "qwen3:8b",
    LLMProvider.OPENROUTER: "google/gemini-2.5-flash",
}

PROVIDER_BASE_URLS = {
    LLMProvider.OPENAI: "",
    LLMProvider.OLLAMA: "http://localhost:11434/v1",
    LLMProvider.OPENROUTER: "https://openrouter.ai/api/v1",
}


@dataclass
class LLMConfig:
    """Configuration for LLM client."""

    provider: LLMProvider
    model: str
    api_key: str
    base_url: str
    temperature: float = 0.4


def create_llm_client(config: LLMConfig) -> Any:
    """Create an OpenAI-compatible client for the specified provider."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "openai package is required. Install with: pip install openai"
        ) from e

    kwargs: dict[str, Any] = {}

    if config.provider == LLMProvider.OLLAMA:
        kwargs["api_key"] = "ollama"
    else:
        kwargs["api_key"] = config.api_key

    if config.base_url:
        kwargs["base_url"] = config.base_url

    return OpenAI(**kwargs)


def chat_completion(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.4,
    json_mode: bool = True,
    provider: LLMProvider = LLMProvider.OPENAI,
) -> str:
    """Call chat completion API with retry logic.

    Args:
        client: OpenAI-compatible client instance.
        model: Model identifier string.
        messages: Chat messages list.
        temperature: Sampling temperature.
        json_mode: Whether to request JSON output.
        provider: LLM provider for provider-specific handling.

    Returns:
        Response content string.

    Raises:
        ValueError: If the LLM returns empty or invalid content.
        openai errors: After exhausting retries for transient API errors.
    """
    from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

    max_retries = 3
    backoff_seconds = [1, 2, 4]

    for attempt in range(max_retries):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }

            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            try:
                response = client.chat.completions.create(**kwargs)
            except APIError as exc:
                status_code = getattr(exc, "status_code", None)
                response_format_rejected = (
                    json_mode
                    and "response_format" in kwargs
                    and (
                        _is_response_format_error(exc)
                        or (
                            provider != LLMProvider.OPENAI
                            and status_code == 400
                        )
                    )
                )
                if response_format_rejected:
                    fallback_kwargs = dict(kwargs)
                    fallback_kwargs.pop("response_format", None)
                    response = client.chat.completions.create(**fallback_kwargs)
                else:
                    raise

            if not response.choices or response.choices[0].message.content is None:
                raise ValueError(
                    f"LLM returned empty content (provider={provider.value}, model={model})"
                )
            content: str = response.choices[0].message.content

            if json_mode:
                try:
                    json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    content = _extract_json_from_text(content)

            return content

        except (RateLimitError, APITimeoutError, APIConnectionError):
            if attempt < max_retries - 1:
                time.sleep(backoff_seconds[attempt])
                continue
            raise
        except APIError as exc:
            status_code = getattr(exc, "status_code", None)
            is_retryable = status_code is None or status_code >= 500 or status_code == 429
            if is_retryable and attempt < max_retries - 1:
                time.sleep(backoff_seconds[attempt])
                continue
            raise

    raise RuntimeError("Unreachable: all retry attempts exhausted without return or raise")


def _extract_json_from_text(text: str) -> str:
    """Extract a JSON object from text that may contain prose or markdown fences.

    Raises:
        ValueError: If no valid JSON object can be extracted.
    """
    import re

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            json.loads(fenced.group(1))
            return fenced.group(1)
        except (json.JSONDecodeError, TypeError):
            pass

    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        pos = text.find("{", idx)
        if pos == -1:
            break
        try:
            obj, end_idx = decoder.raw_decode(text, pos)
            if isinstance(obj, dict):
                return text[pos:end_idx]
        except json.JSONDecodeError:
            pass
        idx = pos + 1

    raise ValueError("No valid JSON object found in LLM response.")


def _is_response_format_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "response_format" in message or "json_object" in message


def get_default_config(
    provider: LLMProvider,
    api_key: str = "",
    base_url: str = "",
    model: str = "",
) -> LLMConfig:
    """Create default LLM configuration for a provider."""
    if not model:
        model = DEFAULT_MODELS[provider]
    if not base_url:
        base_url = PROVIDER_BASE_URLS[provider]
    if provider == LLMProvider.OLLAMA and not api_key:
        api_key = "ollama"

    return LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )


def validate_connection(config: LLMConfig) -> tuple[bool, str]:
    """Validate that the LLM configuration works."""
    try:
        client = create_llm_client(config)
        response = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": "Respond with a single word: OK"}],
            temperature=0.0,
        )
        if not response.choices or not response.choices[0].message.content:
            return False, "Received empty response"
        return True, f"Successfully connected to {config.provider.value}"

    except ImportError as e:
        return False, f"Missing dependency: {e}"
    except Exception as e:
        return False, f"Connection failed: {e}"
