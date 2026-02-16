# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — LLM Provider Adapters
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Unified LLM provider protocol with OpenAI, Anthropic, HuggingFace,
and local server adapters.

Each provider implements ``generate_candidates(prompt, n)`` returning
a list of ``{"text": str, "source": str}`` dicts — the same interface
as ``MockGenerator`` and ``LLMGenerator``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import requests

logger = logging.getLogger("DirectorAI.Providers")


class LLMProvider(ABC):
    """Abstract base for LLM provider adapters."""

    @abstractmethod
    def generate_candidates(self, prompt: str, n: int = 3) -> list[dict[str, str]]:
        """Generate n candidate responses for the given prompt."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        ...


class OpenAIProvider(LLMProvider):
    """OpenAI ChatCompletion API adapter.

    Parameters
    ----------
    api_key : str — OpenAI API key.
    model : str — model name (default: gpt-4o-mini).
    base_url : str — API base URL (for Azure/compatible endpoints).
    temperature : float — sampling temperature.
    max_tokens : int — max tokens per completion.
    timeout : int — request timeout seconds.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.8,
        max_tokens: int = 512,
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    @property
    def name(self) -> str:
        return f"openai/{self.model}"

    def generate_candidates(self, prompt: str, n: int = 3) -> list[dict[str, str]]:
        candidates = []
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "n": n,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            resp = requests.post(
                url, json=payload, headers=headers, timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            for choice in data.get("choices", []):
                text = choice.get("message", {}).get("content", "")
                candidates.append({"text": text, "source": self.name})
        except Exception as e:
            logger.error("OpenAI request failed: %s", e)
            candidates.append({"text": f"[Error: {e}]", "source": "error"})

        return candidates


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API adapter.

    Parameters
    ----------
    api_key : str — Anthropic API key.
    model : str — model ID (default: claude-sonnet-4-5-20250929).
    max_tokens : int — max tokens per message.
    timeout : int — request timeout seconds.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-sonnet-4-5-20250929",
        max_tokens: int = 512,
        timeout: int = 60,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout

    @property
    def name(self) -> str:
        return f"anthropic/{self.model}"

    def generate_candidates(self, prompt: str, n: int = 3) -> list[dict[str, str]]:
        candidates = []
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
        }

        for _ in range(n):
            try:
                resp = requests.post(
                    url, json=payload, headers=headers, timeout=self.timeout
                )
                resp.raise_for_status()
                data = resp.json()
                text = ""
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        text += block.get("text", "")
                candidates.append({"text": text, "source": self.name})
            except Exception as e:
                logger.error("Anthropic request failed: %s", e)
                candidates.append({"text": f"[Error: {e}]", "source": "error"})

        return candidates


class HuggingFaceProvider(LLMProvider):
    """HuggingFace Inference API adapter.

    Parameters
    ----------
    api_key : str — HuggingFace API token.
    model : str — model ID on HuggingFace Hub.
    timeout : int — request timeout seconds.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "mistralai/Mistral-7B-Instruct-v0.3",
        timeout: int = 60,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @property
    def name(self) -> str:
        return f"huggingface/{self.model}"

    def generate_candidates(self, prompt: str, n: int = 3) -> list[dict[str, str]]:
        candidates = []
        url = f"https://api-inference.huggingface.co/models/{self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        for _ in range(n):
            try:
                resp = requests.post(
                    url,
                    json={"inputs": prompt, "parameters": {"max_new_tokens": 256}},
                    headers=headers,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, list) and data:
                    text = data[0].get("generated_text", "")
                else:
                    text = str(data)
                candidates.append({"text": text, "source": self.name})
            except Exception as e:
                logger.error("HuggingFace request failed: %s", e)
                candidates.append({"text": f"[Error: {e}]", "source": "error"})

        return candidates


class LocalProvider(LLMProvider):
    """Local inference server adapter (llama.cpp, vLLM, Ollama).

    Parameters
    ----------
    api_url : str — server endpoint URL.
    model : str — model name (for servers that require it).
    timeout : int — request timeout seconds.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8080/v1/chat/completions",
        model: str = "",
        timeout: int = 30,
    ) -> None:
        self.api_url = api_url
        self.model = model
        self.timeout = timeout

    @property
    def name(self) -> str:
        return f"local/{self.model or 'default'}"

    def generate_candidates(self, prompt: str, n: int = 3) -> list[dict[str, str]]:
        candidates = []
        payload: dict = {
            "messages": [{"role": "user", "content": prompt}],
            "n": n,
            "temperature": 0.8,
            "max_tokens": 512,
        }
        if self.model:
            payload["model"] = self.model

        try:
            resp = requests.post(self.api_url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            for choice in data.get("choices", []):
                text = choice.get("message", {}).get("content", "")
                candidates.append({"text": text, "source": self.name})
        except Exception as e:
            logger.error("Local provider request failed: %s", e)
            candidates.append({"text": f"[Error: {e}]", "source": "error"})

        return candidates
