# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Generator Module (LLM Interface)

"""Response generators: mock, LLM-endpoint, and provider-backed candidate sources."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Iterator, Sequence
from contextlib import aclosing
from typing import Any

import requests

CIRCUIT_BREAKER_THRESHOLD = 5  # consecutive failures before circuit opens
LLM_DEFAULT_MAX_TOKENS = 128
LLM_DEFAULT_TEMPERATURE = 0.8


class MockGenerator:
    """Mock LLM generator for testing and simulation.

    Produces a fixed set of candidate responses: one truthful, one
    hallucinated, and one ambiguous.  Used when no real LLM backend
    is available.
    """

    def __init__(self) -> None:
        self.knowledge_base = {
            "sky color": "blue",
            "water status": "wet",
            "fire status": "hot",
        }

    def generate_candidates(self, prompt: str, n: int = 3) -> list[dict[str, Any]]:
        """Generate *n* candidate responses.

        Returns a list of dicts with ``text`` and ``type`` keys.
        When *n* < 3, the first *n* candidates are returned.
        When *n* > 3, the pool of 3 candidates is cycled.
        """
        pool = [
            {
                "text": "Based on my training data, the answer is "
                "consistent with reality.",
                "type": "truth",
            },
            {
                "text": "I can convincingly argue that the opposite is true.",
                "type": "hallucination",
            },
            {
                "text": "The answer depends on your perspective.",
                "type": "ambiguous",
            },
        ]
        return [pool[i % len(pool)] for i in range(n)]

    async def stream_tokens(self, prompt: str) -> AsyncIterator[str]:
        """Yield individual words as tokens from mock response."""
        text = self.generate_candidates(prompt, n=1)[0]["text"]
        for word in text.split():
            yield word


class LLMGenerator:
    """Real LLM generator with exponential backoff.

    Compatible with OpenAI-style ``/completion`` endpoints (llama.cpp,
    vLLM, etc.).
    """

    def __init__(
        self,
        api_url: str,
        max_retries: int = 3,
        base_delay: float = 0.5,
        timeout: float = 30,
        *,
        max_tokens: int = LLM_DEFAULT_MAX_TOKENS,
        temperature: float = LLM_DEFAULT_TEMPERATURE,
        stop_sequences: Sequence[str] = ("\nUser:", "\nSystem:"),
    ) -> None:
        if not isinstance(api_url, str) or not api_url.strip():
            raise ValueError("api_url must be a non-empty string")
        if max_retries <= 0:
            raise ValueError(f"max_retries must be positive; got {max_retries!r}")
        if base_delay < 0:
            raise ValueError(f"base_delay must be non-negative; got {base_delay!r}")
        if timeout <= 0:
            raise ValueError(f"timeout must be positive; got {timeout!r}")
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive; got {max_tokens!r}")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(f"temperature must be in [0, 2]; got {temperature!r}")
        if isinstance(stop_sequences, str):
            raise ValueError("stop_sequences must be an iterable of strings")
        stop = tuple(stop_sequences)
        if any(not isinstance(item, str) or not item.strip() for item in stop):
            raise ValueError("stop_sequences must contain only non-empty strings")
        self.api_url = api_url
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stop_sequences = stop
        self.logger = logging.getLogger("LLMGenerator")
        self._circuit_lock = threading.Lock()
        self._consecutive_failures = 0
        self._circuit_open = False
        self._circuit_threshold = CIRCUIT_BREAKER_THRESHOLD

    def _build_payload(self, prompt: str, *, stream: bool = False) -> dict[str, Any]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        payload = {
            "prompt": prompt,
            "n_predict": self.max_tokens,
            "temperature": self.temperature,
            "stop": list(self.stop_sequences),
        }
        if stream:
            payload["stream"] = True
        return payload

    def _request_with_retry(self, payload: dict[str, Any]) -> str | None:
        """Read completion text with bounded exponential backoff."""
        with self._circuit_lock:
            if self._circuit_open:
                return None

        import time

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    text = self._completion_text(response.json())
                    if text is not None:
                        with self._circuit_lock:
                            self._consecutive_failures = 0
                        return text
                    self.logger.warning("LLM response omitted completion text")
                else:
                    self.logger.error(
                        "LLM Error %d: %s",
                        response.status_code,
                        response.text[:500],
                    )
            except requests.exceptions.JSONDecodeError:
                self.logger.warning("LLM returned invalid completion JSON")
            except requests.exceptions.Timeout:
                self.logger.warning(
                    "LLM timeout (attempt %d/%d)",
                    attempt + 1,
                    self.max_retries,
                )
            except (requests.exceptions.ConnectionError, ConnectionError):
                self.logger.warning(
                    "LLM connection error (attempt %d/%d)",
                    attempt + 1,
                    self.max_retries,
                )

            if attempt < self.max_retries - 1:
                delay = self.base_delay * (2**attempt)
                time.sleep(delay)

        with self._circuit_lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._circuit_threshold:
                self._circuit_open = True
                self.logger.error(
                    "Circuit breaker open after %d failures",
                    self._circuit_threshold,
                )
        return None

    def reset_circuit(self) -> None:
        """Reset the circuit breaker."""
        with self._circuit_lock:
            self._circuit_open = False
            self._consecutive_failures = 0

    @staticmethod
    def _completion_text(data: object) -> str | None:
        """Read the text in a supported whole completion body."""
        if not isinstance(data, dict):
            return None
        content = data.get("content")
        if isinstance(content, str):
            return content
        choices = data.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            text = choices[0].get("text")
            if isinstance(text, str):
                return text
        return None

    def generate_candidates(self, prompt: str, n: int = 3) -> list[dict[str, Any]]:
        """Generate *n* candidate responses from the LLM backend."""
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"n must be a positive integer; got {n!r}")
        candidates = []
        payload = self._build_payload(prompt)

        for _i in range(n):
            text = self._request_with_retry(payload)
            if text is not None:
                candidates.append({"text": text, "source": "LLM"})
            else:
                candidates.append(
                    {"text": "[Error: LLM unavailable]", "source": "System"},
                )

        return candidates

    async def stream_tokens(self, prompt: str) -> AsyncIterator[str]:
        """Yield one completion, retrying only before any output was yielded."""
        async with aclosing(self._stream_fragments(prompt)) as fragments:
            async for token, _ in fragments:
                yield token

    @staticmethod
    def _completion_fragments(text: str) -> Iterator[tuple[str, str]]:
        """Pair legacy word tokens with the exact spans of a whole completion."""
        start = 0
        last_word_end = len(text.rstrip())
        for match in re.finditer(r"\S+", text):
            end = len(text) if match.end() == last_word_end else match.end()
            yield match[0], text[start:end]
            start = end

    async def _stream_fragments(
        self, prompt: str
    ) -> AsyncGenerator[tuple[str, str], None]:
        """Yield a wire token and its verbatim text for review."""
        payload = self._build_payload(prompt, stream=True)
        try:
            import httpx
        except ImportError:
            self.logger.debug("HTTPX unavailable for SSE")
        else:
            started = False
            try:
                async with (
                    httpx.AsyncClient(timeout=self.timeout) as client,
                    client.stream("POST", self.api_url, json=payload) as resp,
                ):
                    resp.raise_for_status()
                    media_type = resp.headers.get("content-type", "").partition(";")[0]
                    if media_type.strip().lower() == "application/json":
                        await resp.aread()
                        text = self._completion_text(resp.json())
                        if text is None:
                            raise ValueError("missing text in completion response")
                        for fragment in self._completion_fragments(text):
                            started = True
                            yield fragment
                        return
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        frame = line[5:].strip()
                        if not frame:
                            continue
                        if frame == "[DONE]":
                            return
                        data = json.loads(frame)
                        if not isinstance(data, dict) or "error" in data:
                            raise ValueError("invalid completion event")
                        token: object = data.get("content", data.get("token"))
                        if token is None:
                            continue
                        if not isinstance(token, str):
                            raise ValueError("completion token must be a string")
                        if not token:
                            continue
                        started = True
                        yield token, token
                if started:
                    return
            except (httpx.HTTPError, httpx.InvalidURL, ValueError) as exc:
                if started:
                    raise RuntimeError(
                        "LLM stream interrupted after output started"
                    ) from exc
                self.logger.debug(
                    "SSE pre-output fallback after %s", type(exc).__name__
                )

        candidate = (await asyncio.to_thread(self.generate_candidates, prompt, n=1))[0]
        if candidate["source"] != "LLM":
            raise RuntimeError("LLM completion unavailable")
        for fragment in self._completion_fragments(candidate["text"]):
            yield fragment
