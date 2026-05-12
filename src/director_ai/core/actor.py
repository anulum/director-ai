# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Generator Module (LLM Interface)

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence

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

    def __init__(self):
        self.knowledge_base = {
            "sky color": "blue",
            "water status": "wet",
            "fire status": "hot",
        }

    def generate_candidates(self, prompt, n=3) -> list[dict]:
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
        api_url,
        max_retries=3,
        base_delay=0.5,
        timeout=30,
        *,
        max_tokens: int = LLM_DEFAULT_MAX_TOKENS,
        temperature: float = LLM_DEFAULT_TEMPERATURE,
        stop_sequences: Sequence[str] = ("\nUser:", "\nSystem:"),
    ):
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
        self._consecutive_failures = 0
        self._circuit_open = False
        self._circuit_threshold = CIRCUIT_BREAKER_THRESHOLD

    def _build_payload(self, prompt: str, *, stream: bool = False) -> dict:
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

    def _request_with_retry(self, payload) -> dict | None:
        """Single request with exponential backoff. Returns parsed dict or None."""
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
                    self._consecutive_failures = 0
                    return dict(response.json())
                self.logger.error(
                    "LLM Error %d: %s",
                    response.status_code,
                    response.text[:500],
                )
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

        self._consecutive_failures += 1
        if self._consecutive_failures >= self._circuit_threshold:
            self._circuit_open = True
            self.logger.error(
                "Circuit breaker open after %d failures",
                self._circuit_threshold,
            )
        return None

    def reset_circuit(self):
        """Reset the circuit breaker."""
        self._circuit_open = False
        self._consecutive_failures = 0

    def generate_candidates(self, prompt, n=3) -> list[dict]:
        """Generate *n* candidate responses from the LLM backend."""
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"n must be a positive integer; got {n!r}")
        candidates = []
        payload = self._build_payload(prompt)

        for _i in range(n):
            data = self._request_with_retry(payload)
            if data is not None:
                text = data.get(
                    "content",
                    data.get("choices", [{}])[0].get("text", ""),
                )
                candidates.append({"text": text, "source": "LLM"})
            else:
                candidates.append(
                    {"text": "[Error: LLM unavailable]", "source": "System"},
                )

        return candidates

    async def stream_tokens(self, prompt: str) -> AsyncIterator[str]:
        """Yield tokens via SSE streaming if supported, else replay."""
        try:
            import httpx

            payload = self._build_payload(prompt, stream=True)
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", self.api_url, json=payload) as resp:
                    async for line in resp.aiter_lines():  # pragma: no branch
                        if line.startswith("data: "):  # pragma: no branch
                            import json

                            data = json.loads(line[6:])
                            token = data.get("content", data.get("token", ""))
                            if token:  # pragma: no branch
                                yield token
                return
        except (ImportError, Exception) as exc:
            self.logger.debug("SSE streaming unavailable (%s) — replay fallback", exc)

        # Replay fallback
        candidates = self.generate_candidates(prompt, n=1)
        for word in candidates[0]["text"].split():
            yield word
