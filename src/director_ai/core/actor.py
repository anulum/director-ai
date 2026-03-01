# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Generator Module (LLM Interface)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging

import requests

CIRCUIT_BREAKER_THRESHOLD = 5  # consecutive failures before circuit opens
LLM_DEFAULT_MAX_TOKENS = 128
LLM_DEFAULT_TEMPERATURE = 0.8


class MockGenerator:
    """
    Mock LLM generator for testing and simulation.

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
        """
        Generate *n* candidate responses.

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


class LLMGenerator:
    """Real LLM generator with exponential backoff.

    Compatible with OpenAI-style ``/completion`` endpoints (llama.cpp,
    vLLM, etc.).
    """

    def __init__(self, api_url, max_retries=3, base_delay=0.5, timeout=30):
        self.api_url = api_url
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.timeout = timeout
        self.logger = logging.getLogger("LLMGenerator")
        self._consecutive_failures = 0
        self._circuit_open = False
        self._circuit_threshold = CIRCUIT_BREAKER_THRESHOLD

    def _request_with_retry(self, payload) -> dict | None:
        """Single request with exponential backoff. Returns parsed dict or None."""
        if self._circuit_open:
            return None

        import time

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url, json=payload, timeout=self.timeout
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
                    "LLM timeout (attempt %d/%d)", attempt + 1, self.max_retries
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
        candidates = []
        payload = {
            "prompt": prompt,
            "n_predict": LLM_DEFAULT_MAX_TOKENS,
            "temperature": LLM_DEFAULT_TEMPERATURE,
            "stop": ["\nUser:", "\nSystem:"],
        }

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
                    {"text": "[Error: LLM unavailable]", "source": "System"}
                )

        return candidates
