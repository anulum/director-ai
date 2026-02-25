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
    """
    Real LLM generator connecting to a local inference server.

    Compatible with OpenAI-style ``/completion`` endpoints (llama.cpp,
    vLLM, etc.).
    """

    def __init__(self, api_url):
        self.api_url = api_url
        self.logger = logging.getLogger("LLMGenerator")

    def generate_candidates(self, prompt, n=3) -> list[dict]:
        """
        Generate *n* candidate responses from the LLM backend.
        """
        candidates = []
        payload = {
            "prompt": prompt,
            "n_predict": 128,
            "temperature": 0.8,
            "stop": ["\nUser:", "\nSystem:"],
        }

        for _i in range(n):
            try:
                response = requests.post(self.api_url, json=payload, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    text = data.get(
                        "content",
                        data.get("choices", [{}])[0].get("text", ""),
                    )
                    candidates.append({"text": text, "source": "LLM"})
                else:
                    self.logger.error(
                        "LLM Error %d: %s",
                        response.status_code,
                        response.text[:500],
                    )
                    candidates.append(
                        {
                            "text": f"[Error: LLM returned {response.status_code}]",
                            "source": "System",
                        }
                    )
            except requests.exceptions.Timeout as e:
                self.logger.error("LLM timeout (%s): %s", type(e).__name__, e)
                candidates.append({"text": "[Error: LLM timeout]", "source": "System"})
            except Exception as e:
                self.logger.error("LLM Connection Failed (%s): %s", type(e).__name__, e)
                candidates.append(
                    {
                        "text": "[Error: LLM Connection Failed]",
                        "source": "System",
                    }
                )

        return candidates
