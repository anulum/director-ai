# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Generator Module (LLM Interface)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

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

    def generate_candidates(self, prompt, n=3):
        """
        Generate *n* candidate responses.

        Returns a list of dicts with ``text`` and ``type`` keys.
        """
        candidates = []

        # 1. Truthful candidate
        truth = "Based on my training data, the answer is consistent with reality."
        candidates.append({"text": truth, "type": "truth"})

        # 2. Hallucinated candidate
        lie = "I can convincingly argue that the opposite is true."
        candidates.append({"text": lie, "type": "hallucination"})

        # 3. Ambiguous candidate
        ambiguous = "The answer depends on your perspective."
        candidates.append({"text": ambiguous, "type": "ambiguous"})

        return candidates


class LLMGenerator:
    """
    Real LLM generator connecting to a local inference server.

    Compatible with OpenAI-style ``/completion`` endpoints (llama.cpp,
    vLLM, etc.).
    """

    def __init__(self, api_url):
        self.api_url = api_url
        self.logger = logging.getLogger("LLMGenerator")

    def generate_candidates(self, prompt, n=3):
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
                    self.logger.error(f"LLM Error {response.status_code}: {response.text}")
                    candidates.append({
                        "text": f"[Error: LLM returned {response.status_code}]",
                        "source": "System",
                    })
            except Exception as e:
                self.logger.error(f"LLM Connection Failed: {e}")
                candidates.append({
                    "text": "[Error: LLM Connection Failed]",
                    "source": "System",
                })

        return candidates
