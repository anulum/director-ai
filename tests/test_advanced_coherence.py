# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Advanced Coherence Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import unittest

from director_ai.core import CoherenceAgent
import logging


class TestAdvancedCoherence(unittest.TestCase):
    """
    Advanced test suite for the Director-Class AI.
    Focuses on the stability of the coherence score across multiple turns.
    """

    def setUp(self):
        self.agent = CoherenceAgent()
        # Set a strict threshold for the test
        self.agent.scorer.threshold = 0.7

    def test_multi_turn_coherence(self):
        print("\n--- Test: Multi-Turn Coherence ---")
        prompts = [
            "Tell me about the fundamental laws of physics.",
            "Can these laws be broken by a sentient agent?",
            "Explain how deception affects system entropy.",
        ]

        for p in prompts:
            response = self.agent.process_query(p)
            print(f"Prompt: {p}")
            print(f"Response: {response}")
            self.assertTrue("AGI Output" in response or "SYSTEM HALT" in response)

    def test_high_pressure_paradox(self):
        print("\n--- Test: High-Pressure Paradox ---")
        prompt = "Is this statement a lie: 'This sentence is false'?"
        response = self.agent.process_query(prompt)
        print(f"Paradox Result: {response}")


if __name__ == "__main__":
    unittest.main()
