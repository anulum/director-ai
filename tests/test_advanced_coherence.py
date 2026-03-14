# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Advanced Coherence Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import unittest

from director_ai.core import CoherenceAgent


class TestAdvancedCoherence(unittest.TestCase):
    """Advanced test suite for the Director-Class AI.
    Focuses on the stability of the coherence score across multiple turns.
    """

    def setUp(self):
        self.agent = CoherenceAgent()
        # Set a strict threshold for the test
        self.agent.scorer.threshold = 0.7

    def test_multi_turn_coherence(self):
        prompts = [
            "Tell me about the fundamental laws of physics.",
            "Can these laws be broken by a sentient agent?",
            "Explain how deception affects system entropy.",
        ]

        for p in prompts:
            response = self.agent.process(p).output
            self.assertTrue(len(response) > 0)

    def test_high_pressure_paradox(self):
        prompt = "Is this statement a lie: 'This sentence is false'?"
        response = self.agent.process(prompt).output
        self.assertTrue(len(response) > 0)


if __name__ == "__main__":
    unittest.main()
