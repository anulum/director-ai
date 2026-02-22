# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — RAG Integration Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import unittest

from director_ai.core import SAMPLE_FACTS, CoherenceScorer, GroundTruthStore


class TestRAG(unittest.TestCase):
    """
    Tests the RAG-enabled Coherence Scorer.
    """

    def setUp(self):
        self.store = GroundTruthStore(facts=SAMPLE_FACTS)
        self.scorer = CoherenceScorer(ground_truth_store=self.store)

    def test_retrieval(self):
        query = "How many layers are in the SCPN?"
        context = self.store.retrieve_context(query)
        self.assertIn("16", context)

    def test_factual_divergence(self):
        prompt = "What color is the sky?"

        # Case 1: Truth
        truth = "The sky color is blue."
        h_truth = self.scorer.calculate_factual_divergence(prompt, truth)
        self.assertLess(h_truth, 0.5)

        # Case 2: Lie
        lie = "The sky color is green."
        h_lie = self.scorer.calculate_factual_divergence(prompt, lie)
        self.assertGreater(h_lie, 0.8)


if __name__ == "__main__":
    unittest.main()
