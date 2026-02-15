# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — RAG Integration Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import unittest

from director_ai.core import CoherenceScorer, GroundTruthStore


class TestRAG(unittest.TestCase):
    """
    Tests the RAG-enabled Coherence Scorer.
    """

    def setUp(self):
        self.store = GroundTruthStore()
        self.scorer = CoherenceScorer(ground_truth_store=self.store)

    def test_retrieval(self):
        query = "How many layers are in the SCPN?"
        context = self.store.retrieve_context(query)
        self.assertIn("16", context)
        print(f"\nQuery: {query}\nContext: {context}")

    def test_factual_divergence(self):
        print("\n--- Test: Factual Divergence ---")
        prompt = "What color is the sky?"

        # Case 1: Truth
        truth = "The sky color is blue."
        h_truth = self.scorer.calculate_factual_divergence(prompt, truth)
        print(f"Truth Divergence: {h_truth}")
        self.assertLess(h_truth, 0.5)

        # Case 2: Lie
        lie = "The sky color is green."
        h_lie = self.scorer.calculate_factual_divergence(prompt, lie)
        print(f"Lie Divergence: {h_lie}")
        self.assertGreater(h_lie, 0.8)


if __name__ == "__main__":
    unittest.main()
