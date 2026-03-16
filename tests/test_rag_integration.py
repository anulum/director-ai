# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — RAG Integration Tests

import unittest

from director_ai.core import CoherenceScorer, GroundTruthStore


class TestRAG(unittest.TestCase):
    """Tests the RAG-enabled Coherence Scorer."""

    def setUp(self):
        self.store = GroundTruthStore.with_demo_facts()
        self.scorer = CoherenceScorer(ground_truth_store=self.store, use_nli=False)

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
