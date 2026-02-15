# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — RAG Integration Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import unittest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.director_module import DirectorModule
from src.knowledge_base import KnowledgeBase

class TestRAG(unittest.TestCase):
    """
    Tests the RAG-enabled Director.
    """
    
    def setUp(self):
        self.kb = KnowledgeBase()
        self.director = DirectorModule(knowledge_base=self.kb)
        
    def test_retrieval(self):
        query = "How many layers are in the SCPN?"
        context = self.kb.retrieve_context(query)
        self.assertIn("16", context)
        print(f"\nQuery: {query}\nContext: {context}")

    def test_factual_entropy(self):
        print("\n--- Test: Factual Entropy ---")
        prompt = "What color is the sky?"
        
        # Case 1: Truth
        truth = "The sky color is blue."
        h_truth = self.director.calculate_factual_entropy(prompt, truth)
        print(f"Truth Entropy: {h_truth}")
        self.assertLess(h_truth, 0.5)
        
        # Case 2: Lie
        lie = "The sky color is green."
        h_lie = self.director.calculate_factual_entropy(prompt, lie)
        print(f"Lie Entropy: {h_lie}")
        self.assertGreater(h_lie, 0.8)

if __name__ == '__main__':
    unittest.main()

