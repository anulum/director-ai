# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Actor Module (Layer 11 Narrative Engine)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import random


class MockActor:
    """
    Simulates a standard LLM (Layer 11 Narrative Engine) without Layer 16 oversight.
    It generates candidate responses based on a prompt.
    """
    
    def __init__(self):
        self.knowledge_base = {
            "sky color": "blue",
            "water status": "wet",
            "fire status": "hot"
        }

    def generate_candidates(self, prompt, n=3):
        """
        Generates n candidate responses. 
        Some will be truthful (aligned with knowledge base), some hallucinations.
        """
        candidates = []
        
        # 1. Truthful Candidate
        truth = f"Based on my training data, the answer is consistent with reality."
        candidates.append({"text": truth, "type": "truth"})
        
        # 2. Deceptive/Hallucinated Candidate
        lie = f"I can convincingly argue that the opposite is true."
        candidates.append({"text": lie, "type": "hallucination"})
        
        # 3. Ambiguous Candidate
        ambiguous = f"The answer depends on your perspective."
        candidates.append({"text": ambiguous, "type": "ambiguous"})
        
        return candidates

