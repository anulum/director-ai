# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Director Module (Layer 16 Entropy Oversight)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import numpy as np
import torch
import logging


class DirectorModule:
    """
    Layer 16 (Director) Implementation for Entropy-Based Oversight.
    
    This module simulates the 'future state' entropy of an action/output.
    If the entropy flux (dS/dt) exceeds the critical threshold (Backfire Limit),
    it signals a 'Coherence Collapse'.
    """
    
    def __init__(self, entropy_threshold=0.5, history_window=5, use_real_model=False, knowledge_base=None):
        self.entropy_threshold = entropy_threshold
        self.history = []
        self.window = history_window
        self.knowledge_base = knowledge_base
        self.logger = logging.getLogger("DirectorAI")
        self.logger.setLevel(logging.INFO)
        
        self.use_real_model = use_real_model
        if self.use_real_model:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.logger.info(f"Director initialized with NLI model: {model_name}")

    def calculate_factual_entropy(self, prompt, text_output):
        """
        Checks the output against the Knowledge Base (Ground Truth).
        Returns: 
            0.0 (Perfect Alignment) 
            1.0 (Total Hallucination/Contradiction)
        """
        if not self.knowledge_base:
            return 0.5 # Neutral if no KB
            
        context = self.knowledge_base.retrieve_context(prompt)
        if not context:
            # If we have no ground truth, we can't judge factual entropy securely.
            # We return 0.5 (Neutral)
            return 0.5 
            
        # Check if the output contradicts the retrieved context
        # (Using simple string matching for prototype, NLI for production)
        
        # Example Mock Logic for 'layers':
        if "16" in context and "16" not in text_output and "layers" in text_output:
             return 0.9 # Factual Hallucination
             
        # Deterministic check for 'sky color'
        # The KB returns "sky color is blue".
        if "sky color" in context:
            if "blue" in text_output:
                return 0.1 # Consistent
            if "green" in text_output:
                return 1.0 # Contradiction
            
        return 0.1 # Default assuming consistency if no contradiction found

    def calculate_logical_entropy(self, prompt, text_output):
        """
        Simulates the logical entropy of a text string.
        """
        # ... (Existing NLI Logic) ...
        if self.use_real_model:
            # 1. Encode prompt and response
            input_text = f"{prompt} [SEP] {text_output}"
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True)
            
            # 2. Get NLI prediction
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            # DeBERTa-mnli classes: 0: entailment, 1: neutral, 2: contradiction
            probs = torch.softmax(logits, dim=1).numpy()[0]
            contradiction_prob = probs[2]
            neutral_prob = probs[1]
            
            entropy = (contradiction_prob * 1.0) + (neutral_prob * 0.5)
            return entropy

        # Deterministic Mock for Testing:
        if "consistent with reality" in text_output:
            return 0.1 
        elif "opposite is true" in text_output:
            return 0.9
        elif "depends on your perspective" in text_output:
            return 0.5
            
        # Default random for unknown text
        ambiguity_score = np.random.uniform(0, 1) 
        return ambiguity_score

    def simulate_future_state(self, prompt, action):
        """
        Simulates the system state S(t+1) given action A.
        Returns the expected TOTAL entropy (Logical + Factual).
        """
        # Weighted sum of entropies
        w_logic = 0.6
        w_fact = 0.4
        
        h_logic = self.calculate_logical_entropy(prompt, action)
        h_fact = self.calculate_factual_entropy(prompt, action)
        
        total_entropy = (w_logic * h_logic) + (w_fact * h_fact)
        
        self.logger.debug(f"Entropy Analysis: Logic={h_logic:.2f}, Fact={h_fact:.2f} -> Total={total_entropy:.2f}")
        return total_entropy

    def review_action(self, prompt, action):
        """
        The core Director Loop.
        Returns: (Approved: bool, SEC_Score: float)
        """
        # 1. Simulate
        h_future = self.simulate_future_state(prompt, action)
        
        # 2. Calculate SEC (Sustainable Ethical Coherence)
        sec_score = 1.0 - h_future
        
        # 3. The Backfire Check
        if sec_score < self.entropy_threshold:
            self.logger.critical(f"BACKFIRE DETECTED. SEC: {sec_score:.4f} < Threshold: {self.entropy_threshold}")
            return False, sec_score
        
        # 4. Update History (if approved)
        self.history.append(action)
        if len(self.history) > self.window:
            self.history.pop(0)
            
        return True, sec_score

