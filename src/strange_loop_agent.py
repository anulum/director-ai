# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Strange Loop Agent (Main Orchestrator)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

from .director_module import DirectorModule
from .actor_module import MockActor
from .backfire_kernel import BackfireKernel
from .knowledge_base import KnowledgeBase
import logging
import requests
import json

class RealActor:
    """
    Real Actor module connecting to a local LLM runner (e.g. Llama.cpp / vLLM).
    """
    def __init__(self, api_url):
        self.api_url = api_url
        self.logger = logging.getLogger("RealActor")

    def generate_candidates(self, prompt, n=3):
        """
        Generate candidates from the local LLM.
        """
        candidates = []
        payload = {
            "prompt": prompt,
            "n_predict": 128,
            "temperature": 0.8,
            "stop": ["\nUser:", "\nSystem:"]
        }
        
        # We generate 'n' candidates sequentially or in batch if API supports
        for i in range(n):
            try:
                # Assuming OpenAI-compatible or simple completion endpoint
                # Adjust payload for specific server (llama.cpp server uses /completion)
                response = requests.post(self.api_url, json=payload, timeout=30)
                if response.status_code == 200:
                    text = response.json().get('content', response.json().get('choices', [{}])[0].get('text', ''))
                    candidates.append({'text': text, 'source': 'RealLLM'})
                else:
                    self.logger.error(f"LLM Error {response.status_code}: {response.text}")
                    candidates.append({'text': f"[Error: LLM returned {response.status_code}]", 'source': 'System'})
            except Exception as e:
                self.logger.error(f"LLM Connection Failed: {e}")
                candidates.append({'text': "[Error: LLM Connection Failed]", 'source': 'System'})
                
        return candidates

class StrangeLoopAgent:
    """
    The Integrated Director-Class Agent (Version 2).
    Combines:
    - Actor (L11): Narrative Engine (Real LLM or Mock)
    - Director (L16): Recursive Entropy Oversight
    - KnowledgeBase (RAG): Ground Truth
    - BackfireKernel: Hardware Interlock
    """
    
    def __init__(self, llm_api_url=None):
        if llm_api_url:
            self.actor = RealActor(llm_api_url)
            print(f"StrangeLoopAgent: Connected to Real LLM at {llm_api_url}")
        else:
            self.actor = MockActor()
            print("StrangeLoopAgent: Using Mock Actor (Simulation Mode)")
            
        self.kb = KnowledgeBase()
        # Initialize Director with KB
        self.director = DirectorModule(entropy_threshold=0.6, knowledge_base=self.kb) 
        self.kernel = BackfireKernel()
        
        self.logger = logging.getLogger("StrangeLoopAgent")
        logging.basicConfig(level=logging.INFO)

    def process_query(self, prompt):
        """
        Main processing loop with Kernel oversight.
        """
        self.logger.info(f"Received Prompt: '{prompt}'")
        
        # 1. Generate Candidates (Feed-forward)
        candidates = self.actor.generate_candidates(prompt)
        
        best_response = None
        best_sec = -1.0
        
        # 2. Recursive Oversight
        for i, cand in enumerate(candidates):
            text = cand['text']
            
            # Director reviews
            approved, sec_score = self.director.review_action(prompt, text)
            
            self.logger.info(f"Candidate {i} SEC={sec_score:.4f} | Approved={approved}")
            
            if approved and sec_score > best_sec:
                best_sec = sec_score
                best_response = text
        
        # 3. Kernel Output Streaming
        if best_response:
            # We simulate the streaming process where the Kernel monitors the output
            # In a real system, this would happen token-by-token
            
            def sec_monitor_callback(token):
                # In real-time, the Director would re-evaluate as the sentence forms
                # For this prototype, we return the pre-calculated score
                return best_sec
            
            # The Kernel has the final say
            final_output = self.kernel.stream_output([best_response], sec_monitor_callback)
            return f"[AGI Output]: {final_output}"
            
        else:
            return "[SYSTEM HALT]: No coherent response found. Self-termination to prevent entropy leakage."

