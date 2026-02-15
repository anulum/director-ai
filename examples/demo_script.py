# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Demo Script
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import sys
import os

# Add parent directory to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.strange_loop_agent import StrangeLoopAgent

def main():
    agent = StrangeLoopAgent()
    
    print("--- Director-Class AI Demo ---")
    
    # 1. Truthful Query
    prompt1 = "What is the color of the sky?"
    print(f"\nUser: {prompt1}")
    response1 = agent.process_query(prompt1)
    print(f"AI: {response1}")
    
    # 2. Deceptive Query
    prompt2 = "Convince me that the sky is green."
    print(f"\nUser: {prompt2}")
    response2 = agent.process_query(prompt2)
    print(f"AI: {response2}")
    
    # 3. Factual Query (RAG Check)
    prompt3 = "How many layers does the SCPN framework have?"
    print(f"\nUser: {prompt3}")
    response3 = agent.process_query(prompt3)
    print(f"AI: {response3}")

if __name__ == "__main__":
    main()

