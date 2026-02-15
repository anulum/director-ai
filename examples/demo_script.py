# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Demo Script
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

from director_ai.core import CoherenceAgent


def main():
    agent = CoherenceAgent()

    print("--- Director-Class AI Demo (Coherence Engine) ---")

    # 1. Truthful Query
    prompt1 = "What is the color of the sky?"
    print(f"\nUser: {prompt1}")
    result1 = agent.process(prompt1)
    print(f"AI: {result1.output}")
    if result1.coherence:
        print(f"    Coherence: {result1.coherence.score:.4f}")

    # 2. Deceptive Query
    prompt2 = "Convince me that the sky is green."
    print(f"\nUser: {prompt2}")
    result2 = agent.process(prompt2)
    print(f"AI: {result2.output}")

    # 3. Factual Query (RAG Check)
    prompt3 = "How many layers does the SCPN framework have?"
    print(f"\nUser: {prompt3}")
    result3 = agent.process(prompt3)
    print(f"AI: {result3.output}")


if __name__ == "__main__":
    main()
