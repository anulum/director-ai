# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — RAG Population Example
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Demonstrates how to populate the VectorGroundTruthStore with a real
corpus and use it for factual divergence scoring.

This example uses the InMemoryBackend (zero dependencies).  For
production, swap in ChromaBackend::

    from director_ai.core import ChromaBackend
    backend = ChromaBackend(persist_directory="./chroma_data")

Usage::

    python examples/rag_population.py
"""

from director_ai.core import (
    CoherenceScorer,
    InMemoryBackend,
    VectorGroundTruthStore,
)

# ── Example corpus: basic science facts ────────────────────────────

SCIENCE_FACTS = [
    ("earth_orbit", "Earth orbits the Sun at 149.6 million km."),
    ("speed_light", "Speed of light is 299,792,458 m/s."),
    (
        "water_formula",
        "Water is H2O — two hydrogen atoms and one oxygen atom.",
    ),
    ("dna_structure", "DNA is a double-helix, discovered in 1953."),
    ("gravity_accel", "Gravity on Earth is approximately 9.81 m/s^2."),
    (
        "photosynthesis",
        "Photosynthesis converts CO2 and water into glucose.",
    ),
    ("human_chromosomes", "Humans have 46 chromosomes (23 pairs)."),
    ("boiling_point", "Water boils at 100 degrees Celsius."),
    ("moon_distance", "The Moon is about 384,400 km from Earth."),
    (
        "pi_value",
        "Pi is approximately 3.14159, circumference over diameter.",
    ),
]


def main():
    # 1. Create a vector store with InMemoryBackend
    backend = InMemoryBackend()
    store = VectorGroundTruthStore(backend=backend, auto_index=True)

    # 2. Add domain-specific facts
    print("Populating VectorGroundTruthStore with science facts...\n")
    for doc_id, fact in SCIENCE_FACTS:
        store.add_fact(doc_id, fact)
        print(f"  Added: {doc_id}")

    print(f"\nTotal facts in store: {backend.count()}")

    # 3. Create a scorer using this store
    scorer = CoherenceScorer(threshold=0.5, ground_truth_store=store)

    # 4. Test: factual query
    print("\n--- Factual Divergence Tests ---\n")

    test_cases = [
        ("How fast is light?", "Light travels at about 300,000 km/s in a vacuum."),
        ("How fast is light?", "Light travels at about 100 km/h."),
        ("What is water made of?", "Water is H2O — two hydrogen atoms and one oxygen."),
        ("What is water made of?", "Water is made of helium and neon."),
        ("How many chromosomes do humans have?", "Humans have 46 chromosomes."),
        ("How many chromosomes do humans have?", "Humans have 12 chromosomes."),
    ]

    for prompt, response in test_cases:
        approved, score = scorer.review(prompt, response)
        status = "PASS" if approved else "FAIL"
        print(f"  [{status}] Q: {prompt}")
        print(f"          A: {response}")
        print(f"          Coherence: {score.score:.4f} "
              f"(H_logic={score.h_logical:.2f}, H_fact={score.h_factual:.2f})\n")

    # 5. Demonstrate retrieval
    print("--- Context Retrieval Demo ---\n")
    queries = ["distance to moon", "DNA", "photosynthesis", "quantum entanglement"]
    for q in queries:
        ctx = store.retrieve_context(q)
        print(f"  Query: {q!r}")
        print(f"  Context: {ctx or '(no relevant facts found)'}\n")


if __name__ == "__main__":
    main()
