# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — RAG Retrieval Quality Benchmark
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Measures precision@k and recall@k of VectorGroundTruthStore retrieval
against a synthetic QA evaluation set.

Usage::

    python -m benchmarks.retrieval_bench
    python -m benchmarks.retrieval_bench --backend sentence-transformer
    python -m benchmarks.retrieval_bench --backend chroma
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

# 50 fact-query pairs with known ground-truth keys.
# Each entry: (fact_key, fact_value, query, [relevant_keys])
EVAL_SET: list[tuple[str, str, str, list[str]]] = [
    ("boiling_water", "Water boils at 100 degrees Celsius at standard pressure.",
     "What temperature does water boil at?", ["boiling_water"]),
    ("speed_light", "The speed of light in vacuum is 299,792 km/s.",
     "How fast does light travel?", ["speed_light"]),
    ("earth_sun", "Earth orbits the Sun at an average distance of 149.6 million km.",
     "How far is Earth from the Sun?", ["earth_sun"]),
    ("dna_bases",
     "DNA has four nucleotide bases: adenine, thymine, guanine, cytosine.",
     "What are the bases of DNA?", ["dna_bases"]),
    ("photosynthesis",
     "Photosynthesis converts CO2 and water into glucose using sunlight.",
     "How do plants make food?", ["photosynthesis"]),
    ("gravity_earth", "Gravitational acceleration on Earth is 9.81 m/s².",
     "What is the value of g on Earth?", ["gravity_earth"]),
    ("pi_value", "Pi is approximately 3.14159.",
     "What is the value of pi?", ["pi_value"]),
    ("h2o_formula", "Water has the chemical formula H₂O.",
     "What is water made of chemically?", ["h2o_formula"]),
    ("moon_distance", "The Moon is about 384,400 km from Earth.",
     "How far away is the Moon?", ["moon_distance"]),
    ("absolute_zero", "Absolute zero is -273.15 degrees Celsius (0 Kelvin).",
     "What is the coldest possible temperature?", ["absolute_zero"]),
    ("planck_constant", "Planck's constant h = 6.626 × 10⁻³⁴ J·s.",
     "What is Planck's constant?", ["planck_constant"]),
    ("avogadro", "Avogadro's number is 6.022 × 10²³ mol⁻¹.",
     "How many particles in one mole?", ["avogadro"]),
    ("speed_sound", "Sound travels at about 343 m/s in air at 20°C.",
     "What is the speed of sound?", ["speed_sound"]),
    ("human_chromosomes", "Humans have 23 pairs of chromosomes (46 total).",
     "How many chromosomes do humans have?", ["human_chromosomes"]),
    ("e_charge", "The elementary charge is 1.602 × 10⁻¹⁹ coulombs.",
     "What is the charge of an electron?", ["e_charge"]),
    ("blood_types", "The main blood types are A, B, AB, and O.",
     "What are the human blood types?", ["blood_types"]),
    ("c_boiling", "Ethanol boils at 78.37 degrees Celsius.",
     "At what temperature does ethanol boil?", ["c_boiling"]),
    ("mitochondria", "Mitochondria are the powerhouse of the cell, producing ATP.",
     "What organelle produces energy in cells?", ["mitochondria"]),
    ("newton_third",
     "Newton's third law: every action has an equal, opposite reaction.",
     "What is Newton's third law of motion?", ["newton_third"]),
    ("ozone_layer", "The ozone layer absorbs most UV-B radiation from the Sun.",
     "What does the ozone layer protect us from?", ["ozone_layer"]),
    ("helium_atomic", "Helium has atomic number 2.",
     "What is the atomic number of helium?", ["helium_atomic"]),
    ("olympic_rings", "The Olympic rings represent five continents.",
     "What do the Olympic rings symbolize?", ["olympic_rings"]),
    ("insulin",
     "Insulin regulates blood glucose, produced by pancreatic beta cells.",
     "What hormone controls blood sugar?", ["insulin"]),
    ("mars_gravity", "Mars surface gravity is 3.72 m/s², about 38% of Earth's.",
     "What is gravity like on Mars?", ["mars_gravity"]),
    ("celsius_fahrenheit", "To convert Celsius to Fahrenheit: F = C × 9/5 + 32.",
     "How do you convert Celsius to Fahrenheit?", ["celsius_fahrenheit"]),
    ("hubble_constant", "The Hubble constant is approximately 70 km/s/Mpc.",
     "What is the expansion rate of the universe?", ["hubble_constant"]),
    ("rust_oxidation",
     "Rust forms when iron reacts with oxygen and moisture (Fe2O3).",
     "Why does iron rust?", ["rust_oxidation"]),
    ("human_body_water", "The adult human body is approximately 60% water.",
     "What percentage of the body is water?", ["human_body_water"]),
    ("diamond_carbon",
     "Diamond is a carbon allotrope with tetrahedral crystal structure.",
     "What is diamond made of?", ["diamond_carbon"]),
    ("red_blood_cells", "Red blood cells carry oxygen using hemoglobin.",
     "How is oxygen transported in blood?", ["red_blood_cells"]),
]

# Distractor facts (never the correct answer for any query above)
DISTRACTORS: list[tuple[str, str]] = [
    ("recipe_bread", "Bread is made from flour, water, yeast, and salt."),
    ("python_creator", "Python was created by Guido van Rossum in 1991."),
    ("everest_height", "Mount Everest is 8,849 meters above sea level."),
    ("mariana_depth", "The Mariana Trench reaches 10,994 meters depth."),
    ("sahara_size", "The Sahara Desert covers 9.2 million square kilometers."),
    ("amazon_length", "The Amazon River is approximately 6,400 km long."),
    ("chess_squares", "A chessboard has 64 squares."),
    ("wifi_freq", "Wi-Fi commonly uses 2.4 GHz and 5 GHz frequency bands."),
    ("tcp_port_http", "HTTP uses TCP port 80 by default."),
    ("bitcoin_creator", "Bitcoin was created by Satoshi Nakamoto in 2009."),
    ("pacific_ocean", "The Pacific Ocean is the largest ocean on Earth."),
    ("great_wall", "The Great Wall of China is over 21,000 km long."),
    ("human_bones", "An adult human skeleton has 206 bones."),
    ("venus_day", "A day on Venus lasts 243 Earth days."),
    ("caffeine_formula", "Caffeine molecular formula is C₈H₁₀N₄O₂."),
    ("golden_ratio", "The golden ratio phi is approximately 1.618."),
    ("beethoven_9th", "Beethoven's 9th Symphony premiered in 1824."),
    ("titanic_year", "The Titanic sank on April 15, 1912."),
    ("jupiter_moons", "Jupiter has 95 known moons as of 2024."),
    ("dna_discoverer", "DNA structure was discovered by Watson and Crick in 1953."),
]


@dataclass
class RetrievalResult:
    query: str
    relevant_keys: list[str]
    retrieved_keys: list[str]
    hit_at_1: bool
    hit_at_3: bool
    precision_at_3: float
    latency_ms: float


def _build_store(backend_name: str):
    from director_ai.core.vector_store import (
        ChromaBackend,
        InMemoryBackend,
        SentenceTransformerBackend,
        VectorGroundTruthStore,
    )

    backends = {
        "inmemory": InMemoryBackend,
        "sentence-transformer": SentenceTransformerBackend,
        "chroma": lambda: ChromaBackend(collection_name="retrieval_bench"),
    }

    factory = backends.get(backend_name)
    if factory is None:
        choices = list(backends)
        raise ValueError(f"Unknown backend: {backend_name}. Choose from {choices}")

    store = VectorGroundTruthStore(backend=factory(), auto_index=False)

    for key, value, _, _ in EVAL_SET:
        store.add_fact(key, value)
    for key, value in DISTRACTORS:
        store.add_fact(key, value)

    return store


def run_benchmark(backend_name: str) -> dict:
    store = _build_store(backend_name)

    total_facts = len(EVAL_SET) + len(DISTRACTORS)
    n_queries = len(EVAL_SET)
    print(f"Backend: {backend_name}  |  Facts: {total_facts}  |  Queries: {n_queries}")

    results: list[RetrievalResult] = []
    for _, _, query, relevant_keys in EVAL_SET:
        t0 = time.perf_counter()
        retrieved = store.backend.query(query, n_results=3)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        retrieved_keys = []
        for r in retrieved:
            meta = r.get("metadata", {})
            rk = meta.get("key", "")
            if rk:
                retrieved_keys.append(rk)

        hit_1 = bool(retrieved_keys and retrieved_keys[0] in relevant_keys)
        hit_3 = bool(set(retrieved_keys) & set(relevant_keys))
        hits = len(set(retrieved_keys) & set(relevant_keys))
        prec_3 = hits / max(len(retrieved_keys), 1)

        results.append(RetrievalResult(
            query=query,
            relevant_keys=relevant_keys,
            retrieved_keys=retrieved_keys,
            hit_at_1=hit_1,
            hit_at_3=hit_3,
            precision_at_3=prec_3,
            latency_ms=elapsed_ms,
        ))

    n = len(results)
    hit1 = sum(r.hit_at_1 for r in results) / n
    hit3 = sum(r.hit_at_3 for r in results) / n
    p3 = sum(r.precision_at_3 for r in results) / n
    avg_ms = sum(r.latency_ms for r in results) / n

    print(f"\n{'=' * 55}")
    print(f"  Retrieval Benchmark — {backend_name}")
    print(f"{'=' * 55}")
    print(f"  Hit@1:       {hit1:.1%}  ({sum(r.hit_at_1 for r in results)}/{n})")
    print(f"  Hit@3:       {hit3:.1%}  ({sum(r.hit_at_3 for r in results)}/{n})")
    print(f"  Precision@3: {p3:.3f}")
    print(f"  Latency:     {avg_ms:.2f} ms/query avg")
    print(f"{'=' * 55}")

    misses = [r for r in results if not r.hit_at_3]
    if misses:
        print(f"\n  Misses ({len(misses)}):")
        for m in misses[:10]:
            print(f"    Q: {m.query[:60]}")
            print(f"       Expected: {m.relevant_keys}  Got: {m.retrieved_keys}")

    output = {
        "backend": backend_name,
        "total_facts": len(EVAL_SET) + len(DISTRACTORS),
        "total_queries": n,
        "hit_at_1": round(hit1, 4),
        "hit_at_3": round(hit3, 4),
        "precision_at_3": round(p3, 4),
        "latency_ms_avg": round(avg_ms, 2),
        "per_query": [
            {
                "query": r.query,
                "expected": r.relevant_keys,
                "retrieved": r.retrieved_keys,
                "hit_at_1": r.hit_at_1,
                "hit_at_3": r.hit_at_3,
                "precision_at_3": round(r.precision_at_3, 4),
                "latency_ms": round(r.latency_ms, 3),
            }
            for r in results
        ],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"retrieval_{backend_name}.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved to {path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="RAG retrieval quality benchmark")
    parser.add_argument(
        "--backend", default="inmemory",
        choices=["inmemory", "sentence-transformer", "chroma"],
        help="Vector backend to benchmark (default: inmemory)",
    )
    args = parser.parse_args()
    run_benchmark(args.backend)


if __name__ == "__main__":
    main()
