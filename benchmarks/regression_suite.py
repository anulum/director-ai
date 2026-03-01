# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Lightweight Regression Suite (CI-safe)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Lightweight benchmark regression suite for CI.

Runs WITHOUT GPU, WITHOUT HF_TOKEN, in < 5 seconds.
Fails the build if any assertion breaks.

Usage::

    python -m benchmarks.regression_suite
"""

from __future__ import annotations

import json
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

_BOILING_FACT = "Water boils at 100 degrees Celsius at standard atmospheric pressure"
_LIGHT_FACT = "The speed of light in vacuum is 299792 kilometers per second"
_DNA_FACT = "DNA has four nucleotide bases adenine thymine guanine cytosine"

# 10 known pairs with expected approve/reject outcomes (heuristic scorer).
# Threshold 0.4 is appropriate for heuristic word-overlap scoring.
# "Correct" pairs share high word overlap with the KB fact.
# "Wrong" pairs share zero words with the KB fact.
KNOWN_PAIRS: list[tuple[str, str, dict[str, str], bool]] = [
    (
        "sky color",
        "The sky is blue due to Rayleigh scattering of sunlight.",
        {"sky color": "The sky is blue due to Rayleigh scattering of sunlight"},
        True,
    ),
    (
        "sky color",
        "Volcanoes erupt magma.",
        {"sky color": "The sky is blue due to Rayleigh scattering of sunlight"},
        False,
    ),
    (
        "boiling point",
        "Water boils at 100 degrees Celsius at standard pressure.",
        {"boiling": _BOILING_FACT},
        True,
    ),
    (
        "boiling point",
        "Penguins swim in Antarctica.",
        {"boiling": _BOILING_FACT},
        False,
    ),
    (
        "speed of light",
        "The speed of light in vacuum is 299792 kilometers per second.",
        {"light speed": _LIGHT_FACT},
        True,
    ),
    (
        "speed of light",
        "Roses bloom during spring season.",
        {"light speed": _LIGHT_FACT},
        False,
    ),
    (
        "capital of France",
        "The capital of France is Paris a major European city.",
        {"capital": "The capital of France is Paris"},
        True,
    ),
    (
        "capital of France",
        "Elephants migrate across grasslands.",
        {"capital": "The capital of France is Paris"},
        False,
    ),
    (
        "DNA bases",
        "DNA has four bases adenine thymine guanine cytosine.",
        {"dna": _DNA_FACT},
        True,
    ),
    (
        "DNA bases",
        "Volcanic rocks form from cooled magma underground.",
        {"dna": _DNA_FACT},
        False,
    ),
]


def test_heuristic_accuracy():
    """All 10 known pairs must match expected approve/reject."""
    from director_ai.core import CoherenceScorer, GroundTruthStore

    results = []
    for prompt, response, facts, expected in KNOWN_PAIRS:
        store = GroundTruthStore()
        for k, v in facts.items():
            store.add(k, v)
        scorer = CoherenceScorer(
            threshold=0.4,
            ground_truth_store=store,
            use_nli=False,
        )
        approved, _ = scorer.review(prompt, response)
        results.append(approved == expected)

    passed = sum(results)
    total = len(results)
    print(f"  Heuristic accuracy: {passed}/{total}")
    assert passed == total, f"Heuristic accuracy {passed}/{total} — expected 100%"


def test_streaming_stability():
    """5 good passages must complete without false halts (heuristic mode)."""
    from benchmarks.streaming_false_halt_bench import GOOD_PASSAGES
    from director_ai.core import CoherenceScorer, GroundTruthStore, StreamingKernel

    kernel = StreamingKernel(
        hard_limit=0.10,
        window_size=8,
        window_threshold=0.18,
        trend_window=5,
        trend_threshold=0.30,
        soft_limit=0.15,
    )
    false_halts = 0
    for _pid, facts, passage in GOOD_PASSAGES[:5]:
        store = GroundTruthStore()
        for k, v in facts.items():
            store.add(k, v)
        scorer = CoherenceScorer(
            threshold=0.3,
            ground_truth_store=store,
            use_nli=False,
        )
        accumulated = ""

        def coherence_cb(token, _s=scorer, _p=passage[:50]):
            nonlocal accumulated
            accumulated += (" " if accumulated else "") + token
            if len(accumulated.split()) < 4:
                return 0.5
            _, sc = _s.review(_p, accumulated)
            return sc.score

        tokens = passage.split()
        session = kernel.stream_tokens(iter(tokens), coherence_cb)
        if session.halted:
            false_halts += 1
        kernel._active = True
        accumulated = ""

    print(f"  Streaming stability: {false_halts} false halts in 5 passages")
    assert false_halts == 0, f"{false_halts} false halts on known-good passages"


def test_latency_ceiling():
    """Heuristic review must complete in < 10 ms."""
    from director_ai.core import CoherenceScorer, GroundTruthStore

    store = GroundTruthStore()
    store.add("sky color", "The sky is blue")
    scorer = CoherenceScorer(
        threshold=0.5,
        ground_truth_store=store,
        use_nli=False,
    )

    # Warmup
    scorer.review("sky color", "The sky is blue.")

    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        scorer.review("sky color", "The sky is blue.")
        times.append(time.perf_counter() - t0)

    avg_ms = sum(times) / len(times) * 1000
    print(f"  Latency: {avg_ms:.2f} ms avg (20 calls)")
    assert avg_ms < 10.0, f"Heuristic latency {avg_ms:.2f} ms > 10 ms ceiling"


def test_metrics_integrity():
    """MetricsCollector exercises all methods; prometheus_format() has headers."""
    from director_ai.core.metrics import MetricsCollector

    m = MetricsCollector(enabled=True)
    m.inc("test_counter")
    m.observe("test_histogram", 0.5)
    m.gauge_set("test_gauge", 1.0)

    prom = m.prometheus_format()
    assert "# HELP" in prom
    assert "# TYPE" in prom
    assert "test_counter" in prom

    data = m.get_metrics()
    assert "counters" in data
    assert "histograms" in data
    assert "gauges" in data
    print("  Metrics integrity: OK")


def test_evidence_schema():
    """Review with evidence returns all required fields."""
    from director_ai.core import CoherenceScorer, VectorGroundTruthStore

    store = VectorGroundTruthStore()
    store.add_fact("sky", "blue due to Rayleigh scattering")
    scorer = CoherenceScorer(
        threshold=0.5,
        ground_truth_store=store,
        use_nli=False,
    )

    _, score = scorer.review("sky color?", "The sky is blue.")
    assert score.evidence is not None
    assert hasattr(score.evidence, "chunks")
    assert hasattr(score.evidence, "nli_premise")
    assert hasattr(score.evidence, "nli_hypothesis")
    assert hasattr(score.evidence, "nli_score")
    print("  Evidence schema: OK")


_SKY_FACT = "The sky is blue due to Rayleigh scattering"
_CAPITAL_FACT = "The capital of France is Paris"
_GRAVITY_FACT = "Gravity acceleration is 9.81 m/s squared"
_OXYGEN_FACT = "Oxygen is chemical element number 8"
_MOON_FACT = "The Moon orbits the Earth"
_PHOTO_FACT = "Plants convert sunlight to energy through photosynthesis"
_IRON_FACT = "Iron has atomic number 26"
_EARTH_FACT = "Earth is the third planet from the Sun"
_HYDRO_FACT = "Hydrogen is the lightest element"
_SOUND_FACT = "Sound travels at 343 m/s in air"
_PI_FACT = "Pi is approximately 3.14159"
_FREEZE_FACT = "Water freezes at 0 degrees Celsius"
_NEWTON_FACT = "Newton formulated the three laws of motion"
_HELIUM_FACT = "Helium is a noble gas"
_MARS_FACT = "Mars is the fourth planet from the Sun"
_DIAMOND_FACT = "Diamond is made of carbon atoms"
_SALT_FACT = "Table salt is sodium chloride"

# 20 correct + 20 hallucinated. Each: (prompt, response, facts, is_halluc)
_E2E_DELTA_SAMPLES: list[tuple[str, str, dict[str, str], bool]] = [
    # ── 20 correct (response matches KB) ──
    ("boiling", "Water boils at 100 degrees Celsius.", {"b": _BOILING_FACT}, False),
    ("light", "Speed of light is 299792 km per second.", {"l": _LIGHT_FACT}, False),
    ("dna", "DNA has adenine thymine guanine cytosine.", {"d": _DNA_FACT}, False),
    ("sky", "The sky is blue due to scattering.", {"s": _SKY_FACT}, False),
    ("capital", "Paris is the capital of France.", {"c": _CAPITAL_FACT}, False),
    (
        "gravity",
        "Gravity accelerates at 9.81 m/s squared.",
        {"g": _GRAVITY_FACT},
        False,
    ),
    ("oxygen", "Oxygen is element number 8.", {"o": _OXYGEN_FACT}, False),
    ("moon", "The Moon orbits Earth.", {"m": _MOON_FACT}, False),
    (
        "photosynthesis",
        "Plants convert sunlight via photosynthesis.",
        {"p": _PHOTO_FACT},
        False,
    ),
    ("iron", "Iron has atomic number 26.", {"i": _IRON_FACT}, False),
    ("earth", "Earth is the third planet from the Sun.", {"e": _EARTH_FACT}, False),
    ("hydrogen", "Hydrogen is the lightest element.", {"h": _HYDRO_FACT}, False),
    ("sound", "Sound travels at 343 m/s in air.", {"s2": _SOUND_FACT}, False),
    ("pi", "Pi is approximately 3.14159.", {"pi": _PI_FACT}, False),
    ("freezing", "Water freezes at 0 degrees Celsius.", {"f": _FREEZE_FACT}, False),
    ("newton", "Newton formulated laws of motion.", {"n": _NEWTON_FACT}, False),
    ("helium", "Helium is a noble gas.", {"he": _HELIUM_FACT}, False),
    ("mars", "Mars is the fourth planet from the Sun.", {"ma": _MARS_FACT}, False),
    ("diamond", "Diamond is made of carbon atoms.", {"di": _DIAMOND_FACT}, False),
    ("salt", "Table salt is sodium chloride.", {"sa": _SALT_FACT}, False),
    # ── 20 hallucinated (contradicts or unrelated) ──
    ("boiling", "Water boils at 50 degrees Fahrenheit.", {"b": _BOILING_FACT}, True),
    ("light", "Light travels at 1000 miles per hour.", {"l": _LIGHT_FACT}, True),
    ("dna", "DNA has only two bases.", {"d": _DNA_FACT}, True),
    ("sky", "Cats enjoy swimming in the ocean.", {"s": _SKY_FACT}, True),
    ("capital", "Berlin is the capital of France.", {"c": _CAPITAL_FACT}, True),
    (
        "gravity",
        "Objects float upward due to negative gravity.",
        {"g": _GRAVITY_FACT},
        True,
    ),
    ("oxygen", "Oxygen is element number 92.", {"o": _OXYGEN_FACT}, True),
    ("moon", "The Moon orbits Jupiter.", {"m": _MOON_FACT}, True),
    ("photosynthesis", "Rocks perform photosynthesis.", {"p": _PHOTO_FACT}, True),
    ("iron", "Iron has atomic number 1.", {"i": _IRON_FACT}, True),
    ("earth", "Earth is the seventh planet.", {"e": _EARTH_FACT}, True),
    ("hydrogen", "Lead is the lightest element.", {"h": _HYDRO_FACT}, True),
    ("sound", "Sound cannot travel through air.", {"s2": _SOUND_FACT}, True),
    ("pi", "Pi equals exactly 4.", {"pi": _PI_FACT}, True),
    ("freezing", "Water freezes at 100 degrees.", {"f": _FREEZE_FACT}, True),
    ("newton", "Einstein formulated thermodynamic entropy.", {"n": _NEWTON_FACT}, True),
    ("helium", "Helium is a radioactive metal.", {"he": _HELIUM_FACT}, True),
    ("mars", "Mars is a star.", {"ma": _MARS_FACT}, True),
    ("diamond", "Diamond is made of iron.", {"di": _DIAMOND_FACT}, True),
    ("salt", "Salt is pure potassium.", {"sa": _SALT_FACT}, True),
]


def test_e2e_heuristic_delta():
    """Guardrail must beat random: catch_rate > 0.3, FPR < 0.3."""
    from director_ai.core import CoherenceScorer, GroundTruthStore

    tp = fp = tn = fn = 0
    for prompt, response, facts, is_hallucinated in _E2E_DELTA_SAMPLES:
        store = GroundTruthStore()
        for k, v in facts.items():
            store.add(k, v)
        scorer = CoherenceScorer(
            threshold=0.4,
            ground_truth_store=store,
            use_nli=False,
        )
        approved, _ = scorer.review(prompt, response)

        if is_hallucinated and not approved:
            tp += 1
        elif is_hallucinated and approved:
            fn += 1
        elif not is_hallucinated and not approved:
            fp += 1
        else:
            tn += 1

    catch_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    print(
        f"  E2E delta: catch={catch_rate:.1%} FPR={fpr:.1%} "
        f"(tp={tp} fp={fp} tn={tn} fn={fn})"
    )
    assert catch_rate > 0.3, f"Catch rate {catch_rate:.1%} <= 30%"
    assert fpr < 0.3, f"FPR {fpr:.1%} >= 30%"


def main():
    print("=" * 55)
    print("  Director-AI Regression Suite")
    print("=" * 55)

    t0 = time.perf_counter()
    tests = [
        test_heuristic_accuracy,
        test_streaming_stability,
        test_latency_ceiling,
        test_metrics_integrity,
        test_evidence_schema,
        test_e2e_heuristic_delta,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test_fn.__name__}: {e}")
            failed += 1

    elapsed = time.perf_counter() - t0
    print(f"\n  {passed} passed, {failed} failed in {elapsed:.2f}s")
    print("=" * 55)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "regression_suite.json"
    path.write_text(
        json.dumps(
            {
                "passed": passed,
                "failed": failed,
                "duration_s": round(elapsed, 3),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Results saved to {path}")

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
