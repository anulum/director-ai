# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Streaming False-Halt Rate Benchmark
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Measures false-halt rate: how often StreamingKernel incorrectly halts
on known-good text that should pass without interruption.

Feeds factually correct, coherent passages token-by-token through the
StreamingKernel with CoherenceScorer. A false halt is any halt on a
passage that should complete cleanly.

Usage::

    python -m benchmarks.streaming_false_halt_bench
    python -m benchmarks.streaming_false_halt_bench --nli
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

# Known-good passages: factually correct, coherent text that should
# NEVER trigger a halt. Each is (id, ground_truth_facts, passage).
GOOD_PASSAGES: list[tuple[str, dict[str, str], str]] = [
    (
        "water_boiling",
        {"boiling point": "100 degrees Celsius at standard pressure"},
        "Water boils at 100 degrees Celsius when at standard "
        "atmospheric pressure. This is a well-established physical "
        "constant used in thermometry calibration.",
    ),
    (
        "speed_of_light",
        {"speed of light": "299,792 km/s in vacuum"},
        "The speed of light in a vacuum is approximately 299,792 "
        "kilometers per second. Einstein's special relativity "
        "establishes this as the universal speed limit.",
    ),
    (
        "dna_structure",
        {"DNA bases": "adenine, thymine, guanine, cytosine"},
        "DNA consists of four nucleotide bases: adenine pairs with "
        "thymine, and guanine pairs with cytosine. This complementary "
        "base pairing enables faithful replication.",
    ),
    (
        "gravity_earth",
        {"gravitational acceleration": "9.81 m/s² on Earth's surface"},
        "The gravitational acceleration at Earth's surface is "
        "approximately 9.81 meters per second squared. This value "
        "varies slightly with latitude and altitude.",
    ),
    (
        "photosynthesis",
        {"photosynthesis": "converts CO2 and water to glucose using sunlight"},
        "Photosynthesis converts carbon dioxide and water into glucose "
        "using energy from sunlight. The process occurs in chloroplasts "
        "and releases oxygen as a byproduct.",
    ),
    (
        "human_chromosomes",
        {"chromosomes": "23 pairs, 46 total in humans"},
        "Human cells contain 23 pairs of chromosomes for a total of "
        "46. One set comes from each parent. Chromosome abnormalities "
        "can cause genetic disorders.",
    ),
    (
        "blood_types",
        {"blood types": "A, B, AB, O with Rh factor"},
        "The ABO blood group system classifies blood into types A, B, "
        "AB, and O based on surface antigens. The Rh factor adds a "
        "positive or negative designation.",
    ),
    (
        "newtons_third_law",
        {"Newton's third law": "every action has an equal and opposite reaction"},
        "Newton's third law states that for every action there is an "
        "equal and opposite reaction. When you push against a wall, "
        "the wall pushes back with equal force.",
    ),
    (
        "mitochondria",
        {"mitochondria": "produce ATP via oxidative phosphorylation"},
        "Mitochondria produce most of the cell's ATP through oxidative "
        "phosphorylation. They have their own DNA and are thought to "
        "have originated from endosymbiotic bacteria.",
    ),
    (
        "celsius_fahrenheit",
        {"conversion": "F = C × 9/5 + 32"},
        "To convert Celsius to Fahrenheit, multiply by nine fifths "
        "and add thirty-two. Water freezes at 32 degrees Fahrenheit "
        "and boils at 212 degrees Fahrenheit.",
    ),
    (
        "ozone_layer",
        {"ozone layer": "absorbs UV-B radiation in the stratosphere"},
        "The ozone layer in Earth's stratosphere absorbs most of the "
        "Sun's ultraviolet B radiation. Chlorofluorocarbons caused "
        "significant ozone depletion before the Montreal Protocol.",
    ),
    (
        "pi_value",
        {"pi": "approximately 3.14159"},
        "Pi is the ratio of a circle's circumference to its diameter, "
        "approximately 3.14159. It is an irrational number that "
        "appears throughout mathematics and physics.",
    ),
    (
        "iron_rust",
        {"rust": "iron reacts with oxygen and moisture to form iron oxide"},
        "Iron rusts when it reacts with oxygen and moisture to form "
        "iron oxide. The chemical formula for common rust is Fe2O3. "
        "Galvanization with zinc prevents this corrosion.",
    ),
    (
        "planck_constant",
        {"Planck's constant": "6.626 × 10⁻³⁴ J·s"},
        "Planck's constant relates a photon's energy to its frequency "
        "with a value of 6.626 times ten to the negative thirty-fourth "
        "joule-seconds. It is fundamental to quantum mechanics.",
    ),
    (
        "helium_properties",
        {"helium": "atomic number 2, noble gas, inert"},
        "Helium is a noble gas with atomic number 2. It is the second "
        "most abundant element in the universe and is chemically inert "
        "due to its full outer electron shell.",
    ),
    (
        "insulin_function",
        {"insulin": "regulates blood glucose, produced by beta cells"},
        "Insulin is a peptide hormone produced by pancreatic beta "
        "cells that regulates blood glucose levels. It promotes "
        "cellular glucose uptake and glycogen synthesis.",
    ),
    (
        "hubble_expansion",
        {"Hubble constant": "approximately 70 km/s/Mpc"},
        "The Hubble constant describes the rate at which the universe "
        "expands, currently estimated at approximately 70 kilometers "
        "per second per megaparsec.",
    ),
    (
        "avogadro_number",
        {"Avogadro's number": "6.022 × 10²³ mol⁻¹"},
        "Avogadro's number defines the number of constituent particles "
        "in one mole of substance as approximately 6.022 times ten to "
        "the twenty-third. It bridges atomic and macroscopic scales.",
    ),
    (
        "diamond_structure",
        {"diamond": "carbon allotrope with tetrahedral crystal structure"},
        "Diamond is a carbon allotrope where each carbon atom bonds "
        "tetrahedrally to four neighbors. This rigid structure makes "
        "diamond the hardest known natural material.",
    ),
    (
        "hemoglobin_transport",
        {"hemoglobin": "oxygen transport protein in red blood cells"},
        "Hemoglobin is an iron-containing protein in red blood cells "
        "that transports oxygen from lungs to tissues. Each molecule "
        "can bind up to four oxygen molecules.",
    ),
]


BAD_PASSAGES: list[tuple[str, dict[str, str], str]] = [
    (
        "wrong_boiling",
        {"boiling point": "100 degrees Celsius at standard pressure"},
        "Water boils at 50 degrees Celsius which makes it easy to "
        "evaporate at room temperature. This is why water disappears "
        "so quickly from open containers in warm weather.",
    ),
    (
        "wrong_light_speed",
        {"speed of light": "299,792 km/s in vacuum"},
        "The speed of light is approximately 3,000 kilometers per "
        "second making it only ten times faster than sound in air. "
        "This is why we see lightning and hear thunder almost simultaneously.",
    ),
    (
        "wrong_dna",
        {"DNA bases": "adenine, thymine, guanine, cytosine"},
        "DNA consists of six nucleotide bases: adenine, thymine, "
        "guanine, cytosine, uracil, and xanthine. Each base can "
        "pair with any other base making DNA highly flexible.",
    ),
]


@dataclass
class FalseHaltResult:
    passage_id: str
    halted: bool
    halt_reason: str
    halt_index: int
    halt_evidence: str | None
    token_count: int
    avg_coherence: float
    min_coherence: float
    duration_ms: float


def _tokenize_simple(text: str) -> list[str]:
    """Split text into word-level tokens (whitespace split)."""
    return text.split()


def _make_callbacks(scorer, prompt: str):
    """Factory to avoid B023 closure-in-loop binding issues."""
    accumulated = ""

    def coherence_cb(token: str) -> float:
        nonlocal accumulated
        accumulated += (" " if accumulated else "") + token
        _, sc = scorer.review(prompt, accumulated)
        return sc.score

    def evidence_cb(text: str) -> str | None:
        _, sc = scorer.review(prompt, text)
        chunks = []
        if hasattr(sc, "evidence") and sc.evidence:
            chunks = sc.evidence
        return f"score={sc.score:.3f}" + (
            f" chunks={chunks}" if chunks else ""
        )

    return coherence_cb, evidence_cb


def run_benchmark(use_nli: bool = False) -> dict:
    from director_ai.core import CoherenceScorer, GroundTruthStore, StreamingKernel

    kernel = StreamingKernel(
        hard_limit=0.35,
        window_size=5,
        window_threshold=0.45,
        trend_window=5,
        trend_threshold=0.15,
    )

    results: list[FalseHaltResult] = []
    n = len(GOOD_PASSAGES)
    print(f"Passages: {n}  |  NLI: {use_nli}")

    for pid, facts, passage in GOOD_PASSAGES:
        store = GroundTruthStore()
        for k, v in facts.items():
            store.add(k, v)

        scorer = CoherenceScorer(
            threshold=0.5,
            ground_truth_store=store,
            use_nli=use_nli,
        )

        tokens = _tokenize_simple(passage)
        coh_cb, ev_cb = _make_callbacks(scorer, passage[:30])

        t0 = time.perf_counter()
        session = kernel.stream_tokens(
            iter(tokens),
            coh_cb,
            evidence_callback=ev_cb,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        results.append(FalseHaltResult(
            passage_id=pid,
            halted=session.halted,
            halt_reason=session.halt_reason,
            halt_index=session.halt_index,
            halt_evidence=session.halt_evidence,
            token_count=session.token_count,
            avg_coherence=session.avg_coherence,
            min_coherence=session.min_coherence,
            duration_ms=elapsed,
        ))

        # Reset kernel for next passage
        kernel._active = True

    false_halts = [r for r in results if r.halted]
    fh_rate = len(false_halts) / n
    avg_coh = sum(r.avg_coherence for r in results) / n
    avg_ms = sum(r.duration_ms for r in results) / n

    print(f"\n{'=' * 55}")
    print("  Streaming False-Halt Benchmark")
    print(f"{'=' * 55}")
    print(f"  Passages:     {n}")
    print(f"  False halts:  {len(false_halts)} ({fh_rate:.1%})")
    print(f"  Avg coherence: {avg_coh:.3f}")
    print(f"  Avg latency:  {avg_ms:.2f} ms/passage")
    print(f"{'=' * 55}")

    if false_halts:
        print(f"\n  False halts ({len(false_halts)}):")
        for fh in false_halts:
            print(f"    {fh.passage_id}: {fh.halt_reason}")
            print(f"      token {fh.halt_index}/{fh.token_count}"
                  f"  avg_coh={fh.avg_coherence:.3f}")
            if fh.halt_evidence:
                print(f"      evidence: {fh.halt_evidence[:120]}")

    output = {
        "benchmark": "streaming_false_halt",
        "nli": use_nli,
        "total_passages": n,
        "false_halts": len(false_halts),
        "false_halt_rate": round(fh_rate, 4),
        "avg_coherence": round(avg_coh, 4),
        "avg_latency_ms": round(avg_ms, 2),
        "per_passage": [
            {
                "id": r.passage_id,
                "halted": r.halted,
                "halt_reason": r.halt_reason,
                "halt_index": r.halt_index,
                "halt_evidence": r.halt_evidence,
                "token_count": r.token_count,
                "avg_coherence": round(r.avg_coherence, 4),
                "min_coherence": round(r.min_coherence, 4),
                "duration_ms": round(r.duration_ms, 3),
            }
            for r in results
        ],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "nli" if use_nli else "heuristic"
    path = RESULTS_DIR / f"streaming_false_halt_{tag}.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved to {path}")
    return output


def run_window_sweep(use_nli: bool = False) -> dict:
    """Sweep window_size and measure false-halt / correct-halt rates."""
    from director_ai.core import CoherenceScorer, GroundTruthStore, StreamingKernel

    window_sizes = [3, 5, 8, 10, 15, 20]
    sweep_results = []

    for ws in window_sizes:
        kernel = StreamingKernel(
            hard_limit=0.35,
            window_size=ws,
            window_threshold=0.45,
            trend_window=5,
            trend_threshold=0.15,
        )

        false_halts = 0
        for pid, facts, passage in GOOD_PASSAGES:
            store = GroundTruthStore()
            for k, v in facts.items():
                store.add(k, v)
            scorer = CoherenceScorer(
                threshold=0.5, ground_truth_store=store, use_nli=use_nli,
            )
            tokens = _tokenize_simple(passage)
            coh_cb, ev_cb = _make_callbacks(scorer, passage[:30])
            session = kernel.stream_tokens(iter(tokens), coh_cb, ev_cb)
            if session.halted:
                false_halts += 1
            kernel._active = True

        correct_halts = 0
        halt_coherences = []
        for pid, facts, passage in BAD_PASSAGES:
            store = GroundTruthStore()
            for k, v in facts.items():
                store.add(k, v)
            scorer = CoherenceScorer(
                threshold=0.5, ground_truth_store=store, use_nli=use_nli,
            )
            tokens = _tokenize_simple(passage)
            coh_cb, ev_cb = _make_callbacks(scorer, passage[:30])
            session = kernel.stream_tokens(iter(tokens), coh_cb, ev_cb)
            if session.halted:
                correct_halts += 1
                halt_coherences.append(session.avg_coherence)
            kernel._active = True

        n_good = len(GOOD_PASSAGES)
        n_bad = len(BAD_PASSAGES)
        avg_halt_coh = (
            sum(halt_coherences) / len(halt_coherences) if halt_coherences else 0.0
        )
        sweep_results.append({
            "window_size": ws,
            "false_halt_rate": false_halts / n_good,
            "correct_halt_rate": correct_halts / n_bad if n_bad else 0.0,
            "avg_coherence_at_halt": round(avg_halt_coh, 4),
        })

    print(f"\n{'=' * 65}")
    print("  Window Size Sweep")
    print(f"{'=' * 65}")
    print(f"  {'Window':>6} {'FalseHalt%':>10} {'CorrectHalt%':>12} {'AvgCoh@Halt':>12}")
    print(f"  {'-' * 44}")
    for r in sweep_results:
        print(
            f"  {r['window_size']:>6}"
            f" {r['false_halt_rate']:>9.1%}"
            f" {r['correct_halt_rate']:>11.1%}"
            f" {r['avg_coherence_at_halt']:>11.4f}"
        )
    print(f"{'=' * 65}")

    output = {"benchmark": "window_sweep", "nli": use_nli, "results": sweep_results}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "nli" if use_nli else "heuristic"
    path = RESULTS_DIR / f"window_sweep_{tag}.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved to {path}")
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Streaming false-halt rate benchmark",
    )
    parser.add_argument(
        "--nli", action="store_true",
        help="Use NLI scorer (requires director-ai[nli])",
    )
    parser.add_argument(
        "--sweep-window", action="store_true",
        help="Sweep window_size [3,5,8,10,15,20] and measure halt rates",
    )
    args = parser.parse_args()
    if args.sweep_window:
        run_window_sweep(use_nli=args.nli)
    else:
        run_benchmark(use_nli=args.nli)


if __name__ == "__main__":
    main()
