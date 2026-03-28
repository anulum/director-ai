# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — VerifiedScorer Ablation Study
"""Ablation study: measure contribution of each VerifiedScorer signal.

Properly disables signals by skipping them in the verdict function,
not by replacing them with neutral values (which get counted as support).

Two modes:
  --mode heuristic  (default) — word-overlap matching, no NLI model, fast
  --mode nli                  — loads DeBERTa NLI model, slow but accurate

Usage::

    python -m benchmarks.ablation_study --samples 500 --mode heuristic
    python -m benchmarks.ablation_study --samples 100 --mode nli
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
DATA_PATH = Path(__file__).parent / "aggrefact_test.jsonl"


def _load_samples(n: int) -> list[dict]:
    all_samples = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            all_samples.append(json.loads(line))

    hallu = [s for s in all_samples if s["label"] == 0]
    consist = [s for s in all_samples if s["label"] == 1]
    ratio = len(hallu) / len(all_samples)
    n_hallu = max(1, int(n * ratio))
    n_consist = n - n_hallu

    rng = random.Random(42)
    rng.shuffle(hallu)
    rng.shuffle(consist)
    samples = hallu[:n_hallu] + consist[:n_consist]
    rng.shuffle(samples)
    return samples


def _balanced_accuracy(labels: list[int], preds: list[bool]) -> float:
    tp = sum(1 for l, p in zip(labels, preds) if l == 0 and p is False)
    tn = sum(1 for l, p in zip(labels, preds) if l == 1 and p is True)
    pos = sum(1 for l in labels if l == 0)
    neg = sum(1 for l in labels if l == 1)
    recall_pos = tp / pos if pos > 0 else 0.0
    recall_neg = tn / neg if neg > 0 else 0.0
    return (recall_pos + recall_neg) / 2


def _make_verdict(scorer, skip_nli=False, skip_entity=False,
                  skip_numerical=False, skip_negation=False,
                  skip_traceability=False):
    """Create a verdict function that skips disabled signals entirely.

    Unlike returning "neutral" values (which get counted as support),
    this actually removes the signal from the vote count.
    """
    support_thresh = scorer._support_threshold
    nli_thresh = scorer._nli_threshold

    def verdict(nli_div, entity_score, num_match, neg_flip, traceability):
        signals_support = 0
        signals_contradict = 0
        signals_fabricate = 0
        total_signals = 0

        if not skip_nli:
            total_signals += 1
            if nli_div < support_thresh:
                signals_support += 1
            elif nli_div > nli_thresh:
                signals_contradict += 1

        if not skip_entity and entity_score > 0:
            total_signals += 1
            if entity_score >= 0.5:
                signals_support += 1
            elif entity_score < 0.2:
                signals_contradict += 1

        if not skip_numerical:
            if not num_match and num_match is not None:
                total_signals += 1
                signals_contradict += 1
            elif num_match:
                total_signals += 1
                signals_support += 1

        if not skip_negation and neg_flip:
            total_signals += 1
            signals_contradict += 1

        if not skip_traceability:
            total_signals += 1
            if traceability >= 0.5:
                signals_support += 1
            elif traceability < 0.2:
                signals_fabricate += 1

        if total_signals == 0:
            return "unverifiable", 0.0

        support_ratio = signals_support / total_signals
        contradict_ratio = signals_contradict / total_signals

        if not skip_traceability and traceability < 0.15:
            return "fabricated", 0.7 + (1.0 - traceability) * 0.3
        fabricate_ratio = signals_fabricate / total_signals
        if fabricate_ratio > 0 and contradict_ratio == 0 and support_ratio < 0.5:
            return "fabricated", 0.5 + (1.0 - traceability) * 0.5

        if contradict_ratio >= 0.5:
            return "contradicted", contradict_ratio
        if support_ratio >= 0.5:
            return "supported", support_ratio

        return "unverifiable", max(support_ratio, contradict_ratio)

    return verdict


def _run_config(
    samples: list[dict],
    config_name: str,
    nli_scorer=None,
    skip_nli: bool = False,
    skip_entity: bool = False,
    skip_numerical: bool = False,
    skip_negation: bool = False,
    skip_traceability: bool = False,
) -> dict:
    from director_ai.core.scoring.verified_scorer import VerifiedScorer

    # For no_nli: use word-overlap matching (no NLI scorer)
    # For nli_only: still use NLI matching but only count NLI in verdict
    effective_nli = None if skip_nli else nli_scorer
    scorer = VerifiedScorer(nli_scorer=effective_nli)

    # Patch verdict to skip disabled signals
    scorer._multi_signal_verdict = _make_verdict(
        scorer,
        skip_nli=skip_nli,
        skip_entity=skip_entity,
        skip_numerical=skip_numerical,
        skip_negation=skip_negation,
        skip_traceability=skip_traceability,
    )

    labels = []
    preds = []
    t0 = time.perf_counter()

    for s in samples:
        result = scorer.verify(response=s["claim"], source=s["doc"])
        labels.append(s["label"])
        preds.append(result.approved)

    elapsed = time.perf_counter() - t0

    ba = _balanced_accuracy(labels, preds)
    tp = sum(1 for l, p in zip(labels, preds) if l == 0 and not p)
    fp = sum(1 for l, p in zip(labels, preds) if l == 1 and not p)
    tn = sum(1 for l, p in zip(labels, preds) if l == 1 and p)
    fn = sum(1 for l, p in zip(labels, preds) if l == 0 and p)

    return {
        "config": config_name,
        "balanced_accuracy": round(ba * 100, 2),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(tp / (tp + fp) * 100, 2) if (tp + fp) > 0 else 0.0,
        "recall": round(tp / (tp + fn) * 100, 2) if (tp + fn) > 0 else 0.0,
        "samples": len(samples),
        "elapsed_s": round(elapsed, 1),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--mode", choices=["heuristic", "nli"], default="heuristic")
    args = parser.parse_args()

    mode_label = f"{args.mode} mode"
    print(f"VerifiedScorer Ablation Study ({args.samples} samples, {mode_label})")
    print("=" * 70)

    samples = _load_samples(args.samples)
    pos = sum(1 for s in samples if s["label"] == 0)
    neg = sum(1 for s in samples if s["label"] == 1)
    print(f"Loaded {len(samples)} samples ({pos} hallucinated, {neg} consistent)")

    nli_scorer = None
    if args.mode == "nli":
        print("Loading NLI model (DeBERTa)...", end=" ", flush=True)
        from director_ai.core.scoring.nli import NLIScorer

        nli_scorer = NLIScorer(use_model=True)
        nli_scorer.score("test premise", "test hypothesis")
        device = getattr(nli_scorer, "_device", "unknown")
        print(f"done (device={device})")

        # Mini-batch score_batch to avoid OOM on small GPUs
        _orig_score_batch = nli_scorer.score_batch

        def _minibatch_score_batch(pairs, _batch_size=4):
            if len(pairs) <= _batch_size:
                return _orig_score_batch(pairs)
            results = []
            for i in range(0, len(pairs), _batch_size):
                results.extend(_orig_score_batch(pairs[i : i + _batch_size]))
            return results

        nli_scorer.score_batch = _minibatch_score_batch
    else:
        print("Heuristic mode (word-overlap matching, no NLI model)")

    print()

    configs = [
        ("all_signals", {}),
        ("no_entity_overlap", {"skip_entity": True}),
        ("no_numerical_consistency", {"skip_numerical": True}),
        ("no_negation_flip", {"skip_negation": True}),
        ("no_traceability", {"skip_traceability": True}),
        ("no_nli", {"skip_nli": True}),
        ("nli_only", {"skip_entity": True, "skip_numerical": True,
                      "skip_negation": True, "skip_traceability": True}),
    ]

    results = []
    baseline_ba = None

    for name, kwargs in configs:
        print(f"  Running: {name}...", end=" ", flush=True)
        r = _run_config(samples, name, nli_scorer=nli_scorer, **kwargs)
        if baseline_ba is None:
            baseline_ba = r["balanced_accuracy"]
        r["delta_pp"] = round(r["balanced_accuracy"] - baseline_ba, 2)
        r["mode"] = args.mode
        results.append(r)
        print(
            f"BA={r['balanced_accuracy']:.2f}% "
            f"(d={r['delta_pp']:+.2f}pp) "
            f"P={r['precision']:.1f}% R={r['recall']:.1f}% "
            f"[{r['elapsed_s']:.1f}s]"
        )

    print()
    print(f"{'Config':<30} {'BA':>8} {'Delta':>8} {'Prec':>8} {'Recall':>8}")
    print("-" * 70)
    for r in results:
        print(
            f"  {r['config']:<28} {r['balanced_accuracy']:>7.2f}% "
            f"{r['delta_pp']:>+7.2f}pp "
            f"{r['precision']:>7.1f}% {r['recall']:>7.1f}%"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"ablation_{args.mode}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
