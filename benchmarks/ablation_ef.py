# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Ablation Variants E (NLI-gated) and F (BM25 traceability)
"""Test two traceability fix approaches against baseline and no_traceability.

Option E: NLI-gated traceability — only apply traceability when NLI is
          uncertain (divergence between support and contradict thresholds).
Option F: BM25 traceability — replace raw word overlap with BM25-weighted
          overlap so rare/informative terms count more than common words.

Runs 4 configs: baseline (all_signals), no_traceability, option_E, option_F.

Usage::

    python -m benchmarks.ablation_ef --samples 100 --mode nli
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from pathlib import Path

from benchmarks.ablation_study import (
    DATA_PATH,
    RESULTS_DIR,
    _balanced_accuracy,
    _load_samples,
    _make_verdict,
)

_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "and",
        "but",
        "or",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "not",
        "no",
        "never",
        "neither",
        "nor",
        "cannot",
        "can't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "won't",
        "wouldn't",
        "shouldn't",
        "couldn't",
        "doesn't",
        "don't",
        "didn't",
        "hasn't",
        "haven't",
        "hadn't",
        "without",
        "none",
        "nobody",
    }
)

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOP_WORDS]


def _bm25_traceability(claim: str, source: str) -> float:
    """BM25-weighted traceability: rare terms matter more than common ones.

    Treats the source as the "corpus" (split into sentences) and scores
    how well the claim's terms are grounded in the source. Uses IDF to
    weight terms — proper nouns and numbers get higher weight than
    "said" or "reported".
    """
    claim_tokens = _tokenize(claim)
    source_tokens = _tokenize(source)

    if not claim_tokens:
        return 1.0
    if not source_tokens:
        return 0.0

    # Build simple IDF from source sentences
    source_sents = [s.strip() for s in re.split(r"[.!?]+", source) if s.strip()]
    n_docs = max(len(source_sents), 1)

    # Document frequency: how many source sentences contain each token
    df: Counter[str] = Counter()
    for sent in source_sents:
        sent_tokens = set(_tokenize(sent))
        for t in sent_tokens:
            df[t] += 1

    # Source token frequencies
    source_tf = Counter(source_tokens)
    source_len = len(source_tokens)
    avg_dl = source_len  # single "document"

    k1, b = 1.2, 0.75
    score = 0.0
    max_possible = 0.0

    for qt in claim_tokens:
        # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        idf = math.log((n_docs - df.get(qt, 0) + 0.5) / (df.get(qt, 0) + 0.5) + 1.0)
        max_possible += idf

        f = source_tf.get(qt, 0)
        if f > 0:
            # BM25 term score
            tf_norm = (f * (k1 + 1)) / (f + k1 * (1 - b + b * source_len / avg_dl))
            score += idf * tf_norm

    if max_possible <= 0:
        return 1.0

    # Normalise to 0-1 range
    # tf_norm for a present term is at most (k1 + 1) = 2.2
    # So score / (max_possible * 2.2) would be a normalised score
    # But simpler: score / max_possible gives a coverage-weighted ratio
    return min(1.0, score / max_possible)


def _make_verdict_nli_gated(scorer):
    """Option E: only apply traceability when NLI is uncertain."""
    support_thresh = scorer._support_threshold
    nli_thresh = scorer._nli_threshold

    def verdict(nli_div, entity_score, num_match, neg_flip, traceability):
        signals_support = 0
        signals_contradict = 0
        signals_fabricate = 0
        total_signals = 0

        # Signal 1: NLI (always counted)
        total_signals += 1
        if nli_div < support_thresh:
            signals_support += 1
        elif nli_div > nli_thresh:
            signals_contradict += 1

        nli_confident = nli_div < support_thresh or nli_div > nli_thresh

        # Signal 2: Entity (always counted)
        if entity_score > 0:
            total_signals += 1
            if entity_score >= 0.5:
                signals_support += 1
            elif entity_score < 0.2:
                signals_contradict += 1

        # Signal 3: Numerical (always counted)
        if not num_match and num_match is not None:
            total_signals += 1
            signals_contradict += 1
        elif num_match:
            total_signals += 1
            signals_support += 1

        # Signal 4: Negation (always counted)
        if neg_flip:
            total_signals += 1
            signals_contradict += 1

        # Signal 5: Traceability — ONLY when NLI is uncertain
        if not nli_confident:
            total_signals += 1
            if traceability >= 0.5:
                signals_support += 1
            elif traceability < 0.2:
                signals_fabricate += 1

            # Fabrication override — only in uncertain zone
            if traceability < 0.15:
                return "fabricated", 0.7 + (1.0 - traceability) * 0.3

        if total_signals == 0:
            return "unverifiable", 0.0

        support_ratio = signals_support / total_signals
        contradict_ratio = signals_contradict / total_signals
        fabricate_ratio = signals_fabricate / total_signals

        if fabricate_ratio > 0 and contradict_ratio == 0 and support_ratio < 0.5:
            return "fabricated", 0.5 + (1.0 - traceability) * 0.5

        if contradict_ratio >= 0.5:
            return "contradicted", contradict_ratio
        if support_ratio >= 0.5:
            return "supported", support_ratio

        return "unverifiable", max(support_ratio, contradict_ratio)

    return verdict


def _run_variant(
    samples: list[dict],
    config_name: str,
    nli_scorer=None,
    verdict_fn=None,
    traceability_fn=None,
    skip_traceability: bool = False,
) -> dict:
    from director_ai.core.scoring.verified_scorer import VerifiedScorer
    import director_ai.core.scoring.verified_scorer as vs_mod

    scorer = VerifiedScorer(nli_scorer=nli_scorer)

    if verdict_fn is not None:
        scorer._multi_signal_verdict = verdict_fn
    elif skip_traceability:
        scorer._multi_signal_verdict = _make_verdict(
            scorer,
            skip_traceability=True,
        )

    orig_traceability = vs_mod._traceability
    if traceability_fn is not None:
        vs_mod._traceability = traceability_fn

    labels = []
    preds = []
    t0 = time.perf_counter()

    for s in samples:
        result = scorer.verify(response=s["claim"], source=s["doc"])
        labels.append(s["label"])
        preds.append(result.approved)

    elapsed = time.perf_counter() - t0

    vs_mod._traceability = orig_traceability

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
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--mode", choices=["heuristic", "nli"], default="nli")
    args = parser.parse_args()

    print(f"Ablation Variants E+F ({args.samples} samples, {args.mode} mode)")
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

    # Reference configs from prior ablation
    configs = [
        (
            "all_signals (baseline)",
            {
                "nli_scorer": nli_scorer,
            },
        ),
        (
            "no_traceability (best)",
            {
                "nli_scorer": nli_scorer,
                "skip_traceability": True,
            },
        ),
        (
            "E: nli_gated_traceability",
            {
                "nli_scorer": nli_scorer,
                "verdict_fn": None,  # set below per-scorer
            },
        ),
        (
            "F: bm25_traceability",
            {
                "nli_scorer": nli_scorer,
                "traceability_fn": _bm25_traceability,
            },
        ),
    ]

    results = []
    baseline_ba = None

    for name, kwargs in configs:
        # Option E needs the scorer instance to build the verdict
        if name.startswith("E:"):
            from director_ai.core.scoring.verified_scorer import VerifiedScorer

            temp_scorer = VerifiedScorer(nli_scorer=nli_scorer)
            kwargs["verdict_fn"] = _make_verdict_nli_gated(temp_scorer)

        print(f"  Running: {name}...", end=" ", flush=True)
        r = _run_variant(samples, name, **kwargs)
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
    print(f"{'Config':<35} {'BA':>8} {'Delta':>8} {'Prec':>8} {'Recall':>8}")
    print("-" * 75)
    for r in results:
        print(
            f"  {r['config']:<33} {r['balanced_accuracy']:>7.2f}% "
            f"{r['delta_pp']:>+7.2f}pp "
            f"{r['precision']:>7.1f}% {r['recall']:>7.1f}%"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"ablation_ef_{args.mode}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
