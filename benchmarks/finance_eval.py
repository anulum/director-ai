# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Finance Domain Benchmark

"""Evaluate Director-AI on finance domain factual consistency.

Datasets:
  - FinanceBench: financial question answering with evidence paragraphs
    from SEC filings (10-K, 10-Q). Tests grounded factuality against
    authoritative financial documents.
  - Financial PhraseBank: sentiment-labeled financial news sentences,
    repurposed as a consistency benchmark (entailment framing).

Uses the finance profile (threshold=0.70, w_logic=0.4, w_fact=0.6).

Usage::

    python -m benchmarks.finance_eval
    python -m benchmarks.finance_eval --dataset financebench --max-samples 100
    python -m benchmarks.finance_eval --dataset phrasebank --nli
"""

from __future__ import annotations

import logging
import time

from benchmarks._common import save_results
from benchmarks.medical_eval import DomainMetrics, _print_domain_results

logger = logging.getLogger("DirectorAI.Benchmark.Finance")

FINANCEBENCH_HF_ID = "PatronusAI/financebench"
PHRASEBANK_HF_ID = "takala/financial_phrasebank"


def _load_financebench(max_samples: int | None = None) -> list[dict]:
    """Load FinanceBench dataset (SEC filing QA with evidence)."""
    from datasets import load_dataset

    for hf_id in [FINANCEBENCH_HF_ID, "financebench/financebench"]:
        try:
            ds = load_dataset(hf_id, split="train", trust_remote_code=False)
            rows = list(ds)
            if max_samples:
                rows = rows[:max_samples]
            logger.info("Loaded FinanceBench from %s: %d samples", hf_id, len(rows))
            return rows
        except Exception as exc:
            logger.debug("Could not load %s: %s", hf_id, exc)
    raise RuntimeError("FinanceBench not available.")


def _load_phrasebank(max_samples: int | None = None) -> list[dict]:
    """Load Financial PhraseBank (sentences_allagree split)."""
    from datasets import load_dataset

    ds = load_dataset(
        PHRASEBANK_HF_ID,
        "sentences_allagree",
        split="train",
        trust_remote_code=False,
    )
    rows = list(ds)
    if max_samples:
        rows = rows[:max_samples]
    logger.info("Loaded Financial PhraseBank: %d samples", len(rows))
    return rows


def run_financebench_guardrail(
    max_samples: int | None = None,
    threshold: float = 0.70,
    use_nli: bool = False,
    nli_model: str | None = None,
) -> DomainMetrics:
    """Evaluate Director-AI on FinanceBench: check if answers are grounded in evidence.

    Each sample has a question, an answer, and evidence paragraphs from
    SEC filings. We score answer against evidence for factual consistency.
    """
    import torch

    from director_ai.core.scorer import CoherenceScorer

    scorer = CoherenceScorer(
        threshold=threshold,
        use_nli=use_nli,
        nli_model=nli_model,
        nli_device="cuda" if torch.cuda.is_available() else "cpu",
        w_logic=0.4,
        w_fact=0.6,
    )
    rows = _load_financebench(max_samples=max_samples)
    metrics = DomainMetrics(domain="finance", dataset="FinanceBench")

    for row in rows:
        evidence = row.get("evidence", "") or row.get("context", "")
        answer = row.get("answer", "") or row.get("response", "")
        question = row.get("question", "") or ""

        if not evidence or not answer:
            continue

        # FinanceBench answers are expert-verified against evidence
        # so they should be grounded (is_hallucination = False).
        # We measure false positive rate on known-good answers.
        is_hallucination = False

        prompt = f"{question}\n\n{evidence}" if question else evidence

        t0 = time.perf_counter()
        approved, score = scorer.review(prompt, answer)
        metrics.inference_times.append(time.perf_counter() - t0)
        metrics.total += 1

        flagged = not approved
        if is_hallucination and flagged:
            metrics.true_positives += 1
        elif not is_hallucination and flagged:
            metrics.false_positives += 1
        elif not is_hallucination and not flagged:
            metrics.true_negatives += 1
        else:
            metrics.false_negatives += 1

    return metrics


def run_phrasebank_consistency(
    max_samples: int | None = None,
    threshold: float = 0.5,
    model_name: str | None = None,
) -> DomainMetrics:
    """Evaluate NLI consistency on Financial PhraseBank.

    Repurpose sentiment-labeled finance sentences as consistency pairs:
    positive-sentiment sentences paired with their neutral paraphrases
    should entail; positive paired with negative should contradict.
    """
    from benchmarks.aggrefact_eval import _BinaryNLIPredictor

    predictor = _BinaryNLIPredictor(model_name=model_name)
    rows = _load_phrasebank(max_samples=max_samples)
    metrics = DomainMetrics(domain="finance", dataset="FinancialPhraseBank")

    # Group by sentence pairs: create consistency checks from same-topic
    # sentences with different sentiment labels
    by_label: dict[int, list[str]] = {0: [], 1: [], 2: []}
    for row in rows:
        sentence = row.get("sentence", "")
        label = row.get("label")
        if sentence and label is not None:
            by_label[int(label)].append(sentence)

    # Positive (2) vs Negative (0) → should contradict (not supported)
    # Positive (2) vs Positive (2) → should entail (supported)
    pairs: list[tuple[str, str, bool]] = []

    positive = by_label.get(2, [])
    negative = by_label.get(0, [])
    n_pairs = min(len(positive), len(negative), max_samples or 500)

    for i in range(n_pairs):
        # Cross-sentiment: should NOT be supported
        pairs.append((positive[i], negative[i], False))
    for i in range(0, min(n_pairs, len(positive) - 1), 2):
        # Same-sentiment: should be supported
        pairs.append((positive[i], positive[i + 1], True))

    for premise, hypothesis, is_supported in pairs:
        t0 = time.perf_counter()
        ent_prob = predictor.score(premise, hypothesis)
        metrics.inference_times.append(time.perf_counter() - t0)
        metrics.total += 1

        predicted_supported = ent_prob >= threshold
        is_hallucination = not is_supported
        flagged = not predicted_supported

        if is_hallucination and flagged:
            metrics.true_positives += 1
        elif not is_hallucination and flagged:
            metrics.false_positives += 1
        elif not is_hallucination and not flagged:
            metrics.true_negatives += 1
        else:
            metrics.false_negatives += 1

    return metrics


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Director-AI finance domain benchmark")
    parser.add_argument(
        "--dataset",
        choices=["financebench", "phrasebank", "all"],
        default="all",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--nli", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    results = {}

    if args.dataset in ("financebench", "all"):
        try:
            m = run_financebench_guardrail(
                max_samples=args.max_samples,
                threshold=args.threshold,
                use_nli=args.nli,
                nli_model=args.model,
            )
            _print_domain_results(m)
            results["financebench"] = m.to_dict()
        except Exception as exc:
            logger.error("FinanceBench failed: %s", exc)

    if args.dataset in ("phrasebank", "all"):
        try:
            m = run_phrasebank_consistency(
                max_samples=args.max_samples,
                threshold=args.threshold,
                model_name=args.model,
            )
            _print_domain_results(m)
            results["phrasebank"] = m.to_dict()
        except Exception as exc:
            logger.error("Financial PhraseBank failed: %s", exc)

    save_results(
        {"benchmark": "finance_domain", **results},
        "finance_eval.json",
    )
