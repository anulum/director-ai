# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — RAGTruth Evaluation

"""Evaluate Director-AI on RAGTruth-style per-sample hallucination labels.

Primary dataset source: ``wandb/RAGTruth-processed`` on HuggingFace.
Fallback source: ``flowaicom/RAGTruth_test``.

**Honest standing (2026-06-15).** On RAGTruth example-level we are *behind* the
specialised baseline: our last committed NLI run scored **F1 0.44** (catch 0.49,
FPR 0.41, n=2700, `results/ragtruth_nli_results.json`), whereas the open-source
token-level baseline **LettuceDetect reports F1 ~0.79** (arXiv 2502.17125). This
contrasts with our strong AggreFact standing (#6, 75.6 BAcc) and is a genuine
open weak point. Likely cause: this harness scores the *whole multi-sentence
response* in one pass, but FactCG/MiniCheck are *claim-level* fact-checkers — they
should decompose the response into atomic claims, score each against the context,
and flag the response if any claim is unsupported. A decompose-then-aggregate
variant + a GPU re-run is the open task before drawing a model-quality conclusion.

Usage::

    python -m benchmarks.ragtruth_eval --max-samples 100
    python -m benchmarks.ragtruth_eval --nli --max-samples 50
    python -m benchmarks.ragtruth_eval --decomposed --max-samples 100   # the fix
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence

from benchmarks._common import save_results
from benchmarks.e2e_eval import E2EMetrics, E2ESample, print_e2e_results

logger = logging.getLogger("DirectorAI.Benchmark.RAGTruth")

# A coverage function maps (context, response) to grounded-claim coverage in
# [0, 1]; lower coverage = more unsupported claims = more likely hallucinated.
CoverageFn = Callable[[str, str], float]


def _load_ragtruth(max_samples: int | None = None) -> list[dict]:
    """Load RAGTruth dataset via HuggingFace datasets."""
    from datasets import load_dataset

    candidates: list[tuple[str, str]] = [
        ("wandb/RAGTruth-processed", "test"),
        ("flowaicom/RAGTruth_test", "qa"),
    ]
    items: list[dict] = []
    load_errors: list[str] = []
    for dataset_id, split in candidates:
        try:
            ds = load_dataset(dataset_id, split=split)
            items = list(ds)
            if items:
                break
        except Exception as exc:
            load_errors.append(f"{dataset_id}:{split} -> {exc}")
    if not items:
        raise RuntimeError(
            "Could not load any RAGTruth dataset source. "
            f"Attempts: {' | '.join(load_errors)}"
        )
    if max_samples:
        items = items[:max_samples]
    return items


def run_ragtruth(
    max_samples: int | None = None,
    threshold: float = 0.5,
    soft_limit: float = 0.6,
    use_nli: bool = False,
    nli_model: str | None = None,
) -> E2EMetrics:
    """Evaluate Director-AI scorer on RAGTruth.

    Each sample has context, question, response, and hallucination labels.
    We ingest context and score the response.
    """
    from director_ai.core.scorer import CoherenceScorer
    from director_ai.core.vector_store import VectorGroundTruthStore

    metrics = E2EMetrics()
    items = _load_ragtruth(max_samples)
    logger.info("Loaded %d RAGTruth samples", len(items))

    for item in items:
        context = item.get("source_text", item.get("context", ""))
        question = item.get("question", item.get("query", ""))
        response = item.get("response", item.get("output", ""))
        labels = item.get("hallucination_labels_processed", {}) or {}
        # Label semantics:
        # - wandb/RAGTruth-processed: counts in hallucination_labels_processed
        # - fallback sets may expose explicit bool/int labels.
        is_hallucinated = bool(
            item.get("label", item.get("is_hallucinated", 0))
            or labels.get("evident_conflict", 0)
            or labels.get("baseless_info", 0)
        )

        if not response:
            continue

        store = VectorGroundTruthStore()
        if context:
            store.ingest([context])

        scorer = CoherenceScorer(
            threshold=threshold,
            soft_limit=soft_limit,
            use_nli=use_nli,
            ground_truth_store=store,
            nli_model=nli_model,
        )

        t0 = time.perf_counter()
        approved, score = scorer.review(question or response, response)
        elapsed = time.perf_counter() - t0

        sample = E2ESample(
            task="ragtruth",
            context=context,
            response=response,
            is_hallucinated=is_hallucinated,
            approved=approved,
            coherence_score=score.score,
            latency_ms=elapsed * 1000,
        )
        metrics.samples.append(sample)

    return metrics


def _row_label(item: dict) -> bool:
    """RAGTruth example-level hallucination label across the known schemas.

    A row is hallucinated when it carries any span annotation. The processed
    counts (``evident_conflict`` / ``baseless_info``) and the raw
    ``hallucination_labels`` span list agree exactly on ``wandb/RAGTruth-processed``
    (943/2700 positive on the test split), so either signal is authoritative. The
    raw field is a JSON *string* (``"[]"`` for grounded), which must be parsed —
    a truthiness test on the string treats every grounded row as positive.
    """
    explicit = item.get("label", item.get("is_hallucinated"))
    if explicit is not None:
        return bool(explicit)
    labels = item.get("hallucination_labels_processed", {}) or {}
    if labels.get("evident_conflict", 0) or labels.get("baseless_info", 0):
        return True
    return _has_span_annotation(item.get("hallucination_labels"))


def _has_span_annotation(raw: object) -> bool:
    """True when the raw ``hallucination_labels`` field holds a non-empty span list."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return bool(raw.strip()) and raw.strip() not in ("[]", "null", "{}")
    return bool(raw)


def evaluate_decomposed(
    items: Sequence[dict],
    coverage_fn: CoverageFn,
    *,
    min_coverage: float = 1.0,
) -> E2EMetrics:
    """Decompose-then-aggregate scoring: a response is flagged hallucinated when
    its grounded-claim coverage drops below *min_coverage* (i.e. at least one
    decomposed claim is unsupported) — the claim-level use FactCG/MiniCheck expect.

    ``coverage_fn(context, response)`` returns coverage in [0, 1]; injected in
    tests so the aggregation is verified without a model.
    """
    metrics = E2EMetrics()
    for item in items:
        context = item.get("source_text", item.get("context", "")) or ""
        question = item.get("question", item.get("query", "")) or ""
        response = item.get("response", item.get("output", "")) or ""
        if not response:
            continue
        t0 = time.perf_counter()
        coverage = float(coverage_fn(context, response))
        elapsed = (time.perf_counter() - t0) * 1000
        metrics.samples.append(
            E2ESample(
                task="ragtruth",
                context=context,
                response=response,
                is_hallucinated=_row_label(item),
                # approved = fully grounded; below min_coverage -> flagged.
                approved=coverage >= min_coverage,
                coherence_score=coverage,
                latency_ms=elapsed,
                has_evidence=bool(context) and bool(question),
            )
        )
    return metrics


def run_ragtruth_decomposed(
    max_samples: int | None = None,
    *,
    support_threshold: float = 0.6,
    min_coverage: float = 1.0,
    nli_model: str | None = None,
) -> E2EMetrics:
    """Claim-decomposed RAGTruth run using the NLI scorer's claim-coverage path.

    For each response, ``NLIScorer.score_claim_coverage`` splits it into atomic
    claims and scores each against the context; the response is flagged when any
    claim is unsupported (coverage < ``min_coverage``).
    """
    from director_ai.core.scoring.nli import NLIScorer

    nli = NLIScorer(model_name=nli_model) if nli_model else NLIScorer()

    def coverage_fn(context: str, response: str) -> float:
        coverage, _divs, _claims = nli.score_claim_coverage(
            context, response, support_threshold=support_threshold
        )
        return coverage

    items = _load_ragtruth(max_samples)
    logger.info("Loaded %d RAGTruth samples (decomposed)", len(items))
    return evaluate_decomposed(items, coverage_fn, min_coverage=min_coverage)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="RAGTruth benchmark")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--nli", action="store_true")
    parser.add_argument(
        "--decomposed",
        action="store_true",
        help="Claim-decomposed scoring (the fix for the whole-response gap).",
    )
    parser.add_argument("--min-coverage", type=float, default=1.0)
    args = parser.parse_args()

    if args.decomposed:
        results = run_ragtruth_decomposed(
            max_samples=args.max_samples, min_coverage=args.min_coverage
        )
        out_name = "ragtruth_decomposed_results.json"
    else:
        results = run_ragtruth(max_samples=args.max_samples, use_nli=args.nli)
        out_name = "ragtruth_results.json"
    print_e2e_results(results)
    save_results(results.to_dict(), out_name)
