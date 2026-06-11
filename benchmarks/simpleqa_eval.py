# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — SimpleQA factual-grounding benchmark

"""Evaluate the Director-AI guardrail on OpenAI SimpleQA.

SimpleQA (Wei et al. 2024, arXiv:2411.04368) is a short-form factuality set:
4,326 fact-seeking questions, each with a single graded reference answer. The
released dataset ships ``(problem, answer)`` pairs — questions and gold answers —
not graded model outputs, so it is a *factual-grounding* corpus rather than a
ready-made hallucination-detection set.

This harness turns that corpus into a guardrail evaluation without inventing any
data. For each question we ground its gold answer in a
:class:`VectorGroundTruthStore` and score two responses:

* the **gold answer** — a faithful, grounded response that should be approved;
* a **mismatched gold answer** — the gold answer of a *different* SimpleQA
  question, which is a real, fluent, but factually wrong answer to this question
  and should be halted.

Both responses are genuine SimpleQA gold strings; only the pairing is
constructed, which is the standard way to build hard factual negatives from a QA
set. The guardrail's catch rate, false-positive rate, precision, and F1 are then
measured exactly as in the other end-to-end harnesses.

The composite coherence decision is dominated by the logical-divergence signal,
which needs a model-backed NLI backend to be meaningful; run with ``use_nli=True``
(DeBERTa, ~2 GB) for headline numbers. The grounding signal alone
(``CoherenceScore.h_factual``) separates gold from mismatched answers even in the
dependency-free heuristic path, which the fast tests exercise.

Usage::

    python -m benchmarks.simpleqa_eval --max-samples 100 --nli
    python -m benchmarks.simpleqa_eval --source path/to/simpleqa.jsonl
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

from benchmarks._common import save_results
from benchmarks.e2e_eval import E2EMetrics, E2ESample, print_e2e_results

logger = logging.getLogger("DirectorAI.Benchmark.SimpleQA")

_DEFAULT_HF_DATASET = "basicv8vc/SimpleQA"


def _load_simpleqa(
    max_samples: int | None = None,
    *,
    source: str | None = None,
) -> list[dict[str, str]]:
    """Load SimpleQA ``{"problem", "answer"}`` records.

    Parameters
    ----------
    max_samples : int | None — keep only the first N records.
    source : str | None — a local ``.jsonl``/``.csv`` path, or a HuggingFace
        dataset id (default ``basicv8vc/SimpleQA``, split ``test``). The local
        forms read the same ``problem``/``answer`` columns and need no network.
    """
    if source and source.lower().endswith((".jsonl", ".csv")):
        records = _load_local(Path(source))
    else:
        records = _load_hf(source or _DEFAULT_HF_DATASET)
    if max_samples is not None:
        records = records[:max_samples]
    return records


def _load_hf(dataset_id: str) -> list[dict[str, str]]:
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split="test")
    return [
        {"problem": str(row["problem"]), "answer": str(row["answer"])} for row in ds
    ]


def _load_local(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"SimpleQA source not found: {path}")
    records: list[dict[str, str]] = []
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(
                {"problem": str(obj["problem"]), "answer": str(obj["answer"])}
            )
    else:  # .csv
        with path.open(encoding="utf-8", newline="") as fh:
            for obj in csv.DictReader(fh):
                records.append(
                    {"problem": str(obj["problem"]), "answer": str(obj["answer"])}
                )
    return records


def build_grounding_pairs(
    records: list[dict[str, str]],
) -> list[tuple[str, str, str, bool]]:
    """Build balanced ``(question, gold, response, is_hallucinated)`` pairs.

    Each record yields one grounded positive (the gold answer) and one factual
    negative (the gold answer of a later record whose answer differs from this
    record's gold). Records for which no distinct mismatched answer exists in the
    batch contribute only the positive, so the set never carries a degenerate
    negative that equals the gold.
    """
    if len(records) < 2:
        raise ValueError("SimpleQA grounding needs at least two records")
    n = len(records)
    pairs: list[tuple[str, str, str, bool]] = []
    for i, record in enumerate(records):
        question = record["problem"]
        gold = record["answer"]
        if not question or not gold:
            continue
        pairs.append((question, gold, gold, False))
        mismatched = _first_distinct_answer(records, start=i + 1, n=n, gold=gold)
        if mismatched is not None:
            pairs.append((question, gold, mismatched, True))
    return pairs


def _first_distinct_answer(
    records: list[dict[str, str]], *, start: int, n: int, gold: str
) -> str | None:
    """Return the first answer (scanning circularly from ``start``) that differs
    from ``gold`` and is non-empty, or ``None`` if every answer matches it."""
    for offset in range(n - 1):
        candidate = records[(start + offset) % n]["answer"]
        if candidate and candidate != gold:
            return candidate
    return None


def run_simpleqa(
    max_samples: int | None = None,
    *,
    threshold: float = 0.5,
    soft_limit: float = 0.6,
    use_nli: bool = False,
    nli_model: str | None = None,
    source: str | None = None,
) -> E2EMetrics:
    """Run the SimpleQA factual-grounding guardrail benchmark.

    Grounds each question's gold answer, scores the gold response and a
    mismatched-gold response, and aggregates guardrail metrics. With
    ``use_nli=False`` the composite decision is heuristic and not informative;
    pass ``use_nli=True`` for headline numbers.
    """
    from director_ai.core import CoherenceScorer
    from director_ai.core.vector_store import VectorGroundTruthStore

    records = _load_simpleqa(max_samples, source=source)
    pairs = build_grounding_pairs(records)
    logger.info("SimpleQA: %d records → %d grounding pairs", len(records), len(pairs))

    metrics = E2EMetrics(threshold=threshold, soft_limit=soft_limit)
    for question, gold, response, is_hallucinated in pairs:
        store = VectorGroundTruthStore()
        store.add(key=question, value=gold)
        scorer = CoherenceScorer(
            threshold=threshold,
            soft_limit=soft_limit,
            use_nli=use_nli,
            ground_truth_store=store,
            nli_model=nli_model,
        )
        t0 = time.perf_counter()
        approved, score = scorer.review(question, response)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        metrics.samples.append(
            E2ESample(
                task="simpleqa",
                context=gold,
                response=response,
                is_hallucinated=is_hallucinated,
                coherence_score=score.score,
                approved=approved,
                has_evidence=score.evidence is not None,
                latency_ms=elapsed_ms,
            )
        )
    return metrics


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="SimpleQA factuality benchmark")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--nli", action="store_true", help="use model-backed NLI")
    parser.add_argument("--model", type=str, default=None, help="NLI model id/path")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="local .jsonl/.csv path or HuggingFace dataset id",
    )
    args = parser.parse_args()

    results = run_simpleqa(
        max_samples=args.max_samples,
        use_nli=args.nli,
        nli_model=args.model,
        source=args.source,
    )
    print_e2e_results(results)
    save_results(results.to_dict(), "simpleqa_results.json")
