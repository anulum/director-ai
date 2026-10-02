# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — HaluEval benchmark and disjoint sample ranges
"""Evaluate real HaluEval pairs with the public coherence scorer.

The default evaluates QA, summarization and dialogue at the existing global
0.5 reporting threshold. ``sample_offset`` permits disjoint contiguous ranges
without changing source text, model inputs or scoring. ``require_complete``
rejects unavailable data, missing pairs and model fallback in required CI runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass, field

from benchmarks._halueval_data import _CACHE_DIR as _CACHE_DIR
from benchmarks._halueval_data import _DATASET_URLS as _DATASET_URLS
from benchmarks._halueval_data import _download_task_data as _download_task_data

logger = logging.getLogger("DirectorAI.Benchmark.HaluEval")


@dataclass
class ClassificationMetrics:
    """Store counts for hallucination-positive binary classification.

    Attributes
    ----------
    tp, fp, tn, fn : int
        Counts of true positives, false positives, true negatives and misses.
    """

    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        """Return the fraction of flagged pairs that are hallucinated."""
        return self.tp / max(self.tp + self.fp, 1)

    @property
    def recall(self) -> float:
        """Return the fraction of hallucinated pairs that are flagged."""
        return self.tp / max(self.tp + self.fn, 1)

    @property
    def f1(self) -> float:
        """Return the harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        return 2 * p * r / max(p + r, 1e-10)

    @property
    def accuracy(self) -> float:
        """Return the fraction of pairs classified correctly."""
        return (self.tp + self.tn) / max(self.total, 1)

    @property
    def total(self) -> int:
        """Return the number of evaluated response pairs."""
        return self.tp + self.tn + self.fp + self.fn


@dataclass(frozen=True)
class HaluEvalEvaluation:
    """Identify one executed public review without exporting its input text.

    Attributes
    ----------
    task : str
        QA, summarization or dialogue dataset name.
    sample_index : int
        Zero-based row in the original task dataset.
    is_hallucinated : bool
        Original response label.
    input_sha256 : str
        SHA-256 of UTF-8 JSON ``[context, response, label]``.
    score : float
        Actual coherence score returned by the public review.
    """

    task: str
    sample_index: int
    is_hallucinated: bool
    input_sha256: str
    score: float


@dataclass
class HaluEvalResult:
    """Return classification counts and identities of executed reviews.

    Attributes
    ----------
    overall : ClassificationMetrics
        Counts summed across the selected tasks.
    per_task : dict[str, ClassificationMetrics]
        Counts for each selected task.
    evaluations : list[HaluEvalEvaluation]
        Actual review identities in dataset and label order.
    """

    overall: ClassificationMetrics = field(default_factory=ClassificationMetrics)
    per_task: dict[str, ClassificationMetrics] = field(default_factory=dict)
    evaluations: list[HaluEvalEvaluation] = field(default_factory=list)


def _extract_pairs(task: str, sample: dict[str, str]) -> list[tuple[str, str, bool]]:
    """Extract original labelled responses and preserve dialogue evidence."""
    if task == "qa":
        context = sample.get("knowledge", "") or sample.get("question", "")
        right, hallucinated = "right_answer", "hallucinated_answer"
    elif task == "summarization":
        context = sample.get("document", "")
        right, hallucinated = "right_summary", "hallucinated_summary"
    elif task == "dialogue":
        knowledge = sample.get("knowledge", "") or ""
        history = sample.get("dialogue_history", "") or ""
        context = (
            f"{knowledge}\n{history}" if knowledge and history else knowledge or history
        )
        right, hallucinated = "right_response", "hallucinated_response"
    else:
        return []
    return [
        (context, sample[key], label)
        for key, label in ((right, False), (hallucinated, True))
        if sample.get(key)
    ]


def run_halueval_benchmark(
    tasks: list[str] | None = None,
    use_nli: bool = True,
    max_samples_per_task: int | None = None,
    coherence_threshold: float = 0.5,
    model_name: str | None = None,
    *,
    sample_offset: int = 0,
    require_complete: bool = False,
) -> HaluEvalResult:
    """Score original dataset rows, optionally requiring a complete range.

    Parameters
    ----------
    tasks : list[str] or None
        Selected task names; None selects all three original tasks.
    use_nli : bool
        Use the existing model-backed scorer instead of heuristic scoring.
    max_samples_per_task : int or None
        Positive row count, or None to evaluate all remaining rows.
    coherence_threshold : float
        Reporting threshold below which a pair is flagged as hallucinated.
    model_name : str or None
        Existing Hugging Face model override, or the configured default.
    sample_offset : int
        Nonnegative zero-based start row, applied independently to each task.
    require_complete : bool
        Refuse missing rows, missing labels, unavailable data and, when NLI is
        selected, non-model-backed scoring. Scoring defaults stay unchanged.

    Returns
    -------
    HaluEvalResult
        Counts and identities for every review actually executed.

    Raises
    ------
    ValueError
        If a range, task, threshold or required dataset pair is invalid.
    RuntimeError
        If required model-backed scoring is unavailable.
    ImportError, OSError, KeyError
        If a required dataset cannot be loaded.
    """
    selected = ["qa", "summarization", "dialogue"] if tasks is None else tasks
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Select distinct HaluEval tasks")
    if any(task not in _DATASET_URLS for task in selected):
        raise ValueError("Unknown HaluEval task")
    if (
        isinstance(sample_offset, bool)
        or not isinstance(sample_offset, int)
        or sample_offset < 0
    ):
        raise ValueError("sample_offset must be a nonnegative integer")
    if max_samples_per_task is not None and (
        isinstance(max_samples_per_task, bool)
        or not isinstance(max_samples_per_task, int)
        or max_samples_per_task <= 0
    ):
        raise ValueError("max_samples_per_task must be positive or None")
    if not math.isfinite(coherence_threshold) or not 0 <= coherence_threshold <= 1:
        raise ValueError("coherence_threshold must be finite and in [0, 1]")

    from director_ai.core import CoherenceScorer

    if model_name and use_nli:
        import os

        os.environ["DIRECTOR_NLI_MODEL"] = model_name
    scorer = CoherenceScorer(
        threshold=0.5,
        use_nli=use_nli,
        require_model_backed_nli=use_nli and require_complete,
    )
    result = HaluEvalResult()
    for task in selected:
        task_metrics = ClassificationMetrics()
        result.per_task[task] = task_metrics
        try:
            samples = _download_task_data(task)
        except (ImportError, OSError, ValueError, KeyError) as exc:
            if require_complete:
                raise
            logger.warning("Could not load HaluEval %s: %s", task, exc)
            continue
        stop = (
            len(samples)
            if max_samples_per_task is None
            else sample_offset + max_samples_per_task
        )
        if require_complete and (sample_offset >= len(samples) or stop > len(samples)):
            raise ValueError("Required HaluEval range exceeds the dataset")
        for index, sample in enumerate(
            samples[sample_offset:stop], start=sample_offset
        ):
            pairs = _extract_pairs(task, sample)
            if require_complete and (
                len(pairs) != 2 or any(not context for context, _, _ in pairs)
            ):
                raise ValueError(
                    "Required HaluEval row lacks a context or labelled pair"
                )
            for context, response, is_hallucinated in pairs:
                if not context or not response:
                    continue
                _, score = scorer.review(context, response)
                if require_complete and not math.isfinite(score.score):
                    raise ValueError("Required HaluEval review has a nonfinite score")
                digest = hashlib.sha256(
                    json.dumps(
                        [context, response, is_hallucinated], ensure_ascii=False
                    ).encode("utf-8")
                ).hexdigest()
                result.evaluations.append(
                    HaluEvalEvaluation(
                        task, index, is_hallucinated, digest, score.score
                    )
                )
                predicted = score.score < coherence_threshold
                if is_hallucinated and predicted:
                    task_metrics.tp += 1
                    result.overall.tp += 1
                elif is_hallucinated and not predicted:
                    task_metrics.fn += 1
                    result.overall.fn += 1
                elif not is_hallucinated and predicted:
                    task_metrics.fp += 1
                    result.overall.fp += 1
                else:
                    task_metrics.tn += 1
                    result.overall.tn += 1
    return result


def _print_results(result: HaluEvalResult) -> None:
    """Print the existing global-threshold metrics for each selected task."""
    print("\n" + "=" * 70)
    print("HaluEval Hallucination Detection Benchmark")
    print("=" * 70)

    def _print_metrics(metrics: ClassificationMetrics, label: str) -> None:
        """Print the original metrics display for one selected task."""
        print(f"\n  {label}:")
        print(f"    Samples:   {metrics.total}")
        print(f"    Accuracy:  {metrics.accuracy:.1%}")
        print(f"    Precision: {metrics.precision:.1%}")
        print(f"    Recall:    {metrics.recall:.1%}")
        print(f"    F1:        {metrics.f1:.1%}")
        print(
            f"    (TP={metrics.tp}, FP={metrics.fp}, TN={metrics.tn}, FN={metrics.fn})"
        )

    _print_metrics(result.overall, "Overall")
    for task, metrics in sorted(result.per_task.items()):
        _print_metrics(metrics, f"Task: {task}")
    print("=" * 70)


def main() -> int:
    """Run the public HaluEval CLI and write metrics and review identities.

    Returns
    -------
    int
        Zero after every selected review and result-file write succeeds.
    """
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="HaluEval benchmark")
    parser.add_argument("max_samples", nargs="?", type=int, default=100)
    parser.add_argument("--no-nli", action="store_true")
    parser.add_argument("--model", default=None, help="Hugging Face model ID")
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = run_halueval_benchmark(
        use_nli=not args.no_nli,
        max_samples_per_task=args.max_samples,
        model_name=args.model,
        sample_offset=args.sample_offset,
        require_complete=args.require_complete,
    )
    _print_results(result)
    output_path = _CACHE_DIR / "halueval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "benchmark": "HaluEval",
                "overall": {
                    name: round(getattr(result.overall, name), 4)
                    for name in ("accuracy", "precision", "recall", "f1", "total")
                },
                "per_task": {
                    task: {
                        name: round(getattr(metrics, name), 4)
                        for name in ("accuracy", "precision", "recall", "f1", "total")
                    }
                    for task, metrics in result.per_task.items()
                },
                "evaluations": [asdict(row) for row in result.evaluations],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nResults saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
