# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Public hallucination detection competitor benchmarks
"""Run public hallucination detection classifiers on AggreFact 29 K.

Currently supports:

- ``vectara/hallucination_evaluation_model`` (HHEM-2.1, 184 M, classifier head)
- ``lytang/MiniCheck-Roberta-Large`` (1.4 GB, T5-style entailment classifier)

These are NLI / entailment classifiers, not generative judges. Each model
has its own input format. We adapt the prompt per backend so that the
output is comparable to our other AggreFact JSONs (same schema).

Output schema matches benchmarks/gemma_aggrefact_eval.py so the
sentinel_judge_analyser.py can ensemble them with the LLM judges.

Usage::

    HIP_VISIBLE_DEVICES=4 python benchmarks/competitor_aggrefact.py \\
        --model vectara/hallucination_evaluation_model \\
        --output benchmarks/results/competitor_hhem_21.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, TypedDict, cast

from _judge_common import compute_balanced_accuracy as balanced_accuracy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class AggreFactRow(TypedDict):
    """AggreFact row consumed by the competitor benchmark harness."""

    doc: str
    claim: str
    label: int
    dataset: str


class ScoringBackend(Protocol):
    """Classifier interface shared by model-backed and replay-backed scoring."""

    def score(self, premise: str, hypothesis: str) -> float:
        """Return the probability that ``hypothesis`` is supported by ``premise``.

        Parameters
        ----------
        premise:
            Source context or document text.
        hypothesis:
            Claim being checked against the premise.

        Returns
        -------
        float
            Support probability in the inclusive range ``[0, 1]``.
        """


# ── HHEM-2.1 (Vectara) ───────────────────────────────────────────────────


class HHEMBackend:
    """vectara/hallucination_evaluation_model — sequence classifier."""

    def __init__(self, model_id: str, max_length: int) -> None:
        """Load the HHEM sequence-classification backend.

        Parameters
        ----------
        model_id:
            Hugging Face model identifier for the HHEM classifier.
        max_length:
            Maximum tokenised sequence length.
        """
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch: Any = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, torch_dtype=torch.float32, trust_remote_code=True
        )
        self.model.to("cuda")
        self.model.eval()
        self.max_length = max_length

    def score(self, premise: str, hypothesis: str) -> float:
        """Return the HHEM support probability for a premise/claim pair."""
        # HHEM expects ``premise<sep>hypothesis`` and returns P(consistent).
        text = f"{premise}<eos>{hypothesis}"
        with self.torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            ).to(self.model.device)
            logits = self.model(**inputs).logits
            return float(self.torch.sigmoid(logits[0, 0]).item())


# ── MiniCheck-Roberta-Large ──────────────────────────────────────────────


class MiniCheckBackend:
    """lytang/MiniCheck-Roberta-Large — Roberta entailment classifier."""

    def __init__(self, model_id: str, max_length: int) -> None:
        """Load the MiniCheck sequence-classification backend.

        Parameters
        ----------
        model_id:
            Hugging Face model identifier for the MiniCheck classifier.
        max_length:
            Maximum tokenised sequence length.
        """
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch: Any = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to("cuda")
        self.model.eval()
        self.max_length = max_length

    def score(self, premise: str, hypothesis: str) -> float:
        """Return the MiniCheck entailment probability for a premise/claim pair."""
        with self.torch.no_grad():
            inputs = self.tokenizer(
                premise,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            ).to(self.model.device)
            logits = self.model(**inputs).logits
            probs = self.torch.softmax(logits, dim=-1)
            # Roberta NLI label 0 = entailment / supported
            return float(probs[0, 0].item())


BACKENDS = {
    "vectara/hallucination_evaluation_model": HHEMBackend,
    "lytang/MiniCheck-Roberta-Large": MiniCheckBackend,
}


def _as_aggrefact_row(row: Mapping[str, object], *, row_number: int) -> AggreFactRow:
    """Validate and normalise one AggreFact-style row."""
    missing = [key for key in ("doc", "claim", "label", "dataset") if key not in row]
    if missing:
        raise ValueError(
            f"row {row_number} missing required AggreFact field(s): "
            f"{', '.join(missing)}"
        )

    label = int(row["label"])
    if label not in (0, 1):
        raise ValueError(f"row {row_number} label must be 0 or 1, got {label}")

    return {
        "doc": str(row["doc"]),
        "claim": str(row["claim"]),
        "label": label,
        "dataset": str(row["dataset"]),
    }


def load_jsonl_dataset(path: Path) -> list[AggreFactRow]:
    """Load a local AggreFact-compatible JSONL dataset.

    Parameters
    ----------
    path:
        JSONL file containing one object per row with ``doc``, ``claim``,
        ``label``, and ``dataset`` keys.

    Returns
    -------
    list[AggreFactRow]
        Validated rows in file order.

    Raises
    ------
    ValueError
        If a row is not a JSON object or fails schema validation.
    """
    rows: list[AggreFactRow] = []
    for row_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"row {row_number} must be a JSON object")
        rows.append(
            _as_aggrefact_row(cast(Mapping[str, object], raw), row_number=row_number)
        )
    return rows


def load_remote_aggrefact_dataset(max_samples: int | None) -> list[AggreFactRow]:
    """Load the public AggreFact test split through ``datasets``.

    Parameters
    ----------
    max_samples:
        Optional upper bound on rows retained from the split.

    Returns
    -------
    list[AggreFactRow]
        Validated rows in dataset order.
    """
    from datasets import load_dataset

    loaded = load_dataset("lytang/LLM-AggreFact", split="test")
    if max_samples:
        loaded = loaded.select(range(min(max_samples, len(loaded))))
    return [
        _as_aggrefact_row(cast(Mapping[str, object], sample), row_number=index)
        for index, sample in enumerate(loaded, 1)
    ]


def load_precomputed_scores(path: Path) -> list[float]:
    """Load replay scores from a JSON array.

    Parameters
    ----------
    path:
        JSON file containing a list of numeric probabilities.

    Returns
    -------
    list[float]
        Scores parsed from the file.

    Raises
    ------
    ValueError
        If the file is not a numeric JSON array or a score is outside
        ``[0, 1]``.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("precomputed scores must be a JSON array")

    scores: list[float] = []
    for index, value in enumerate(raw, 1):
        score = float(value)
        if not 0.0 <= score <= 1.0:
            raise ValueError(
                f"precomputed score {index} must be in [0, 1], got {score}"
            )
        scores.append(score)
    return scores


def _iter_scores(
    *,
    rows: Sequence[AggreFactRow],
    backend: ScoringBackend | None,
    precomputed_scores: Sequence[float] | None,
) -> Iterable[float]:
    """Yield scores from either a backend or a validated replay vector."""
    if precomputed_scores is not None:
        if len(precomputed_scores) != len(rows):
            raise ValueError(
                f"precomputed scores length {len(precomputed_scores)} does not "
                f"match dataset rows {len(rows)}"
            )
        yield from precomputed_scores
        return

    if backend is None:
        raise ValueError("a scoring backend is required when replay scores are absent")

    for row in rows:
        yield backend.score(row["doc"], row["claim"])


def run_competitor_benchmark(
    *,
    model: str,
    rows: Sequence[AggreFactRow],
    backend: ScoringBackend | None,
    precomputed_scores: Sequence[float] | None,
    threshold: float,
    log_every: int,
) -> dict[str, object]:
    """Score AggreFact rows and return the benchmark result schema.

    Parameters
    ----------
    model:
        Model identifier recorded in the result artifact.
    rows:
        Validated AggreFact rows.
    backend:
        Model-backed scorer used when ``precomputed_scores`` is absent.
    precomputed_scores:
        Optional replay scores aligned one-to-one with ``rows``.
    threshold:
        Probability threshold for predicting support.
    log_every:
        Emit progress logs after this many rows. Values less than one disable
        progress logging.

    Returns
    -------
    dict[str, object]
        JSON-serialisable benchmark result following the historical schema.
    """
    scores: list[float] = []
    preds: list[int] = []
    labels: list[int] = []
    datasets_list: list[str] = []
    latencies: list[float] = []
    t_start = time.time()

    for i, (sample, score_input) in enumerate(
        zip(
            rows,
            _iter_scores(
                rows=rows,
                backend=backend,
                precomputed_scores=precomputed_scores,
            ),
            strict=True,
        )
    ):
        t0 = time.time()
        try:
            score = float(score_input)
            pred = 1 if score >= threshold else 0
        except Exception as e:
            logger.warning("Sample %d failed: %s", i, e)
            score, pred = -1.0, -1
        latencies.append(time.time() - t0)

        scores.append(score)
        preds.append(pred)
        labels.append(sample["label"])
        datasets_list.append(sample["dataset"])

        if log_every > 0 and (i + 1) % log_every == 0:
            elapsed = time.time() - t_start
            ba = balanced_accuracy(preds, labels)
            eta = (len(rows) - i - 1) * elapsed / (i + 1) / 60
            logger.info(
                "[%d/%d] BA=%.4f %.0fms/sample ETA=%.1fmin",
                i + 1,
                len(rows),
                ba,
                1000 * elapsed / (i + 1),
                eta,
            )

    by_ds: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for p_, l_, d_ in zip(preds, labels, datasets_list, strict=True):
        by_ds[d_][0].append(p_)
        by_ds[d_][1].append(l_)
    per_ds = {
        d: {"samples": len(l_), "balanced_accuracy": balanced_accuracy(p_, l_)}
        for d, (p_, l_) in by_ds.items()
    }

    total = time.time() - t_start
    backend_name = (
        "precomputed-score-replay"
        if precomputed_scores is not None
        else "transformers-classifier"
    )
    return {
        "model": model,
        "backend": backend_name,
        "samples": len(rows),
        "global_balanced_accuracy": balanced_accuracy(preds, labels),
        "per_dataset": per_ds,
        "scores": scores,
        "predictions": preds,
        "labels": labels,
        "datasets_per_sample": datasets_list,
        "threshold": threshold,
        "unknown_predictions": sum(1 for p in preds if p < 0),
        "total_time_seconds": total,
        "p50_latency_ms": 1000 * sorted(latencies)[len(latencies) // 2]
        if latencies
        else 0,
        "p99_latency_ms": (
            1000 * sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Parse CLI arguments and run the competitor AggreFact benchmark."""
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(BACKENDS.keys()))
    p.add_argument(
        "--input-jsonl",
        type=Path,
        default=None,
        help="Local AggreFact-compatible JSONL input; defaults to public dataset",
    )
    p.add_argument(
        "--precomputed-scores",
        type=Path,
        default=None,
        help="JSON array of support probabilities aligned to the selected rows",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--output", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--log-every", type=int, default=500)
    args = p.parse_args(argv)

    try:
        rows = (
            load_jsonl_dataset(args.input_jsonl)
            if args.input_jsonl is not None
            else load_remote_aggrefact_dataset(args.max_samples)
        )
        if args.max_samples is not None and args.input_jsonl is not None:
            rows = rows[: args.max_samples]
        logger.info("Samples: %d", len(rows))

        precomputed_scores = (
            load_precomputed_scores(args.precomputed_scores)
            if args.precomputed_scores is not None
            else None
        )
        if precomputed_scores is not None and args.max_samples is not None:
            precomputed_scores = precomputed_scores[: len(rows)]

        if precomputed_scores is None:
            logger.info("Loading: %s", args.model)
            backend: ScoringBackend | None = BACKENDS[args.model](
                args.model, args.max_length
            )
            logger.info("Loaded")
        else:
            logger.info("Using precomputed scores from %s", args.precomputed_scores)
            backend = None

        results = run_competitor_benchmark(
            model=args.model,
            rows=rows,
            backend=backend,
            precomputed_scores=precomputed_scores,
            threshold=args.threshold,
            log_every=args.log_every,
        )
    except ValueError as exc:
        p.error(str(exc))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    logger.info("=" * 60)
    logger.info("Global BA: %.4f", cast(float, results["global_balanced_accuracy"]))
    logger.info("Time:      %.1fmin", cast(float, results["total_time_seconds"]) / 60)
    logger.info("=" * 60)
    per_dataset = cast(dict[str, dict[str, object]], results["per_dataset"])
    for n, m in sorted(per_dataset.items()):
        logger.info("  %-20s %5d  %.4f", n, m["samples"], m["balanced_accuracy"])


if __name__ == "__main__":
    main()
