# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — FEVER Dev Benchmark (Held-Out Fact Verification)

"""Evaluate NLI model on FEVER dev set (fact verification).

Training included FEVER train (pietrolesci/nli_fever train split).
The dev split is held-out. FEVER tests whether the model can verify
claims against evidence passages — directly relevant to RAG
hallucination detection.

Usage::

    python -m benchmarks.fever_eval --max-samples 500 --out fever_results.json
    python -m benchmarks.fever_eval --model training/output/deberta-v3-base-hallucination

The pytest smoke tests live in ``tests/test_fever_benchmark.py`` (moved there so
this module does not import pytest — a minimal remote runner without pytest was
breaking the ``benchmarks.*`` import chain, KIMI3/#66).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any

from benchmarks._common import (
    NLIMetrics,
    NLIPredictor,
    print_nli_metrics,
    save_results,
)
from benchmarks._provenance import stamp
from director_ai.core.model_revisions import DEFAULT_NLI_MODEL

logger = logging.getLogger("DirectorAI.Benchmark.FEVER")

_LABEL_MAP = {"entailment": 0, "neutral": 1, "contradiction": 2}


def _load_fever_dev() -> list[dict]:
    # Lazy import: `datasets` is a benchmark-only dependency absent from the core
    # CI test extras, so keep it out of the module import chain — moving the
    # smoke tests into tests/ made this module collectable by `pytest tests/`
    # (KIMI3/#66 class), which broke on a module-level datasets import.
    from datasets import load_dataset

    logger.info("Loading FEVER NLI dev split ...")
    ds = load_dataset("pietrolesci/nli_fever", split="dev")
    return list(ds)


def run_fever_benchmark(
    max_samples: int | None = None,
    model_name: str | None = None,
) -> NLIMetrics:
    predictor = NLIPredictor(model_name=model_name)
    num_labels = predictor.model.config.num_labels
    if num_labels != 3:
        raise ValueError(
            "FEVER dev is a 3-class NLI benchmark "
            "(entailment / neutral / contradiction); model "
            f"{predictor.model_name!r} exposes {num_labels} labels. Pass a "
            "3-class model via model_name (e.g. "
            "training/output/deberta-v3-base-hallucination) — the 2-class "
            "FactCG default cannot produce a valid FEVER verdict, so scoring "
            "it here would fabricate an accuracy that is not meaningful."
        )
    rows = _load_fever_dev()
    if max_samples:
        rows = rows[:max_samples]

    metrics = NLIMetrics()
    for row in rows:
        premise = row.get("premise", "")
        hypothesis = row.get("hypothesis", "")
        raw_label = row.get("label")

        if isinstance(raw_label, str):
            label = _LABEL_MAP.get(raw_label.lower())
        elif isinstance(raw_label, int):
            label = raw_label
        else:
            continue

        if label is None or not premise or not hypothesis:
            continue

        t0 = time.perf_counter()
        pred = predictor.predict(premise, hypothesis)
        metrics.inference_times.append(time.perf_counter() - t0)
        metrics.y_true.append(label)
        metrics.y_pred.append(pred)

    return metrics


def build_fever_artefact(
    metrics: NLIMetrics, *, git_sha: str | None = None, model_name: str | None = None
) -> dict[str, Any]:
    """Assemble a reproducible FEVER artefact: aggregates + per-sample rows.

    Carries one row per scored pair (true and predicted NLI label), the NLI
    model that produced them, and a source-commit provenance stamp, so the
    committed accuracy can be re-derived without re-running the model — the
    artefact-integrity contract the 2026-07-17 review asked for. Recording
    ``model`` keeps the number honest: FEVER dev must be scored by a 3-class
    model, distinct from the 2-class FactCG production default.
    """
    payload: dict[str, Any] = {
        "benchmark": "FEVER_dev",
        "model": model_name,
        **metrics.to_dict(),
    }
    payload["rows"] = [
        {"index": i, "label": int(true), "predicted": int(pred)}
        for i, (true, pred) in enumerate(
            zip(metrics.y_true, metrics.y_pred, strict=True)
        )
    ]
    stamp(payload, git_sha=git_sha)
    return payload


def main(argv: list[str] | None = None) -> int:
    """Run the FEVER dev benchmark and write a reproducible artefact."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="FEVER fact-verification benchmark")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--model", default=None, help="HuggingFace NLI model id/path")
    parser.add_argument(
        "--out",
        default=None,
        help="explicit artefact path; default keeps benchmarks/results/",
    )
    parser.add_argument(
        "--git-sha",
        default=None,
        help="git commit SHA to record in the artefact for provenance",
    )
    args = parser.parse_args(argv)

    resolved_model = args.model or os.environ.get(
        "DIRECTOR_NLI_MODEL", DEFAULT_NLI_MODEL
    )
    metrics = run_fever_benchmark(max_samples=args.max_samples, model_name=args.model)
    print_nli_metrics(metrics, "FEVER Dev")
    artefact = build_fever_artefact(
        metrics, git_sha=args.git_sha, model_name=resolved_model
    )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(artefact, handle, indent=2)
    else:
        save_results(artefact, "fever_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
