#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Competitor Benchmark Suite
"""Pluggable harness for running commercial hallucination-detection APIs
against the AggreFact 29 K benchmark.

The harness defines a tiny ``Backend`` protocol — one ``score(premise,
hypothesis) -> float`` call per sample — plus a ``run_suite()`` driver
that loads `lytang/LLM-AggreFact` (or an injected dataset for tests),
times each call, and writes a v2-schema-compatible JSON.

**No backend currently has a real implementation.** Each commercial
backend ``score()`` raises ``NotImplementedError`` so the script cannot
silently emit fabricated metrics. Implementations land one at a time
once the corresponding API key is provisioned through the credentials
vault. The ``MockBackend`` is used by the test suite only.

Usage::

    python -m benchmarks.competitor_aggrefact_suite --backend mock \\
           --max-samples 100 --output /tmp/mock.json

Real-API runs are gated on credential availability and explicit CEO
authorisation per ``feedback_cloud_explicit_approval``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def balanced_accuracy(preds: list[int], labels: list[int]) -> float:
    """Two-class balanced accuracy with -1 (unknown) prediction filtering.

    Returns 0.0 when both classes have zero samples after filtering.
    Falls back to single-class accuracy when only one class remains
    (the AggreFact subsets are unbalanced; this is acceptable for the
    progress logger but downstream consumers should always look at the
    full per-dataset breakdown).
    """
    pos = neg = tp = tn = 0
    for p, lab in zip(preds, labels, strict=True):
        if p < 0:
            continue
        if lab == 1:
            pos += 1
            if p == 1:
                tp += 1
        else:
            neg += 1
            if p == 0:
                tn += 1
    if pos == 0 and neg == 0:
        return 0.0
    if pos == 0:
        return tn / neg
    if neg == 0:
        return tp / pos
    return (tp / pos + tn / neg) / 2


class BaseBackend:
    """Abstract scoring backend.

    Subclasses must override ``score()``. The contract is:

    * Returns a float in [0, 1] where 1 means *fully supported* and 0
      means *contradicted*.
    * May raise on transient errors — the driver catches and records
      a -1 prediction for the sample.
    """

    name: str = "base"

    def score(self, premise: str, hypothesis: str) -> float:
        raise NotImplementedError(
            f"{self.__class__.__name__}.score() is not yet implemented. "
            "Real API calls require credential provisioning and explicit "
            "CEO sign-off; see GEMINI_RULES.md and "
            "feedback_cloud_explicit_approval."
        )


# ── Lakera Guard ────────────────────────────────────────────────────────


class LakeraBackend(BaseBackend):
    """Lakera Guard API — https://api.lakera.ai/v1/guard.

    Note: Lakera focuses on injection detection and content safety, not
    hallucination grounding. Including the backend here is a placeholder
    for a future hallucination check; it will currently only refuse to
    run.
    """

    name = "lakera"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("LAKERA_API_KEY")

    def score(self, premise: str, hypothesis: str) -> float:
        raise NotImplementedError(
            "LakeraBackend.score() is not yet implemented. The Lakera "
            "API is primarily an injection guard and currently has no "
            "documented hallucination-grounding endpoint compatible "
            "with the AggreFact (premise, hypothesis) interface."
        )


# ── Galileo Luna ─────────────────────────────────────────────────────────


class GalileoBackend(BaseBackend):
    """Galileo Luna SLM API."""

    name = "galileo"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("GALILEO_API_KEY")

    def score(self, premise: str, hypothesis: str) -> float:
        raise NotImplementedError(
            "GalileoBackend.score() is not yet implemented. Provision a "
            "Galileo API key via the credentials vault before enabling."
        )


# ── Azure AI Content Safety ──────────────────────────────────────────────


class AzureBackend(BaseBackend):
    """Azure AI Content Safety — Groundedness Detection."""

    name = "azure"

    def __init__(
        self, endpoint: str | None = None, key: str | None = None
    ) -> None:
        self.endpoint = endpoint or os.getenv("AZURE_SAFETY_ENDPOINT")
        self.key = key or os.getenv("AZURE_SAFETY_KEY")

    def score(self, premise: str, hypothesis: str) -> float:
        raise NotImplementedError(
            "AzureBackend.score() is not yet implemented. Azure "
            "Groundedness Detection has rate limits and cost controls "
            "that need CEO sign-off before any 29 K-sample run."
        )


# ── Guardrails AI ────────────────────────────────────────────────────────


class GuardrailsBackend(BaseBackend):
    """Guardrails AI Validator."""

    name = "guardrails"

    def __init__(self) -> None:
        pass

    def score(self, premise: str, hypothesis: str) -> float:
        raise NotImplementedError(
            "GuardrailsBackend.score() is not yet implemented. The "
            "guardrails-ai package needs to be installed and a "
            "Validator chain configured before this backend can run."
        )


# ── Mock Backend ─────────────────────────────────────────────────────────


class MockBackend(BaseBackend):
    """Deterministic backend for unit tests.

    Always returns ``fixed_score`` regardless of input. Tests use it to
    drive the ``run_suite()`` plumbing without touching any real API or
    HuggingFace dataset.
    """

    name = "mock"

    def __init__(self, fixed_score: float = 0.8) -> None:
        if not 0.0 <= fixed_score <= 1.0:
            raise ValueError(
                f"fixed_score must be in [0, 1], got {fixed_score}"
            )
        self.fixed_score = fixed_score

    def score(self, premise: str, hypothesis: str) -> float:
        return self.fixed_score


BACKENDS: dict[str, type[BaseBackend]] = {
    "lakera": LakeraBackend,
    "galileo": GalileoBackend,
    "azure": AzureBackend,
    "guardrails": GuardrailsBackend,
    "mock": MockBackend,
}


def run_suite(
    backend: str | BaseBackend,
    dataset: Iterable[dict[str, Any]] | None = None,
    max_samples: int | None = None,
    threshold: float = 0.5,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Score every sample in ``dataset`` with ``backend``.

    Parameters
    ----------
    backend
        Either a key from ``BACKENDS`` or an already-instantiated
        ``BaseBackend``. Tests typically pass an instance to inject a
        configured ``MockBackend``.
    dataset
        Iterable of dicts with ``doc``, ``claim``, ``label``,
        ``dataset`` keys. ``None`` loads the gated
        ``lytang/LLM-AggreFact`` test split via ``datasets`` (requires
        ``HF_TOKEN``).
    max_samples
        Truncate the iterable to the first N samples (post-shuffle by
        the underlying dataset). Used for smoke tests.
    threshold
        Score ≥ threshold ⇒ predict 1 (supported), else 0.
    output
        Optional JSON path. The schema matches v2 written by
        ``benchmarks.aggrefact_eval.score_and_save()``.

    Returns
    -------
    dict
        Result payload (same structure as the JSON file).
    """
    if isinstance(backend, BaseBackend):
        backend_obj = backend
        backend_name = backend.name
    else:
        backend_name = backend
        if backend_name not in BACKENDS:
            raise ValueError(
                f"Unknown backend '{backend_name}'. "
                f"Available: {sorted(BACKENDS)}"
            )
        backend_obj = BACKENDS[backend_name]()

    logger.info("Initialising backend: %s", backend_name)

    if dataset is None:
        from datasets import load_dataset

        logger.info("Loading AggreFact dataset from HuggingFace…")
        ds: Any = load_dataset("lytang/LLM-AggreFact", split="test")
    else:
        ds = dataset

    if max_samples is not None:
        if hasattr(ds, "select"):
            ds = ds.select(range(min(max_samples, len(ds))))
        else:
            ds = list(ds)[:max_samples]

    scores: list[float] = []
    preds: list[int] = []
    labels: list[int] = []
    datasets_list: list[str] = []
    latencies: list[float] = []
    t_start = time.time()

    for i, sample in enumerate(ds):
        t0 = time.time()
        try:
            score = backend_obj.score(sample["doc"], sample["claim"])
            pred = 1 if score >= threshold else 0
        except NotImplementedError:
            # Stub backend — never silently fall through, raise so the
            # caller knows the suite produced no real numbers.
            raise
        except Exception as exc:  # noqa: BLE001 — driver catches all
            logger.warning("Sample %d failed: %s", i, exc)
            score, pred = -1.0, -1

        latencies.append(time.time() - t0)
        scores.append(score)
        preds.append(pred)
        labels.append(int(sample["label"]))
        datasets_list.append(sample.get("dataset", "unknown"))

        if (i + 1) % 500 == 0:
            running_ba = balanced_accuracy(preds, labels)
            logger.info("[%d] BA=%.4f", i + 1, running_ba)

    results: dict[str, Any] = {
        "schema_version": 2,
        "model": backend_name,
        "backend": "competitor-suite",
        "samples": len(preds),
        "global_balanced_accuracy": balanced_accuracy(preds, labels),
        "scores": scores,
        "predictions": preds,
        "labels": labels,
        "datasets_per_sample": datasets_list,
        "latencies_per_sample": latencies,
        "total_time_seconds": time.time() - t_start,
    }

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", out_path)

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True, choices=list(BACKENDS))
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    run_suite(
        args.backend,
        None,
        args.max_samples,
        args.threshold,
        args.output,
    )
