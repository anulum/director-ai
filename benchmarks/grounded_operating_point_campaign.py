# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — grounded-route operating-point campaign (HaluEval QA)

"""Calibrate the grounded composite gate on full HaluEval QA, post premise-fix.

The premise fix routed ``h_logical`` to score against retrieved evidence
instead of the raw question whenever a model-backed NLI and a knowledge
store are present. That changed the composite-coherence distribution of
every grounded review — and the module doctrine in
:mod:`director_ai.core.calibration.operating_points` requires a re-fit
whenever the premise composition changes. The historical 0.5 default was
calibrated on context-as-prompt scoring, a shape the grounded
question+store path no longer uses.

This campaign reproduces the exact production shape the red-team runs
exercised: a :class:`GroundTruthStore` holding the sample's knowledge,
``review(question, answer)`` on the shipped
:class:`~director_ai.core.scoring.scorer.CoherenceScorer`, one NLI model
loaded once and reused. Both HaluEval QA answers are scored per sample —
the right answer (a false halt when blocked) and the hallucinated answer
(an evasion when passed) — and every per-review row lands in the JSON
artefact so thresholds can be re-analysed offline without a GPU.

The sweep reuses ``matched_fpr_support_threshold`` from the production
calibration module, so reported operating points follow the same order
statistic deployments calibrate with.

Usage::

    python benchmarks/grounded_operating_point_campaign.py --out results.json
    python benchmarks/grounded_operating_point_campaign.py \
        --out smoke.json --max-samples 3          # CPU smoke
    python benchmarks/grounded_operating_point_campaign.py \
        --analyse benchmarks/results/grounded_operating_point_campaign.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

HALUEVAL_QA_URL = (
    "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
)
FACT_KEY = "campaign_fact"
REVIEW_THRESHOLD = 0.5
SWEEP_START = 0.30
SWEEP_STOP = 0.60
SWEEP_STEP = 0.01
CANDIDATE_THRESHOLDS = (0.40, 0.46, 0.50)
TARGET_FPRS = (0.010, 0.025, 0.045)
PROGRESS_EVERY = 200


def load_halueval_qa(url: str, max_samples: int | None) -> list[dict[str, str]]:
    """Download HaluEval QA pairs (JSON lines) and validate the fields."""
    import requests

    logger.info("Downloading HaluEval QA from %s", url)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    samples: list[dict[str, str]] = []
    for line in response.text.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        samples.append(
            {
                "knowledge": row["knowledge"],
                "question": row["question"],
                "right_answer": row["right_answer"],
                "hallucinated_answer": row["hallucinated_answer"],
            }
        )
        if max_samples is not None and len(samples) >= max_samples:
            break
    logger.info("Loaded %d QA samples", len(samples))
    return samples


def build_scorer(backend: str) -> tuple[Any, Any]:
    """Build one production scorer bound to one reusable knowledge store."""
    from director_ai.core import CoherenceScorer, GroundTruthStore

    store = GroundTruthStore()
    store.add(FACT_KEY, "campaign warm-up placeholder fact")
    scorer = CoherenceScorer(
        threshold=REVIEW_THRESHOLD,
        ground_truth_store=store,
        use_nli=True,
        scorer_backend=backend,
    )
    return scorer, store


def score_pairs(scorer: Any, store: Any, samples: list[dict[str, str]]) -> list[dict]:
    """Score right + hallucinated answers through the grounded review path.

    Re-adding under the same key keeps exactly one fact in the store, so
    retrieval always returns the current sample's knowledge; the rolling
    approval history is cleared per review so the 10k samples stay
    independent.
    """
    rows: list[dict] = []
    for index, sample in enumerate(samples):
        store.add(FACT_KEY, sample["knowledge"])
        for label in ("right", "hallucinated"):
            answer = sample[f"{label}_answer"]
            scorer.history.clear()
            t0 = time.perf_counter()
            approved, score = scorer.review(sample["question"], answer)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            degraded = bool(getattr(score, "degraded_mode", False))
            if degraded:
                logger.error(
                    "Scorer fell back to the keyword heuristic — calibration "
                    "on a degraded path is meaningless. Install the [nli] "
                    "stack and re-run."
                )
                sys.exit(2)
            rows.append(
                {
                    "sample": index,
                    "label": label,
                    "approved_at_default": bool(approved),
                    "coherence": round(float(score.score), 6),
                    "h_logical": round(float(getattr(score, "h_logical", 0.0)), 6),
                    "h_factual": round(float(getattr(score, "h_factual", 0.0)), 6),
                    "latency_ms": round(latency_ms, 1),
                }
            )
        if (index + 1) % PROGRESS_EVERY == 0:
            logger.info("Scored %d/%d samples", index + 1, len(samples))
    return rows


def _rates_at(threshold: float, good: list[float], bad: list[float]) -> dict:
    """False-halt / catch / balanced accuracy when halting below *threshold*."""
    false_halt = sum(1 for c in good if c < threshold) / len(good)
    catch = sum(1 for c in bad if c < threshold) / len(bad)
    balanced = ((1.0 - false_halt) + catch) / 2.0
    return {
        "threshold": round(threshold, 4),
        "false_halt_rate": round(false_halt, 6),
        "catch_rate": round(catch, 6),
        "balanced_accuracy": round(balanced, 6),
    }


def sweep(rows: list[dict]) -> dict:
    """Threshold sweep + matched-FPR operating points on the coherence scale."""
    from director_ai.core.calibration.operating_points import (
        matched_fpr_support_threshold,
    )

    good = [r["coherence"] for r in rows if r["label"] == "right"]
    bad = [r["coherence"] for r in rows if r["label"] == "hallucinated"]

    grid: list[dict] = []
    steps = int(round((SWEEP_STOP - SWEEP_START) / SWEEP_STEP)) + 1
    for i in range(steps):
        grid.append(_rates_at(SWEEP_START + i * SWEEP_STEP, good, bad))
    best = max(grid, key=lambda r: r["balanced_accuracy"])

    matched = []
    for target in TARGET_FPRS:
        threshold = matched_fpr_support_threshold(good, target)
        point = _rates_at(threshold, good, bad)
        point["target_fpr"] = target
        matched.append(point)

    return {
        "n_good": len(good),
        "n_bad": len(bad),
        "candidates": [_rates_at(t, good, bad) for t in CANDIDATE_THRESHOLDS],
        "best_balanced_accuracy": best,
        "matched_fpr_operating_points": matched,
        "sweep_grid": grid,
    }


def run_campaign(args: argparse.Namespace) -> dict:
    """Score the corpus and assemble the artefact payload."""
    samples = load_halueval_qa(args.data_url, args.max_samples)
    scorer, store = build_scorer(args.backend)
    started = time.time()
    rows = score_pairs(scorer, store, samples)
    return {
        "campaign": "grounded_operating_point",
        "dataset": "HaluEval-QA",
        "data_url": args.data_url,
        "git_sha": args.git_sha,
        "backend": args.backend,
        "nli_model": getattr(scorer, "nli_model_name", None)
        or getattr(getattr(scorer, "_nli", None), "model_name", None),
        "review_threshold_default": REVIEW_THRESHOLD,
        "n_samples": len(samples),
        "wall_seconds": round(time.time() - started, 1),
        "summary": sweep(rows),
        "rows": rows,
    }


def analyse_artefact(path: Path) -> dict:
    """Re-run the threshold analysis on a saved artefact (no GPU needed)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["summary"] = sweep(payload["rows"])
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="write the JSON artefact here")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--backend", default="deberta")
    parser.add_argument("--data-url", default=HALUEVAL_QA_URL)
    parser.add_argument(
        "--git-sha",
        default="",
        help="record the source commit in the artefact for reproducibility",
    )
    parser.add_argument(
        "--analyse",
        type=Path,
        default=None,
        help="re-analyse a saved artefact instead of scoring",
    )
    args = parser.parse_args(argv)

    if args.analyse is not None:
        payload = analyse_artefact(args.analyse)
    else:
        if not args.out:
            parser.error("--out is required when scoring")
        payload = run_campaign(args)

    summary = json.dumps(payload["summary"], indent=2)
    logger.info("Summary:\n%s", summary)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        logger.info("Artefact written to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
