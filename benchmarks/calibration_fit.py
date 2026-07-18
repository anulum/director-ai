# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Calibration Fit

"""Fit probability calibration maps from a per-sample operating-point artefact.

Reads the ``(coherence, label)`` rows of a grounded-operating-point campaign
artefact, fits an isotonic and a Platt calibrator, and emits a versioned
calibration artefact recording each map plus the Expected Calibration Error and
Brier score before and after calibration. The artefact is the reusable,
reproducible calibration evidence the 2026-07-17 red-team review asked for —
carrying per-sample rows so a reader can re-derive the ECE, and a source-commit
provenance stamp.

Usage::

    python -m benchmarks.calibration_fit \\
        --campaign benchmarks/results/grounded_operating_point_campaign.json \\
        --out calibration_fit.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks._common import save_results
from benchmarks._provenance import stamp
from director_ai.core.calibration.probability_calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    brier_score,
    expected_calibration_error,
)

SCHEMA_VERSION = "director-ai.calibration-fit.v1"

#: Campaign row labels that mark a grounded (correct) sample.
_GROUNDED_LABELS = frozenset({"right", "grounded", "correct"})


def _load_campaign_rows(path: str | Path) -> tuple[list[float], list[int]]:
    """Return ``(coherence_scores, grounded_labels)`` from a campaign artefact.

    ``label`` is the campaign's per-sample verdict — ``right`` for a grounded
    sample, anything else (``hallucinated``/``wrong``) for a hallucination. The
    coherence score is the predicted probability of being grounded.
    """
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"artefact {path} carries no per-sample rows")
    scores: list[float] = []
    labels: list[int] = []
    for row in rows:
        scores.append(float(row["coherence"]))
        labels.append(1 if str(row["label"]).lower() in _GROUNDED_LABELS else 0)
    return scores, labels


def _metric_block(
    scores: list[float], labels: list[int], *, n_bins: int
) -> dict[str, float]:
    """ECE and Brier of *scores* against *labels*."""
    return {
        "ece": round(expected_calibration_error(scores, labels, n_bins=n_bins), 6),
        "brier": round(brier_score(scores, labels), 6),
    }


def fit_calibration(
    campaign_path: str | Path, *, n_bins: int = 10, git_sha: str | None = None
) -> dict[str, Any]:
    """Fit both calibrators and assemble the reproducible calibration artefact."""
    scores, labels = _load_campaign_rows(campaign_path)

    isotonic = IsotonicCalibrator.fit(scores, labels)
    platt = PlattCalibrator.fit(scores, labels)
    iso_scores = isotonic.transform(scores)
    platt_scores = platt.transform(scores)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "calibration_fit",
        "source_campaign": str(campaign_path),
        "n_samples": len(scores),
        "n_bins": n_bins,
        "raw": _metric_block(scores, labels, n_bins=n_bins),
        "isotonic": {
            **_metric_block(iso_scores, labels, n_bins=n_bins),
            "x_thresholds": list(isotonic.x_thresholds),
            "y_values": list(isotonic.y_values),
        },
        "platt": {
            **_metric_block(platt_scores, labels, n_bins=n_bins),
            "a": platt.a,
            "b": platt.b,
        },
        # Per-sample rows so the aggregate ECE is independently re-derivable.
        "rows": [
            {
                "coherence": round(raw, 6),
                "grounded": label,
                "isotonic": round(iso, 6),
                "platt": round(pl, 6),
            }
            for raw, label, iso, pl in zip(
                scores, labels, iso_scores, platt_scores, strict=True
            )
        ],
    }
    stamp(payload, git_sha=git_sha)
    return payload


def main(argv: list[str] | None = None) -> int:
    """Fit calibration from the campaign artefact and write the result."""
    parser = argparse.ArgumentParser(description="Fit probability calibration maps")
    parser.add_argument(
        "--campaign",
        default="benchmarks/results/grounded_operating_point_campaign.json",
        help="per-sample operating-point campaign artefact",
    )
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument(
        "--out",
        default=None,
        help="explicit artefact path; default keeps benchmarks/results/",
    )
    parser.add_argument(
        "--git-sha",
        default=None,
        help="git commit SHA to record for provenance",
    )
    args = parser.parse_args(argv)

    payload = fit_calibration(args.campaign, n_bins=args.n_bins, git_sha=args.git_sha)
    raw, iso, platt = (
        payload["raw"]["ece"],
        payload["isotonic"]["ece"],
        payload["platt"]["ece"],
    )
    print(
        f"ECE  raw {raw:.4f} -> isotonic {iso:.4f} / platt {platt:.4f}  "
        f"(n={payload['n_samples']})"
    )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    else:
        save_results(payload, "calibration_fit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
