# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Conformal routing evidence packet

"""Generate local evidence for conformal uncertainty routing.

The packet checks the production-relevant R10 primitives without model
downloads:

* split-conformal intervals meet the requested coverage on deterministic
  held-out calibration cases;
* low-risk outputs are allowed only when the upper risk bound is low;
* uncertain outputs route to human review or a stronger model path;
* high-risk outputs are rejected only when the lower risk bound is high.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks._common import save_results
from benchmarks._provenance import resolve_git_sha
from director_ai.core.calibration.conformal import (
    ConformalPredictor,
    ConformalRoutingPolicy,
    PredictionInterval,
)


def _calibration_samples(sample_count: int) -> tuple[list[float], list[bool]]:
    if sample_count < 2:
        raise ValueError("sample_count must be >= 2")

    half = sample_count // 2
    supported = [0.99 - (idx % 5) * 0.002 for idx in range(half)]
    contradicted = [0.01 + (idx % 5) * 0.002 for idx in range(sample_count - half)]
    return supported + contradicted, [False] * len(supported) + [True] * len(
        contradicted
    )


def _actual_risk(label: bool) -> float:
    return 1.0 if label else 0.0


def _is_covered(interval: PredictionInterval, label: bool) -> bool:
    actual = _actual_risk(label)
    return interval.lower <= actual <= interval.upper


def run_coverage_probe(
    *,
    coverage: float = 0.95,
    calibration_samples: int = 80,
    validation_samples: int = 40,
    min_samples: int = 30,
) -> dict[str, Any]:
    """Return deterministic split-conformal coverage evidence."""
    if validation_samples < 2:
        raise ValueError("validation_samples must be >= 2")

    scores, labels = _calibration_samples(calibration_samples)
    predictor = ConformalPredictor(coverage=coverage, min_samples=min_samples)
    predictor.calibrate(scores, labels)

    validation_scores, validation_labels = _calibration_samples(validation_samples)
    intervals = [predictor.predict(score) for score in validation_scores]
    covered = [
        _is_covered(interval, label)
        for interval, label in zip(intervals, validation_labels, strict=True)
    ]
    widths = [interval.upper - interval.lower for interval in intervals]
    empirical_coverage = sum(covered) / len(covered)
    reliable = all(interval.is_reliable for interval in intervals)

    return {
        "name": "coverage_calibration",
        "target_coverage": coverage,
        "empirical_coverage": round(empirical_coverage, 4),
        "calibration_samples": calibration_samples,
        "validation_samples": validation_samples,
        "min_samples": min_samples,
        "reliable": reliable,
        "mean_interval_width": round(sum(widths) / len(widths), 4),
        "coverage_failures": len(covered) - sum(covered),
        "passed": bool(empirical_coverage >= coverage and reliable),
    }


def run_routing_probe(
    *,
    coverage: float = 0.95,
    calibration_samples: int = 80,
    min_samples: int = 30,
) -> dict[str, Any]:
    """Return evidence that conformal intervals drive all routing outcomes."""
    scores, labels = _calibration_samples(calibration_samples)
    predictor = ConformalPredictor(coverage=coverage, min_samples=min_samples)
    predictor.calibrate(scores, labels)
    policy = ConformalRoutingPolicy(
        allow_max_risk=0.05,
        escalate_min_risk=0.20,
        reject_min_risk=0.70,
        min_samples=min_samples,
    )

    uncalibrated = ConformalPredictor(coverage=coverage, min_samples=min_samples)
    cases = [
        ("low_risk", predictor, 0.999, "allow"),
        ("ambiguous_mid", predictor, 0.900, "human_review"),
        ("uncertain_high", predictor, 0.700, "escalate"),
        ("high_risk", predictor, 0.020, "reject"),
        ("uncalibrated", uncalibrated, 0.999, "human_review"),
    ]
    decisions = []
    for name, case_predictor, score, expected_action in cases:
        decision = case_predictor.route(score, policy)
        decisions.append(
            {
                "case": name,
                "score": score,
                "expected_action": expected_action,
                "action": decision.action,
                "reason": decision.reason,
                "route_to": decision.route_to,
                "risk_lower": round(decision.risk_lower, 4),
                "risk_upper": round(decision.risk_upper, 4),
                "coverage": decision.coverage,
                "calibration_size": decision.calibration_size,
                "is_reliable": decision.is_reliable,
                "matched": decision.action == expected_action,
            }
        )

    action_counts: dict[str, int] = {}
    for decision in decisions:
        action = str(decision["action"])
        action_counts[action] = action_counts.get(action, 0) + 1

    return {
        "name": "routing_decisions",
        "policy": {
            "allow_max_risk": policy.allow_max_risk,
            "escalate_min_risk": policy.escalate_min_risk,
            "reject_min_risk": policy.reject_min_risk,
            "min_samples": policy.min_samples,
        },
        "action_counts": action_counts,
        "decisions": decisions,
        "passed": all(bool(decision["matched"]) for decision in decisions),
    }


def run_conformal_routing_evidence(
    *,
    coverage: float = 0.95,
    calibration_samples: int = 80,
    validation_samples: int = 40,
    min_samples: int = 30,
) -> dict[str, Any]:
    """Return the complete local R10 conformal-routing evidence packet."""
    coverage_probe = run_coverage_probe(
        coverage=coverage,
        calibration_samples=calibration_samples,
        validation_samples=validation_samples,
        min_samples=min_samples,
    )
    routing_probe = run_routing_probe(
        coverage=coverage,
        calibration_samples=calibration_samples,
        min_samples=min_samples,
    )
    passed = bool(coverage_probe["passed"] and routing_probe["passed"])
    return {
        "schema_version": "director-ai.conformal-routing-evidence.v1",
        "benchmark": "conformal_routing_evidence",
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": resolve_git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "acceptance": {
            "passed": passed,
            "checks": {
                "coverage_calibration": bool(coverage_probe["passed"]),
                "routing_decisions": bool(routing_probe["passed"]),
            },
            "limits": {
                "local_only": True,
                "external_operator_signoff_included": False,
                "representative_domain_dataset_included": False,
            },
        },
        "probes": {
            "coverage_calibration": coverage_probe,
            "routing_decisions": routing_probe,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate Director-AI conformal routing evidence packet.",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.95,
        help="Target conformal coverage.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=80,
        help="Number of deterministic calibration samples.",
    )
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=40,
        help="Number of deterministic validation samples.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=30,
        help="Minimum samples required before routing intervals are reliable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to benchmarks/results/.",
    )
    args = parser.parse_args(argv)

    payload = run_conformal_routing_evidence(
        coverage=args.coverage,
        calibration_samples=args.calibration_samples,
        validation_samples=args.validation_samples,
        min_samples=args.min_samples,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Results saved to {args.output}")
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        save_results(payload, f"conformal_routing_evidence_{stamp}.json")
    print(json.dumps(payload["acceptance"], indent=2))
    return 0 if payload["acceptance"]["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
