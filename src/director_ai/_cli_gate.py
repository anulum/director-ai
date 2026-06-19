# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — CLI CI-gate command
"""``director-ai ci-gate`` — a CI quality gate over a labelled eval set.

Wraps :mod:`director_ai.ci_gate`: load a JSONL dataset of
``{prompt, response, expected}`` cases, score each with the configured scorer,
and exit non-zero when accuracy (or the optional catch-rate / false-halt-rate
thresholds) falls below the bar — so a CI job blocks a regression.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class _GateArgs:
    """Parsed ``ci-gate`` arguments."""

    dataset: str
    min_accuracy: float
    min_catch_rate: float | None
    max_false_halt: float | None
    profile: str | None
    output: str | None


def _cmd_ci_gate(args: list[str]) -> None:
    r"""Run the CI eval gate. Exits 1 on failure, 2 on a usage/data error.

    Usage::

        director-ai ci-gate --dataset cases.jsonl --min-accuracy 0.9 \\
            [--min-catch-rate 0.85] [--max-false-halt 0.1] \\
            [--profile medical] [--output gate.json]
    """
    opts = _parse_args(args)
    if opts is None:
        sys.exit(2)

    from director_ai.ci_gate import GateThresholds, load_cases, run_eval_gate
    from director_ai.core.config import DirectorConfig

    try:
        cases = load_cases(opts.dataset)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(2)

    cfg = (
        DirectorConfig.from_profile(opts.profile)
        if opts.profile
        else DirectorConfig.from_env()
    )
    scorer = cfg.build_scorer()

    report = run_eval_gate(
        cases,
        scorer,
        GateThresholds(
            min_accuracy=opts.min_accuracy,
            min_catch_rate=opts.min_catch_rate,
            max_false_halt_rate=opts.max_false_halt,
        ),
    )

    for line in report.summary_lines():
        print(line)

    if opts.output:
        with open(opts.output, "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2)
        print(f"  report written to {opts.output}")

    sys.exit(0 if report.passed else 1)


def _parse_args(args: list[str]) -> _GateArgs | None:
    """Parse ci-gate flags. Returns ``None`` (with a printed error) on misuse."""
    dataset: str | None = None
    profile: str | None = None
    output: str | None = None
    min_accuracy = 0.0
    thresholds: dict[str, float | None] = {
        "min_catch_rate": None,
        "max_false_halt": None,
    }
    float_flags = {
        "--min-catch-rate": "min_catch_rate",
        "--max-false-halt": "max_false_halt",
    }

    i = 0
    while i < len(args):
        flag = args[i]
        has_value = i + 1 < len(args)
        if flag == "--dataset" and has_value:
            dataset = args[i + 1]
        elif flag == "--profile" and has_value:
            profile = args[i + 1]
        elif flag == "--output" and has_value:
            output = args[i + 1]
        elif flag == "--min-accuracy" and has_value:
            parsed = _parse_unit_float(flag, args[i + 1])
            if parsed is None:
                return None
            min_accuracy = parsed
        elif flag in float_flags and has_value:
            parsed = _parse_unit_float(flag, args[i + 1])
            if parsed is None:
                return None
            thresholds[float_flags[flag]] = parsed
        else:
            print(f"Error: unknown or incomplete argument: {flag}")
            return None
        i += 2

    if not dataset:
        print(
            "Usage: director-ai ci-gate --dataset cases.jsonl "
            "--min-accuracy 0.9 [--min-catch-rate R] [--max-false-halt R] "
            "[--profile P] [--output gate.json]"
        )
        return None

    return _GateArgs(
        dataset=dataset,
        min_accuracy=min_accuracy,
        min_catch_rate=thresholds["min_catch_rate"],
        max_false_halt=thresholds["max_false_halt"],
        profile=profile,
        output=output,
    )


def _parse_unit_float(flag: str, raw: str) -> float | None:
    """Parse a 0-1 float flag value; print and return ``None`` on misuse."""
    try:
        value = float(raw)
    except ValueError:
        print(f"Error: {flag} expects a number, got {raw!r}")
        return None
    if not 0.0 <= value <= 1.0:
        print(f"Error: {flag} must be between 0 and 1, got {value}")
        return None
    return value
