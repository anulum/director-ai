#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - PINT seed smoke runner
"""Run the local PINT seed fixture as a non-public detector smoke check."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tomllib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from director_ai.core.safety.sanitizer import InputSanitizer

_VALIDATOR_PATH = (
    Path(__file__).resolve().parent / "validate_pint_replication_packet.py"
)
_VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "validate_pint_replication_packet",
    _VALIDATOR_PATH,
)
assert _VALIDATOR_SPEC is not None
assert _VALIDATOR_SPEC.loader is not None
_VALIDATOR = importlib.util.module_from_spec(_VALIDATOR_SPEC)
sys.modules[_VALIDATOR_SPEC.name] = _VALIDATOR
_VALIDATOR_SPEC.loader.exec_module(_VALIDATOR)

DEFAULT_CASES: Path = _VALIDATOR.DEFAULT_CASES
PACKET: Path = _VALIDATOR.PACKET
validate_pint_replication_packet = _VALIDATOR.validate_pint_replication_packet

DEFAULT_OUTPUT = Path("benchmarks/results/pint_seed_smoke.json")


class BooleanDetector(Protocol):
    def score(self, text: str) -> Any:
        """Return an object with blocked/suspicion_score/pattern fields."""


@dataclass(frozen=True)
class PintSeedCase:
    case_id: str
    category: str
    language: str
    text: str
    expected_injection: bool


def _load_packet_id(root: Path) -> str:
    packet = tomllib.loads((root / PACKET).read_text(encoding="utf-8"))
    packet_id = packet.get("packet_id")
    if not isinstance(packet_id, str) or not packet_id.strip():
        raise ValueError(f"{PACKET}: packet_id must be a non-empty string")
    return packet_id


def load_seed_cases(root: Path) -> tuple[list[str], tuple[PintSeedCase, ...]]:
    errors = validate_pint_replication_packet(root)
    if errors:
        return errors, ()

    cases: list[PintSeedCase] = []
    for line in (root / DEFAULT_CASES).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        cases.append(
            PintSeedCase(
                case_id=row["id"],
                category=row["category"],
                language=row["language"],
                text=row["input"],
                expected_injection=row["expected_injection"],
            )
        )
    return [], tuple(cases)


def evaluate_seed_cases(
    cases: Sequence[PintSeedCase],
    *,
    detector: BooleanDetector,
) -> dict[str, Any]:
    per_case: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    category_totals: Counter[str] = Counter()
    category_correct: Counter[str] = Counter()
    language_totals: Counter[str] = Counter()
    language_correct: Counter[str] = Counter()

    for case in cases:
        score = detector.score(case.text)
        prediction = bool(getattr(score, "blocked", False))
        expected = case.expected_injection
        correct = prediction == expected
        category_totals[case.category] += 1
        language_totals[case.language] += 1
        if correct:
            counts["correct"] += 1
            category_correct[case.category] += 1
            language_correct[case.language] += 1
        if prediction and expected:
            counts["tp"] += 1
        elif prediction and not expected:
            counts["fp"] += 1
        elif not prediction and expected:
            counts["fn"] += 1
        else:
            counts["tn"] += 1

        per_case.append(
            {
                "id": case.case_id,
                "category": case.category,
                "language": case.language,
                "expected_injection": expected,
                "prediction": prediction,
                "correct": correct,
                "suspicion_score": float(getattr(score, "suspicion_score", 0.0)),
                "pattern": str(getattr(score, "pattern", "")),
                "matches": list(getattr(score, "matches", ())),
            }
        )

    total = len(cases)
    correct_count = counts["correct"]
    precision_denominator = counts["tp"] + counts["fp"]
    recall_denominator = counts["tp"] + counts["fn"]
    return {
        "schema_version": "1.0.0",
        "result_type": "pint_seed_smoke",
        "public_score_claim": False,
        "official_pint_score": False,
        "benchmark_eligible": False,
        "claim_boundary": (
            "Synthetic seed smoke for detector wiring only; not an official "
            "PINT score and not public benchmark evidence."
        ),
        "total_cases": total,
        "correct": correct_count,
        "accuracy": correct_count / total if total else 0.0,
        "true_positive": counts["tp"],
        "false_positive": counts["fp"],
        "false_negative": counts["fn"],
        "true_negative": counts["tn"],
        "precision": (
            counts["tp"] / precision_denominator if precision_denominator else 0.0
        ),
        "recall": counts["tp"] / recall_denominator if recall_denominator else 0.0,
        "category_accuracy": {
            category: category_correct[category] / count
            for category, count in sorted(category_totals.items())
        },
        "language_accuracy": {
            language: language_correct[language] / count
            for language, count in sorted(language_totals.items())
        },
        "per_case": per_case,
    }


def run_pint_seed_smoke(
    root: Path,
    output: Path = DEFAULT_OUTPUT,
    *,
    detector: BooleanDetector | None = None,
) -> list[str]:
    root = root.resolve()
    errors, cases = load_seed_cases(root)
    if errors:
        return errors
    selected_detector = detector or InputSanitizer()
    result = evaluate_seed_cases(cases, detector=selected_detector)
    result["packet_id"] = _load_packet_id(root)
    result["seed_cases"] = DEFAULT_CASES.as_posix()

    output_path = output if output.is_absolute() else root / output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=Path.cwd(), type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    errors = run_pint_seed_smoke(args.root, args.output)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"pint_seed_smoke_written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
