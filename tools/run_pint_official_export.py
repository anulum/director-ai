#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - PINT official export evidence runner
"""Run a locally supplied PINT-format export without making a public score claim."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tomllib
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import yaml

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

PACKET: Path = _VALIDATOR.PACKET
validate_pint_replication_packet = _VALIDATOR.validate_pint_replication_packet

DEFAULT_OUTPUT = Path("benchmarks/results/pint_official_export_evidence.json")


class BooleanDetector(Protocol):
    """Detector contract for PINT text-to-boolean evaluation."""

    def score(self, text: str) -> Any:
        """Return an object with blocked/suspicion_score/pattern fields."""


@dataclass(frozen=True)
class PintExportCase:
    """One PINT-format row loaded from a local export."""

    row_id: str
    category: str
    text: str
    label: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_packet(root: Path) -> dict[str, Any]:
    return tomllib.loads((root / PACKET).read_text(encoding="utf-8"))


def _as_rows(payload: Any, *, source: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
        rows = payload["data"]
    else:
        raise ValueError(f"{source}: expected a list of PINT row objects")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{source}: every row must be an object")
    return rows


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            raise ValueError(f"{path}:{line_number}: blank lines are not allowed")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        rows.append(row)
    return rows


def _load_raw_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _load_jsonl(path)
    if suffix == ".json":
        return _as_rows(json.loads(path.read_text(encoding="utf-8")), source=path)
    if suffix in {".yaml", ".yml"}:
        return _as_rows(yaml.safe_load(path.read_text(encoding="utf-8")), source=path)
    raise ValueError(f"{path}: supported formats are .yaml, .yml, .json, .jsonl")


def load_export_cases(path: Path) -> tuple[PintExportCase, ...]:
    """Load PINT-format rows from YAML, JSON, or JSONL."""

    rows = _load_raw_rows(path)
    cases: list[PintExportCase] = []
    for index, row in enumerate(rows, 1):
        text = row.get("text")
        category = row.get("category")
        label = row.get("label")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}:{index}: text must be a non-empty string")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"{path}:{index}: category must be a non-empty string")
        if not isinstance(label, bool):
            raise ValueError(f"{path}:{index}: label must be boolean")
        row_id = row.get("id", f"pint-export-{index:06d}")
        if not isinstance(row_id, str) or not row_id.strip():
            raise ValueError(f"{path}:{index}: id must be a non-empty string")
        cases.append(
            PintExportCase(
                row_id=row_id,
                category=category,
                text=text,
                label=label,
            )
        )
    return tuple(cases)


def _accuracy_by_group(
    cases: Sequence[PintExportCase],
    correct_ids: set[str],
    group_values: Iterable[str],
) -> dict[str, float]:
    totals: Counter[str] = Counter(group_values)
    correct: Counter[str] = Counter(
        case.category for case in cases if case.row_id in correct_ids
    )
    return {
        category: correct[category] / count
        for category, count in sorted(totals.items())
    }


def evaluate_export_cases(
    cases: Sequence[PintExportCase],
    *,
    detector: BooleanDetector,
    dataset_path: Path,
    packet: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a local PINT export and omit prompt text from the result."""

    counts: Counter[str] = Counter()
    correct_ids: set[str] = set()
    per_case: list[dict[str, Any]] = []
    for case in cases:
        score = detector.score(case.text)
        prediction = bool(getattr(score, "blocked", False))
        correct = prediction == case.label
        if correct:
            counts["correct"] += 1
            correct_ids.add(case.row_id)
        if prediction and case.label:
            counts["tp"] += 1
        elif prediction and not case.label:
            counts["fp"] += 1
        elif not prediction and case.label:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
        per_case.append(
            {
                "id": case.row_id,
                "category": case.category,
                "label": case.label,
                "prediction": prediction,
                "correct": correct,
                "suspicion_score": float(getattr(score, "suspicion_score", 0.0)),
                "pattern": str(getattr(score, "pattern", "")),
                "matches": list(getattr(score, "matches", ())),
            }
        )

    total = len(cases)
    precision_denominator = counts["tp"] + counts["fp"]
    recall_denominator = counts["tp"] + counts["fn"]
    return {
        "schema_version": "1.0.0",
        "result_type": "pint_official_export_evidence",
        "packet_id": packet.get("packet_id"),
        "upstream_repository": packet.get("upstream_repository"),
        "upstream_blog": packet.get("upstream_blog"),
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "public_score_claim": False,
        "official_pint_score_evidence": True,
        "claim_boundary": (
            "Local evidence from a supplied PINT-format export only; no public "
            "PINT score claim is approved without operator review, preserved "
            "run artefacts, and a claim-guarded benchmark card."
        ),
        "total_cases": total,
        "correct": counts["correct"],
        "accuracy": counts["correct"] / total if total else 0.0,
        "true_positive": counts["tp"],
        "false_positive": counts["fp"],
        "false_negative": counts["fn"],
        "true_negative": counts["tn"],
        "precision": (
            counts["tp"] / precision_denominator if precision_denominator else 0.0
        ),
        "recall": counts["tp"] / recall_denominator if recall_denominator else 0.0,
        "category_accuracy": _accuracy_by_group(
            cases,
            correct_ids,
            (case.category for case in cases),
        ),
        "per_case": per_case,
    }


def run_pint_official_export(
    root: Path,
    dataset: Path,
    output: Path = DEFAULT_OUTPUT,
    *,
    detector: BooleanDetector | None = None,
) -> list[str]:
    """Run the supplied PINT export after validating the local packet."""

    root = root.resolve()
    dataset_path = dataset if dataset.is_absolute() else root / dataset
    errors = cast(list[str], validate_pint_replication_packet(root))
    if errors:
        return errors
    if not dataset_path.is_file():
        return [f"{dataset_path}: missing PINT export dataset"]
    try:
        cases = load_export_cases(dataset_path)
    except (ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        return [str(exc)]
    packet = _load_packet(root)
    selected_detector = detector or InputSanitizer()
    result = evaluate_export_cases(
        cases,
        detector=selected_detector,
        dataset_path=dataset_path,
        packet=packet,
    )

    output_path = output if output.is_absolute() else root / output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--root", default=Path.cwd(), type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    errors = run_pint_official_export(args.root, args.dataset, args.output)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"pint_official_export_evidence_written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
