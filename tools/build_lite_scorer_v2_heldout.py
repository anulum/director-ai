#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 held-out dataset builder

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SOURCE = Path("training/data/eval")
DEFAULT_OUTPUT = Path("benchmarks/heldout/lite_scorer_v2.jsonl")
DEFAULT_MANIFEST = Path("benchmarks/heldout/lite_scorer_v2.manifest.toml")


@dataclass(frozen=True)
class HeldoutBuildConfig:
    target_rows: int
    seed: int
    min_sources: int
    output: Path
    manifest: Path
    source_dataset: str


def _normalise_row(
    raw: dict[str, Any], row_number: int
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    premise = raw.get("premise")
    hypothesis = raw.get("hypothesis")
    label = raw.get("label")
    source = raw.get("source")
    if not isinstance(premise, str) or not premise.strip():
        errors.append(f"row {row_number}: premise must be a non-empty string")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        errors.append(f"row {row_number}: hypothesis must be a non-empty string")
    if not isinstance(source, str) or not source.strip():
        errors.append(f"row {row_number}: source must be a non-empty string")
    if label not in {0, 1, 2}:
        errors.append(f"row {row_number}: label must be one of 0, 1, or 2")
    if errors:
        return None, errors
    return (
        {
            "premise": premise,
            "hypothesis": hypothesis,
            "label": label == 0,
            "source": source,
            "source_label": label,
        },
        [],
    )


def _validate_selection_request(
    rows: list[dict[str, Any]],
    target_rows: int,
    min_sources: int,
) -> list[str]:
    errors: list[str] = []
    if target_rows < 2:
        errors.append("target_rows must be at least 2")
    if target_rows % 2 != 0:
        errors.append("target_rows must be even for balanced labels")
    if min_sources < 1:
        errors.append("min_sources must be positive")
    if not rows:
        errors.append("source rows must not be empty")
    return errors


def select_lite_scorer_v2_heldout_rows(
    rows: Iterable[dict[str, Any]],
    *,
    target_rows: int,
    seed: int,
    min_sources: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    raw_rows = list(rows)
    errors = _validate_selection_request(raw_rows, target_rows, min_sources)
    if errors:
        return [], errors

    normalised: list[dict[str, Any]] = []
    for row_number, raw in enumerate(raw_rows, 1):
        row, row_errors = _normalise_row(raw, row_number)
        if row_errors:
            return [], row_errors
        if row is not None:
            normalised.append(row)

    sources = {row["source"] for row in normalised}
    if len(sources) < min_sources:
        return [], [f"source rows must contain at least {min_sources} distinct sources"]

    supported = [row for row in normalised if row["label"] is True]
    unsupported = [row for row in normalised if row["label"] is False]
    per_label = target_rows // 2
    if len(supported) < per_label:
        return [], [
            f"source rows contain {len(supported)} supported rows; need {per_label}"
        ]
    if len(unsupported) < per_label:
        return [], [
            f"source rows contain {len(unsupported)} unsupported rows; need {per_label}"
        ]

    rng = random.Random(seed)
    selected = rng.sample(supported, per_label) + rng.sample(unsupported, per_label)
    rng.shuffle(selected)
    selected_sources = {row["source"] for row in selected}
    if len(selected_sources) < min_sources:
        return [], [
            f"selected rows must contain at least {min_sources} distinct sources"
        ]
    return selected, []


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _render_manifest(
    config: HeldoutBuildConfig, rows: list[dict[str, Any]], sha256: str
) -> str:
    labels = Counter(row["label"] for row in rows)
    sources = Counter(str(row["source"]) for row in rows)
    source_lines = [
        f"{json.dumps(source)} = {count}" for source, count in sorted(sources.items())
    ]
    header = [
        "# SPDX-License-Identifier: AGPL-3.0-or-later",
        "# Commercial licence available",
        "# Concepts 1996-2026 Miroslav Sotek. All rights reserved.",
        "# Code 2020-2026 Miroslav Sotek. All rights reserved.",
        "# ORCID: 0009-0009-3560-0851",
        "# Contact: www.anulum.li | protoscience@anulum.li",
        "# Director-Class AI - Lite Scorer v2 held-out dataset manifest",
        "",
    ]
    fields = [
        'schema_version = "1.0.0"',
        'dataset_id = "lite-scorer-v2-heldout"',
        f"source_dataset = {_toml_string(config.source_dataset)}",
        f"output = {_toml_string(config.output.as_posix())}",
        f"rows = {len(rows)}",
        f"supported_rows = {labels[True]}",
        f"unsupported_rows = {labels[False]}",
        f"seed = {config.seed}",
        f"min_sources = {config.min_sources}",
        f"sha256 = {_toml_string(sha256)}",
        "",
        "[source_counts]",
        *source_lines,
    ]
    return "\n".join(header + fields) + "\n"


def build_lite_scorer_v2_heldout_from_rows(
    rows: Iterable[dict[str, Any]],
    config: HeldoutBuildConfig,
) -> list[str]:
    selected, errors = select_lite_scorer_v2_heldout_rows(
        rows,
        target_rows=config.target_rows,
        seed=config.seed,
        min_sources=config.min_sources,
    )
    if errors:
        return errors
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected)
    config.output.write_text(payload, encoding="utf-8")
    digest = _sha256(config.output)
    config.manifest.write_text(
        _render_manifest(config, selected, digest),
        encoding="utf-8",
    )
    return []


def _load_hf_eval_rows(path: Path) -> list[dict[str, Any]]:
    from datasets import load_from_disk

    dataset = load_from_disk(str(path))
    return [dict(row) for row in dataset]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--target-rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--min-sources", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    source = args.source
    if not source.exists():
        print(f"{source}: source dataset does not exist", file=sys.stderr)
        return 1
    rows = _load_hf_eval_rows(source)
    config = HeldoutBuildConfig(
        target_rows=args.target_rows,
        seed=args.seed,
        min_sources=args.min_sources,
        output=args.output,
        manifest=args.manifest,
        source_dataset=source.as_posix(),
    )
    errors = build_lite_scorer_v2_heldout_from_rows(rows, config)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"lite_scorer_v2_heldout_built rows={args.target_rows} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
