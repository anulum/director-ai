# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Threshold-tuner data feeder

"""Export a JSONL feed for the Julia threshold tuner.

The Julia tuner consumes a simple line-per-example format:

    {"score": 0.74, "label": true,  "source": "aggrefact:summ"}
    {"score": 0.31, "label": false, "source": "aggrefact:summ"}

This script accepts one of two inputs:

1. An AggreFact-style per-sample CSV / JSONL with columns
   ``score`` and ``label`` (and optional ``dataset``/``source``).
2. A Director-AI ``benchmarks/results/*.json`` produced by
   ``aggrefact_eval.py`` when run in ``--per-sample`` mode.

Records with ``NaN``/``None`` scores are dropped with a warning.
Labels coerce from boolean, 0/1, and the strings
``"support"/"SUPPORTED"/"true"`` → ``true``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger("DirectorAI.PrepareThresholdData")

_TRUE_STRINGS = frozenset({"true", "1", "yes", "supported", "support", "grounded"})
_FALSE_STRINGS = frozenset(
    {"false", "0", "no", "unsupported", "not_supported", "hallucinated"}
)


def _coerce_label(raw: Any) -> bool | None:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        if isinstance(raw, float) and math.isnan(raw):
            return None
        return bool(raw)
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in _TRUE_STRINGS:
            return True
        if lowered in _FALSE_STRINGS:
            return False
    return None


def _coerce_score(raw: Any) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield raw rows from the input file (CSV or JSONL)."""
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    yield json.loads(stripped)
    elif suffix == ".json":
        with path.open(encoding="utf-8") as f:
            doc = json.load(f)
        if isinstance(doc, list):
            yield from doc
        elif isinstance(doc, dict) and isinstance(doc.get("records"), list):
            yield from doc["records"]
        else:
            raise ValueError(f"{path}: expected a list or a dict with 'records' key")
    elif suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as f:
            yield from csv.DictReader(f)
    else:
        raise ValueError(f"Unsupported extension: {suffix!r}")


def normalise(
    records: Iterable[dict[str, Any]],
    *,
    score_key: str,
    label_key: str,
    source_key: str | None,
) -> Iterator[dict[str, Any]]:
    """Coerce, filter, and reshape records for the Julia tuner."""
    dropped = 0
    kept = 0
    for row in records:
        score = _coerce_score(row.get(score_key))
        label = _coerce_label(row.get(label_key))
        if score is None or label is None:
            dropped += 1
            continue
        out: dict[str, Any] = {"score": score, "label": label}
        if source_key:
            source = row.get(source_key)
            if source is not None:
                out["source"] = str(source)
        kept += 1
        yield out
    if dropped:
        logger.warning("%d record(s) dropped (missing score/label)", dropped)
    if not kept:
        raise ValueError("no usable records found — check score/label keys")


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, separators=(",", ":"), ensure_ascii=False))
            f.write("\n")
            count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prepare_threshold_data",
        description="Export labelled scorer outputs for the Julia tuner.",
    )
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--score-key", default="score")
    parser.add_argument("--label-key", default="label")
    parser.add_argument(
        "--source-key",
        default="dataset",
        help="Column to copy as 'source' (set empty string to disable).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    source_key = args.source_key or None
    records = normalise(
        iter_records(args.input),
        score_key=args.score_key,
        label_key=args.label_key,
        source_key=source_key,
    )
    n = write_jsonl(records, args.output)
    logger.info("wrote %d records → %s", n, args.output)
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
