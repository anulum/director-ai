# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — common native pytest timing snapshots
"""Validate immutable scheduling advice without selecting or proving any tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from tools.ci_workflow_inventory import object_mapping

PYTHON_LANES = ("3.11", "3.12", "3.13")
TIMING_SCHEMA = "director-pytest-timings.v1"
LANE_SCHEMA = "director-pytest-timing-lane.v1"
COLD_SNAPSHOT = (
    json.dumps(
        {
            "schema": TIMING_SCHEMA,
            "source_revision": None,
            "lanes": {lane: {} for lane in PYTHON_LANES},
        },
        sort_keys=True,
    )
    + "\n"
).encode()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Refuse duplicate members rather than letting a later hint replace one."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate timing snapshot member")
        result[key] = value
    return result


def _invalid_constant(value: str) -> object:
    """Reject nonfinite JSON constants before scheduling starts."""
    raise ValueError("Nonfinite timing snapshot constant: " + value)


def _decode_mapping(raw: bytes) -> dict[str, object]:
    """Decode a complete mapping with strict duplicate and number handling."""
    value: object = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_invalid_constant,
    )
    return object_mapping(value)


def _validated_lanes(value: object) -> dict[str, dict[str, float]]:
    """Validate every lane, including hints unused by the current interpreter."""
    lanes = object_mapping(value)
    if set(lanes) != set(PYTHON_LANES):
        raise ValueError("All supported Python timing lanes are required")
    result: dict[str, dict[str, float]] = {}
    for lane, raw_weights in lanes.items():
        weights = object_mapping(raw_weights)
        result[lane] = {}
        for unit, seconds in weights.items():
            if (
                not unit
                or isinstance(seconds, bool)
                or not isinstance(seconds, (int, float))
            ):
                raise ValueError("Timing weights require names and finite numbers")
            try:
                number = float(seconds)
            except OverflowError as exc:
                raise ValueError("Timing weight exceeds finite numeric bounds") from exc
            if not math.isfinite(number) or number <= 0:
                raise ValueError("Timing weights must be positive and finite")
            result[lane][unit] = number
    return result


def read_timing_snapshot(path: Path | None, lane: str) -> tuple[dict[str, float], str]:
    """Read one lane's advice and hash the exact shared snapshot bytes.

    Parameters
    ----------
    path : Path or None
        Existing immutable snapshot, or None for complete cold scheduling.
        An explicitly supplied absent or corrupt file is never a cold fallback.
    lane : str
        Supported Python major.minor lane.

    Returns
    -------
    tuple of dict and str
        Validated positive scheduling weights and the snapshot SHA-256.

    Raises
    ------
    ValueError, KeyError, OSError
        If any snapshot lane, provenance, weight, mapping or file is invalid.
    """
    if lane not in PYTHON_LANES:
        raise ValueError("Unsupported Python timing lane")
    raw = COLD_SNAPSHOT if path is None else path.read_bytes()
    data = _decode_mapping(raw)
    if (
        set(data) != {"schema", "source_revision", "lanes"}
        or data["schema"] != TIMING_SCHEMA
    ):
        raise ValueError("Unsupported timing snapshot contract")
    revision = data["source_revision"]
    if revision is not None and (
        not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)
    ):
        raise ValueError("Timing source revision must be an exact commit")
    lanes = _validated_lanes(data["lanes"])
    if revision is None and any(lanes.values()):
        raise ValueError("Observed timing hints require their source revision")
    return lanes[lane], hashlib.sha256(raw).hexdigest()


def merge_timing_reports(reports: Path, revision: str, output: Path) -> None:
    """Publish common scheduling advice from the three verified lane writers.

    Parameters
    ----------
    reports : Path
        Fresh downloaded per-lane artifacts containing timings.json. These
        contain scheduling weights, not replacement test-execution receipts.
    revision : str
        Exact current commit required from every lane writer.
    output : Path
        Fresh immutable snapshot destination.

    Raises
    ------
    ValueError, KeyError, OSError
        If any lane is missing, repeated, malformed or from another revision.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("An exact timing producer revision is required")
    paths = sorted(reports.glob("*/timings.json"))
    if len(paths) != len(PYTHON_LANES):
        raise ValueError("Every supported timing lane is required")
    lanes: dict[str, object] = {}
    for path in paths:
        data = _decode_mapping(path.read_bytes())
        lane = data["python_version"]
        if (
            set(data) != {"schema", "revision", "python_version", "weights"}
            or data["schema"] != LANE_SCHEMA
            or data["revision"] != revision
            or not isinstance(lane, str)
            or lane not in PYTHON_LANES
            or lane in lanes
        ):
            raise ValueError("Timing lane identity or provenance differs")
        lanes[lane] = data["weights"]
    result = {
        "schema": TIMING_SCHEMA,
        "source_revision": revision,
        "lanes": _validated_lanes(lanes),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> int:
    """Prepare, validate or retain scheduling advice through the normal CLI.

    Returns
    -------
    int
        Zero for valid scheduling advice, one for a refused snapshot.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cold = commands.add_parser("cold")
    cold.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--input", type=Path, required=True)
    record = commands.add_parser("record")
    record.add_argument("--input", type=Path, required=True)
    record.add_argument("--revision", required=True)
    record.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "cold":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("xb") as stream:
                stream.write(COLD_SNAPSHOT)
        elif args.command == "validate":
            read_timing_snapshot(args.input, PYTHON_LANES[0])
        else:
            merge_timing_reports(args.input, args.revision, args.output)
    except (ValueError, KeyError, OSError):
        print("Test timing snapshot refused")
        return 1
    print("Test timing snapshot validated; advice does not select test membership")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
