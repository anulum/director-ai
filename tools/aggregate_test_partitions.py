# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — complete native pytest and model evidence aggregate
"""Require all native test partitions and all original 1200 HaluEval reviews."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from tools.ci_workflow_inventory import (
    ROOT,
    expected_halueval_inputs,
    load_ci_workflow_policy,
    object_mapping,
    string_tuple,
)
from tools.pytest_partition_artifacts import partition_artifact_reports
from tools.pytest_partition_schedule import HALUEVAL_PREFIX, partition_assignments
from tools.pytest_partition_timings import LANE_SCHEMA, read_timing_snapshot


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Refuse repeated JSON members in native evidence."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate native evidence member")
        result[key] = value
    return result


def _read_report(path: Path) -> dict[str, object]:
    """Read original JSON with duplicate-member and nonfinite-value refusal."""
    value: object = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_invalid_constant,
    )
    return object_mapping(value)


def _invalid_constant(value: str) -> object:
    """Refuse nonfinite JSON constants in native evidence."""
    raise ValueError("Nonfinite JSON constant: " + value)


def verify_test_partitions(
    report_paths: list[Path],
    revision: str,
    python_version: str,
    root: Path = ROOT,
    timing_snapshot: Path | None = None,
) -> dict[str, object]:
    """Verify the complete disjoint collection, native outcomes and model inputs.

    Parameters
    ----------
    report_paths : list[Path]
        One original terminal report from each required worker.
    revision : str
        Exact source SHA expected from every worker.
    python_version : str
        Expected Python major.minor lane, such as 3.11.
    root : Path
        Checkout containing the immutable workflow and input contracts.
    timing_snapshot : Path or None
        Exact common scheduling snapshot, or None for native cold scheduling.

    Returns
    -------
    dict[str, object]
        Measured complete case count, original model-review count and hashes
        of all verified native report files and the common timing snapshot.
        Scheduling weights sum the native phases for each live unit; zero
        totals receive a positive one-nanosecond scheduling floor while the
        original phase measurements remain unchanged in the worker reports.

    Raises
    ------
    ValueError, KeyError, OSError
        If any required report, test, phase, input or real review is absent,
        duplicated, mismatched, failed, cancelled or unexpectedly skipped.
    """
    policy = load_ci_workflow_policy(root)
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or python_version not in {
        "3.11",
        "3.12",
        "3.13",
    }:
        raise ValueError("An exact source SHA and supported Python lane are required")
    if len(report_paths) != policy.partitions:
        raise ValueError("Every native test partition is required")
    hints, snapshot_digest = read_timing_snapshot(timing_snapshot, python_version)
    expected_inputs = expected_halueval_inputs(root)
    collected: tuple[str, ...] | None = None
    selected_union: set[str] = set()
    indices: set[int] = set()
    actual_inputs: dict[tuple[str, int, bool], str] = {}
    report_hashes = {}
    measured_weights: dict[str, float] = {}
    for path in report_paths:
        report = _read_report(path)
        index = report["index"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < policy.partitions
            or index in indices
        ):
            raise ValueError("Invalid or repeated native partition index")
        indices.add(index)
        version = report["python_version"]
        if (
            not isinstance(version, str)
            or ".".join(version.split(".")[:2]) != python_version
        ):
            raise ValueError("Native Python lane differs from the requested version")
        if (
            report["schema"] != "director-test-partition.v2"
            or report["revision"] != revision
            or report["exit_code"] != 0
            or report["count"] != policy.partitions
            or report["rows_per_task"] != policy.rows_per_task
            or report["targets"] != ["tests/"]
            or type(report["exit_code"]) is not int
            or type(report["count"]) is not int
            or type(report["rows_per_task"]) is not int
            or report["timing_snapshot_sha256"] != snapshot_digest
        ):
            raise ValueError(
                "Native test report does not prove the complete required run"
            )
        all_cases = string_tuple(report["collected"])
        if not all_cases or (collected is not None and all_cases != collected):
            raise ValueError("Native collections differ across required workers")
        collected = all_cases
        selected = string_tuple(report["selected"])
        assignments = partition_assignments(
            list(collected), policy.partitions, policy.rows_per_task, hints
        )
        expected = tuple(case for case in collected if assignments[case] == index)
        if (
            not selected
            or selected != expected
            or selected_union.intersection(selected)
        ):
            raise ValueError(
                "Selected native cases are missing, overlapping or assigned incorrectly"
            )
        selected_union.update(selected)
        results = report["results"]
        if not isinstance(results, list):
            raise ValueError("Native phase results must be a list")
        phases: dict[str, dict[str, dict[str, object]]] = {
            case: {} for case in selected
        }
        for raw in results:
            row = object_mapping(raw)
            case, phase, outcome = row["nodeid"], row["phase"], row["outcome"]
            if (
                not isinstance(case, str)
                or case not in phases
                or not isinstance(phase, str)
                or phase not in {"setup", "call", "teardown"}
            ):
                raise ValueError("Unexpected native case or test phase")
            if (
                phase in phases[case]
                or not isinstance(outcome, str)
                or outcome not in {"passed", "skipped"}
            ):
                raise ValueError("Repeated, failed or invalid native test phase")
            phases[case][phase] = row
            duration = row["duration"]
            if (
                type(duration) is not float
                or not math.isfinite(duration)
                or duration < 0
            ):
                raise ValueError("Native phase duration must be finite and nonnegative")
            unit = (
                case
                if case.startswith(HALUEVAL_PREFIX)
                else case.split("::", maxsplit=1)[0]
            )
            elapsed = measured_weights.get(unit, 0.0) + duration
            if not math.isfinite(elapsed):
                raise ValueError("Native phase durations overflow the measured unit")
            measured_weights[unit] = elapsed
        for case, parts in phases.items():
            if "setup" not in parts or "teardown" not in parts:
                raise ValueError(
                    "A selected native case lacks setup or teardown evidence"
                )
            if parts["setup"]["outcome"] == "passed" and "call" not in parts:
                raise ValueError("A selected native case did not execute its test call")
            if parts["setup"]["outcome"] == "skipped" and "call" in parts:
                raise ValueError("A skipped setup cannot have an executed test call")
            if parts["teardown"]["outcome"] != "passed":
                raise ValueError("A selected native case did not finish teardown")
            if not case.startswith(
                "tests/test_halueval_benchmark.py::test_halueval_full"
            ):
                continue
            if set(parts) != {"setup", "call", "teardown"} or any(
                row["outcome"] != "passed" for row in parts.values()
            ):
                raise ValueError("Required model-backed HaluEval case did not pass")
            evidence = parts["call"]["halueval"]
            if (
                not isinstance(evidence, list)
                or len(evidence) != 1
                or not isinstance(evidence[0], str)
            ):
                raise ValueError(
                    "Required HaluEval case lacks unique actual review evidence"
                )
            payload = object_mapping(
                json.loads(
                    evidence[0],
                    object_pairs_hook=_unique_object,
                    parse_constant=_invalid_constant,
                )
            )
            task, offset = payload["task"], payload["sample_offset"]
            size = 1
            if (
                not isinstance(task, str)
                or task not in policy.dataset_sha256
                or isinstance(offset, bool)
                or not isinstance(offset, int)
            ):
                raise ValueError("HaluEval task range identity is malformed")
            if (
                case
                != f"tests/test_halueval_benchmark.py::test_halueval_full[{task}-{offset}]"
                or not 0 <= offset < policy.rows_per_task
            ):
                raise ValueError("HaluEval case and original row range differ")
            if (
                payload["dataset_sha256"] != policy.dataset_sha256[task]
                or payload["model_required"] is not True
            ):
                raise ValueError("Required HaluEval dataset or model evidence differs")
            reviews = payload["evaluations"]
            if not isinstance(reviews, list) or len(reviews) != 2 * size:
                raise ValueError("Required HaluEval range lacks both original labels")
            for raw_review in reviews:
                review = object_mapping(raw_review)
                sample, label, score, digest = (
                    review["sample_index"],
                    review["is_hallucinated"],
                    review["score"],
                    review["input_sha256"],
                )
                if (
                    isinstance(sample, bool)
                    or not isinstance(sample, int)
                    or not isinstance(label, bool)
                    or review["task"] != task
                ):
                    raise ValueError("HaluEval review identity is malformed")
                if (
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(score)
                    or not 0 <= score <= 1
                ):
                    raise ValueError(
                        "Required public review has an invalid actual score"
                    )
                if not isinstance(digest, str):
                    raise ValueError("HaluEval input digest must be a string")
                key = (task, sample, label)
                if (
                    not offset <= sample < offset + size
                    or key in actual_inputs
                    or digest != expected_inputs.get(key)
                ):
                    raise ValueError(
                        "HaluEval review is repeated or differs from the original required input"
                    )
                actual_inputs[key] = digest
        report_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if (
        collected is None
        or selected_union != set(collected)
        or indices != set(range(policy.partitions))
    ):
        raise ValueError(
            "Native partitions do not cover the complete collected test suite"
        )
    required_cases = {
        f"tests/test_halueval_benchmark.py::test_halueval_full[{task}-{offset}]"
        for task in policy.dataset_sha256
        for offset in range(policy.rows_per_task)
    }
    if not required_cases <= selected_union or actual_inputs != expected_inputs:
        raise ValueError("The complete original 1200-review HaluEval run is not proven")
    return {
        "revision": revision,
        "python_version": python_version,
        "partitions": len(indices),
        "collected_cases": len(collected),
        "halueval_reviews": len(actual_inputs),
        "report_sha256": report_hashes,
        "timing_snapshot_sha256": snapshot_digest,
        "scheduling_weights": {
            unit: max(seconds, 1e-9) for unit, seconds in measured_weights.items()
        },
    }


def main() -> int:
    """Verify actual downloaded worker artefacts through the public CLI.

    Returns
    -------
    int
        Zero after complete native evidence verification, one for any refusal.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timings", type=Path)
    parser.add_argument("--timing-output", type=Path)
    args = parser.parse_args()
    try:
        result = verify_test_partitions(
            partition_artifact_reports(
                args.input, args.python_version, load_ci_workflow_policy().partitions
            ),
            args.revision,
            args.python_version,
            timing_snapshot=args.timings,
        )
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        if args.timing_output is not None:
            with args.timing_output.open("x", encoding="utf-8") as stream:
                json.dump(
                    {
                        "schema": LANE_SCHEMA,
                        "revision": args.revision,
                        "python_version": args.python_version,
                        "weights": result["scheduling_weights"],
                    },
                    stream,
                    sort_keys=True,
                    allow_nan=False,
                )
    except (OSError, ValueError, KeyError):
        print("Complete native test and model evidence verification refused")
        return 1
    print(
        f"Verified {result['collected_cases']} native cases and {result['halueval_reviews']} original model-backed reviews"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
