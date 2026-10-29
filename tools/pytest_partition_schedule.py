# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — complete native pytest scheduling
"""Assign discovered modules and original HaluEval rows to exclusive workers."""

from __future__ import annotations

import hashlib
import heapq
import math
import re
import statistics

HALUEVAL_PREFIX = "tests/test_halueval_benchmark.py::test_halueval_full"


def partition_for_nodeid(nodeid: str, count: int, rows_per_task: int) -> int:
    """Assign a native case without historical timing information.

    Parameters
    ----------
    nodeid : str
        Identifier from the actual pytest collection.
    count, rows_per_task : int
        Positive worker count and complete original HaluEval task row count.

    Returns
    -------
    int
        Exclusive worker index. Ordinary modules stay together; HaluEval rows
        are distributed cyclically without changing either original response.

    Raises
    ------
    ValueError
        If the identifier or complete partition contract is invalid.
    """
    if (
        isinstance(count, bool)
        or isinstance(rows_per_task, bool)
        or not isinstance(count, int)
        or not isinstance(rows_per_task, int)
        or count <= 0
        or rows_per_task <= 0
        or rows_per_task % count
    ):
        raise ValueError("Invalid complete test partition contract")
    if not nodeid or not nodeid.split("::", maxsplit=1)[0].endswith(".py"):
        raise ValueError("A native pytest module identifier is required")
    if nodeid.startswith(HALUEVAL_PREFIX):
        match = re.fullmatch(
            re.escape(HALUEVAL_PREFIX) + r"\[(qa|summarization|dialogue)-(\d+)\]",
            nodeid,
        )
        if match is None:
            raise ValueError("Full HaluEval case lacks an original task row")
        row = int(match.group(2))
        if row >= rows_per_task or str(row) != match.group(2):
            raise ValueError("Full HaluEval case has an undeclared original row")
        return row % count
    module = nodeid.split("::", maxsplit=1)[0]
    return int.from_bytes(hashlib.sha256(module.encode()).digest(), "big") % count


def partition_assignments(
    nodeids: list[str], count: int, rows_per_task: int, timings: dict[str, float]
) -> dict[str, int]:
    """Schedule every discovered case once, using durations only as advice.

    Parameters
    ----------
    nodeids : list of str
        Complete native discovery in its original order, without duplicates.
    count, rows_per_task : int
        Existing complete worker and original dataset contracts.
    timings : dict of str to float
        Positive measured duration weights keyed by ordinary module or full
        HaluEval row identifier. Entries absent from discovery add no tests.

    Returns
    -------
    dict of str to int
        Every discovered identifier mapped to exactly one worker. A cold run
        uses deterministic module hashing and cyclic model rows. A warm run
        assigns longest units first, breaking ties by unit and worker index.
        New units receive the median duration of currently observed units.

    Raises
    ------
    ValueError
        If collection, partition contract or any supplied weight is invalid.
    """
    if not nodeids or len(set(nodeids)) != len(nodeids):
        raise ValueError("Native collection must be nonempty with unique case IDs")
    weights: dict[str, float] = {}
    for name, seconds in timings.items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(seconds, bool)
            or not isinstance(seconds, (float, int))
        ):
            raise ValueError(
                "Timing weights require names and positive finite durations"
            )
        try:
            value = float(seconds)
        except OverflowError as exc:
            raise ValueError(
                "Timing weights require positive finite durations"
            ) from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Timing weights require positive finite durations")
        weights[name] = value
    cold = {
        nodeid: partition_for_nodeid(nodeid, count, rows_per_task) for nodeid in nodeids
    }
    units = {
        nodeid: nodeid
        if nodeid.startswith(HALUEVAL_PREFIX)
        else nodeid.split("::", maxsplit=1)[0]
        for nodeid in nodeids
    }
    live = set(units.values())
    observed = {name: seconds for name, seconds in weights.items() if name in live}
    if not observed:
        return cold
    default = statistics.median(observed.values())
    loads = [(0.0, index) for index in range(count)]
    heapq.heapify(loads)
    owners: dict[str, int] = {}
    for unit in sorted(live, key=lambda name: (-observed.get(name, default), name)):
        load, index = heapq.heappop(loads)
        owners[unit] = index
        updated = load + observed.get(unit, default)
        if not math.isfinite(updated):
            raise ValueError("Timing weights overflow the finite worker load")
        heapq.heappush(loads, (updated, index))
    return {nodeid: owners[unit] for nodeid, unit in units.items()}
