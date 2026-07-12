# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Host-condition metadata for benchmark artefacts

"""Record host load, CPU affinity and an isolation verdict (WCE-2).

Latency figures are only comparable when the artefact says what the
host was doing at measurement time. The 2026-07-12 BEIR CPU run made
the gap concrete: per-query latencies measured at load average 24–35
on 12 hardware threads sat next to figures from an idle host with no
marker distinguishing them. Every benchmark environment block now
embeds :func:`host_conditions` so an artefact carries its own
comparability evidence.

The isolation verdict is deliberately coarse — three honest labels,
not a quality score:

- ``isolated-quiet``: the process is pinned to a strict subset of CPUs
  and the host has head-room (1-minute load per online CPU at or
  below ``QUIET_LOAD_PER_CPU``).
- ``pinned-loaded-host``: pinned, but the rest of the host is busy —
  run-queue pressure can still steal the pinned cores' cycles, so
  latency numbers remain suspect.
- ``shared``: not pinned; the scheduler may migrate the process and
  every other process competes for the same cores.

Pinning is read from ``os.sched_getaffinity``, so running a benchmark
under ``taskset -c …`` is picked up without any code cooperation.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["QUIET_LOAD_PER_CPU", "host_conditions", "isolation_verdict"]

#: 1-minute load average per online CPU at or below which a pinned run
#: is labelled quiet. 0.25 means at most a quarter of the host's
#: hardware threads were busy — conservative on purpose: benchmark
#: claims should err towards the "loaded" label, never away from it.
QUIET_LOAD_PER_CPU = 0.25

_GOVERNOR_PATH = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")


def host_conditions() -> dict[str, Any]:
    """Return the load/affinity snapshot to embed in an artefact."""
    load_1m, load_5m, load_15m = os.getloadavg()
    cpu_total = os.cpu_count() or 1
    affinity = sorted(os.sched_getaffinity(0))
    conditions: dict[str, Any] = {
        "captured_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "load_avg": [round(load_1m, 2), round(load_5m, 2), round(load_15m, 2)],
        "load_per_cpu_1m": round(load_1m / cpu_total, 3),
        "cpu_count": cpu_total,
        "cpu_affinity": affinity,
        "pinned": len(affinity) < cpu_total,
        "cpu_governor": _read_governor(),
    }
    conditions["isolation_verdict"] = isolation_verdict(conditions)
    return conditions


def isolation_verdict(conditions: dict[str, Any]) -> str:
    """Classify a snapshot as isolated-quiet / pinned-loaded-host / shared."""
    if not conditions["pinned"]:
        return "shared"
    if conditions["load_per_cpu_1m"] <= QUIET_LOAD_PER_CPU:
        return "isolated-quiet"
    return "pinned-loaded-host"


def _read_governor() -> str | None:
    """CPU frequency governor, or None where the sysfs knob is absent."""
    try:
        return _GOVERNOR_PATH.read_text(encoding="ascii").strip()
    except OSError:
        return None
