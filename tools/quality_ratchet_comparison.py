# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — source-bound quality debt comparison
"""Refuse new findings without trading them against removed historical debt."""

from __future__ import annotations

from tools.quality_ratchet_contracts import Snapshot


def compare_snapshots(original: Snapshot, current: Snapshot) -> tuple[str, ...]:
    """Compare complete native findings with an independently trusted baseline.

    Parameters
    ----------
    original : Snapshot
        Original source-linked historical allowance. A candidate-edited
        baseline must never be supplied as trusted authority by the caller.
    current : Snapshot
        Actual current native analysis using the same effective profile.

    Returns
    -------
    tuple[str, ...]
        Deterministic authored refusals. Empty means this comparison passes;
        it does not establish that the caller ran a native tool or supplied
        the independently trusted original baseline.

    Notes
    -----
    New and modified whole files require zero findings. Unchanged files may
    retain only exact original individual findings. Removing a source or a
    finding cannot pay for an introduced, moved or replaced finding.
    """
    if not original.complete or not current.complete:
        return ("Native quality analysis is incomplete",)
    if original.profile != current.profile:
        return ("Native quality profile changed",)
    if original.identity != current.identity:
        return ("Native quality tool, configuration or scope identity changed",)
    previous = set(original.findings)
    violations = set()
    for finding in current.findings:
        path = finding.path
        if original.sources.get(path) != current.sources[path]:
            violations.add(f"New or modified source has native findings: {path}")
        elif finding not in previous:
            violations.add(f"Unchanged source gained a native finding: {path}")
    return tuple(sorted(violations))
