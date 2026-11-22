# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — the typescript-docs canary run
"""Run the typescript-docs profile's own analysis over its seeded canary fixture.

The run goes through :func:`analyse_typescript`, the function the profile's
capture uses: the same recipe and engine checks, the same helper, and the
same conversion of the helper's report into findings. A change to any of
them reaches the canaries, so no adapter-side change can bypass them.
"""

from __future__ import annotations

from pathlib import Path

from tools.quality_ratchet_succession import FindingKey
from tools.quality_ratchet_succession_check import finding_key
from tools.quality_ratchet_typescript import analyse_typescript


def typescript_canary_findings(
    root: Path, node: str, paths: tuple[str, ...], *, timeout: float
) -> tuple[str, str, tuple[FindingKey, ...]]:
    """Return the engine's platform, its binary digest and the canary findings.

    Parameters
    ----------
    root : Path
        Repository root holding the fixture.
    node : str
        The Node executable the profile runs.
    paths : tuple[str, ...]
        Repository-relative fixture files, sorted.
    timeout : float
        Positive finite whole-analysis deadline in seconds.

    Returns
    -------
    tuple[str, str, tuple[FindingKey, ...]]
        The platform, the executed binary's digest and the sorted finding keys.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If the profile's own analysis refuses the run.
    """
    snapshot, platform, engine = analyse_typescript(
        root, node, lambda _root, _deadline: paths, timeout=timeout
    )
    return platform, engine, tuple(sorted(map(finding_key, snapshot.findings)))
