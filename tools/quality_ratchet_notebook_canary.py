# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — the notebook-types canary run
"""Run the notebook-types profile's own analysis over its seeded canary fixture.

The run goes through :func:`analyse_notebooks`, the function the profile's
capture uses: the same engine checks, projection, MyPy run and rebinding to
notebook cells. A change to any of them reaches the canaries.

The enabled rules are derived mechanically from the trusted base commit's
committed notebook-types baseline, never from the candidate's: one rule per
MyPy error code recorded there. Other MyPy codes are not checked by this
canary. Four named projection items join them, each a canary key with its
notebook cell and line, so a projection that moves code changes a key:

- ``notebook-shell-escape``: a static shell escape the projection kept out
  of the compiled text, from the engine's escape report;
- ``notebook-top-level-await``: the native top-level await advisory;
- ``notebook-refused-shell-escape`` and ``notebook-refused-python-syntax``:
  a ``refused-*`` fixture, whose last code cell ends with the refused
  statement, refused for that reason. Its key carries that cell and line;
  where the refusal reports coordinates, they must be the same.
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path

from tools.ci_workflow_inventory import object_mapping
from tools.quality_ratchet_baseline import read_committed_baseline
from tools.quality_ratchet_contracts import decode_quality_json
from tools.quality_ratchet_inventory import source_path
from tools.quality_ratchet_notebook import analyse_notebooks
from tools.quality_ratchet_succession import FindingKey
from tools.quality_ratchet_succession_check import finding_key

CANARY_NOTEBOOK = "canary.ipynb.json"
"""The fixture notebook holding the seeded findings and projected items."""

REFUSED_PREFIX = "refused-"
"""The name prefix of each fixture notebook the projection must refuse."""

PROJECTION_RULES = (
    "notebook-refused-python-syntax",
    "notebook-refused-shell-escape",
    "notebook-shell-escape",
    "notebook-top-level-await",
)
"""The named projection items, which every canary run must fire."""

_REASONS = {
    "Unsupported notebook shell escape": "notebook-refused-shell-escape",
    "Unsupported notebook Python syntax": "notebook-refused-python-syntax",
}
_ADVISORIES = {
    "notebook-shell-escape": "notebook-shell-escape",
    "notebook-native-top-level-await": "notebook-top-level-await",
}
_REPORTED = re.compile(r" at cell ([1-9][0-9]*) line ([1-9][0-9]*)$")


def enabled_rules(root: Path, base: str) -> tuple[str, ...]:
    """Derive the canary's rules from the trusted base's committed baseline.

    Parameters
    ----------
    root : Path
        Repository root.
    base : str
        The resolved trusted base commit.

    Returns
    -------
    tuple[str, ...]
        The sorted MyPy codes of the base's notebook-types findings, with
        :data:`PROJECTION_RULES`.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If Git refuses, or the base has no notebook-types baseline.
    """
    baseline = read_committed_baseline(root, base)
    snapshot = None if baseline is None else baseline.profiles.get("notebook-types")
    if snapshot is None:
        raise ValueError("The trusted base has no notebook-types baseline")
    codes = {finding.rule for finding in snapshot.findings}
    return tuple(sorted(codes | set(PROJECTION_RULES)))


def _remaining(deadline: float) -> float:
    """Return the positive time left before the whole canary deadline."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ValueError("Native notebook canary exhausted its whole deadline")
    return remaining


def _advisory_keys(path: str, advisories: tuple[str, ...]) -> list[FindingKey]:
    """Return the keys of the projected escapes and native awaits of one path."""
    keys = []
    for advisory in advisories:
        kind, _, rest = advisory.partition(":")
        rule = _ADVISORIES.get(kind)
        if rule is None:
            continue
        named, cell, line = rest.rsplit(":", 2)
        if named == path:
            keys.append(FindingKey(path, rule, int(line), None, int(cell)))
    return keys


def _last_statement(root: Path, path: str) -> tuple[int, int]:
    """Return the one-based cell and line of a fixture's last code line."""
    notebook = object_mapping(decode_quality_json(source_path(root, path).read_bytes()))
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise ValueError(f"Notebook canary fixture {path} has no cells")
    position = None
    for index, raw in enumerate(cells, 1):
        cell = object_mapping(raw)
        source = cell.get("source")
        text = "".join(source) if isinstance(source, list) else source
        if cell.get("cell_type") == "code" and isinstance(text, str) and text.strip():
            position = (index, len(text.rstrip("\n").split("\n")))
    if position is None:
        raise ValueError(f"Notebook canary fixture {path} has no code cell")
    return position


def _refused_key(root: Path, python: str, path: str, timeout: float) -> FindingKey:
    """Run one refused fixture and return the key of its refusal."""
    try:
        analyse_notebooks(root, python, lambda _root: (path,), timeout=timeout)
    except ValueError as error:
        message = str(error)
    else:
        raise ValueError(f"Notebook canary fixture {path} was not refused")
    rule = next(
        (
            name
            for reason, name in _REASONS.items()
            if message.startswith(f"{reason} in {path}")
        ),
        None,
    )
    if rule is None:
        raise ValueError(
            f"Notebook canary fixture {path} was refused for another reason: {message}"
        )
    cell, line = _last_statement(root, path)
    reported = _REPORTED.search(message)
    if reported is not None and (int(reported[1]), int(reported[2])) != (cell, line):
        raise ValueError(
            f"Notebook canary fixture {path} was refused at other coordinates"
        )
    return FindingKey(path, rule, line, None, cell)


def notebook_canary_findings(
    root: Path,
    python: str,
    base: str,
    paths: tuple[str, ...],
    *,
    timeout: float,
) -> tuple[tuple[str, ...], str, str, tuple[FindingKey, ...]]:
    """Return the canary's rules, the engine's platform and digest, and its keys.

    Parameters
    ----------
    root : Path
        Repository root holding the fixture.
    python : str
        The interpreter with the pinned MyPy that the profile runs.
    base : str
        The resolved trusted base commit, the source of the enabled rules.
    paths : tuple[str, ...]
        Repository-relative fixture files, sorted: :data:`CANARY_NOTEBOOK`
        and at least one :data:`REFUSED_PREFIX` notebook, nothing else.
    timeout : float
        Positive finite whole-run deadline in seconds.

    Returns
    -------
    tuple[tuple[str, ...], str, str, tuple[FindingKey, ...]]
        The enabled rules, the platform, the verified interpreter binary's
        digest and the sorted canary keys.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If the fixture files are not as described, the profile's own
        analysis refuses the canary notebook, or a refused fixture is
        accepted or refused for another reason or at other coordinates.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Native notebook canary deadline must be positive and finite")
    deadline = time.monotonic() + timeout
    canary = tuple(path for path in paths if Path(path).name == CANARY_NOTEBOOK)
    refused = tuple(
        path
        for path in paths
        if Path(path).name.startswith(REFUSED_PREFIX) and path.endswith(".ipynb.json")
    )
    if len(canary) != 1 or not refused or len(paths) != 1 + len(refused):
        raise ValueError("Invalid notebook canary fixture files")
    rules = enabled_rules(root, base)
    snapshot, platform, engine = analyse_notebooks(
        root, python, lambda _root: canary, timeout=_remaining(deadline)
    )
    keys = [finding_key(finding) for finding in snapshot.findings]
    keys.extend(_advisory_keys(canary[0], snapshot.advisories))
    keys.extend(
        _refused_key(root, python, path, _remaining(deadline)) for path in refused
    )
    return rules, platform, engine, tuple(sorted(keys))
