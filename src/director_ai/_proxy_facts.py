# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Proxy Facts-File Loading

"""Facts-file loading into the ground-truth store, with root confinement.

``facts_root`` confines the resolved ``facts_path`` (symlinks followed)
to an allowed directory for deployments where the path comes from
untrusted configuration.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from director_ai.core import GroundTruthStore


def _load_facts(
    store: GroundTruthStore,
    path: str,
    *,
    facts_root: str | None = None,
) -> None:
    try:
        resolved = pathlib.Path(path).resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Facts file not found: {path}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"Facts file not found: {path}")
    if facts_root is not None:
        try:
            root_resolved = pathlib.Path(facts_root).resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"facts_root not found: {facts_root}") from exc
        if not root_resolved.is_dir():
            raise ValueError(f"facts_root must be a directory: {facts_root}")
        if not resolved.is_relative_to(root_resolved):
            raise ValueError(
                f"facts_path {resolved} is outside facts_root {root_resolved}"
            )
    with open(resolved, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, value = line.partition(":")
                store.add(key.strip(), value.strip())
            else:
                store.add(line[:30], line)
