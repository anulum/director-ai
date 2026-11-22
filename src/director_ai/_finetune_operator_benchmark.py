# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — operator-held general benchmark for API fine-tuning jobs

"""The operator's general benchmark for the anti-regression gate of API jobs.

A ``/v1/finetune/start`` request cannot name a general benchmark, so a
caller cannot choose an easy one. The operator configures one on the server
(``DirectorConfig.finetune_general_benchmark_path``). The router checks it
once, at creation, and each job trains against a private copy taken when the
job starts, whose SHA-256 the job's regression report names. Without one,
API jobs keep reporting ``reject``.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

__all__ = ["operator_benchmark", "snapshot_operator_benchmark"]


def operator_benchmark(path: str | Path | None) -> Path | None:
    """Check the configured general benchmark, or return None when unset.

    Parameters
    ----------
    path : str, Path or None
        The configured benchmark; an empty string or None means unset.

    Returns
    -------
    Path or None
        The resolved benchmark file.

    Raises
    ------
    FileNotFoundError
        If the path does not exist or is not a regular file; the benchmark
        loader the job uses refuses it the same way.
    ValueError
        If the file holds no labelled premise and hypothesis rows.
    """
    if path is None or str(path) == "":
        return None
    from director_ai.core.training.finetune_benchmark import _load_benchmark_jsonl

    resolved = Path(path).expanduser().resolve(strict=True)
    if not _load_benchmark_jsonl(resolved):
        raise ValueError(f"general benchmark has no labelled rows: {resolved}")
    return resolved


def snapshot_operator_benchmark(source: Path, destination: Path) -> str:
    """Copy the benchmark for one job and return the copy's SHA-256.

    Parameters
    ----------
    source : Path
        The operator's benchmark.
    destination : Path
        The job's private copy, which the caller removes afterwards.

    Returns
    -------
    str
        The hexadecimal SHA-256 of the copied bytes, the ones the job uses.
    """
    shutil.copyfile(source, destination)
    return hashlib.sha256(destination.read_bytes()).hexdigest()
