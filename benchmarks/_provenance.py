# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Benchmark artefact reproducibility provenance

"""Reproducibility provenance for benchmark artefacts.

A published benchmark number is only reproducible when a reader can (1) tie the
artefact back to the exact source commit that produced it and (2) re-derive the
headline aggregate from the individual per-sample scores. An artefact that
carries neither — the RAGTruth result flagged in the 2026-07-17 red-team review
recorded ``catch 49.3 %`` with no commit and no per-sample rows, so the number
could not be checked when a later run measured ``25.4 %`` — is a data-integrity
liability, not evidence.

This module centralises that contract so every release-grade benchmark writer
uses one canonical implementation instead of an ad-hoc inline ``git rev-parse``:

* :func:`resolve_git_sha` / :func:`working_tree_dirty` capture the source state,
  degrading to ``"unknown"`` / ``None`` outside a git checkout rather than
  crashing a long GPU run;
* :func:`provenance_block` assembles the full reproducibility envelope;
* :func:`stamp` attaches it to an artefact payload;
* :func:`assert_reproducible` is the fail-closed gate — it refuses to certify a
  release artefact that omits the resolved commit or the per-sample rows, so the
  defect cannot silently recur.
"""

from __future__ import annotations

import platform
import shutil
import subprocess  # nosec B404 — fixed git argv, no shell, bounded timeout
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ``benchmarks/`` sits directly under the repository root; resolve once so the
# git queries run against the checkout regardless of the caller's cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]

#: Sentinel recorded when the source commit cannot be resolved.
UNKNOWN_SHA = "unknown"

#: Payload keys that hold genuine *per-sample* rows (individual scores from which
#: the aggregate can be re-derived). Aggregate breakdowns such as ``per_task`` or
#: ``matrix`` are deliberately excluded — they summarise, they do not reproduce.
DEFAULT_SAMPLE_KEYS: tuple[str, ...] = ("rows", "per_sample", "samples")


class ProvenanceError(RuntimeError):
    """Raised when a benchmark artefact cannot be certified reproducible."""


def _git(args: Sequence[str], root: Path) -> str | None:
    """Run ``git <args>`` in *root*, returning stdout or ``None`` on any failure.

    Every failure mode — git absent, not a repository, timeout, non-zero exit —
    collapses to ``None`` so provenance capture never aborts the benchmark it is
    meant to annotate.
    """
    git = shutil.which("git")
    if not git:
        return None
    try:
        completed = subprocess.run(  # nosec B603 — trusted argv, no shell
            [git, *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(root),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def resolve_git_sha(root: Path | None = None) -> str:
    """Return the full HEAD commit SHA, or :data:`UNKNOWN_SHA` outside a checkout.

    Parameters
    ----------
    root:
        Directory to resolve the commit against. Defaults to the repository root
        so a script invoked from any working directory still records the commit.

    Returns
    -------
    str
        The 40-character HEAD SHA, or ``"unknown"`` when git is unavailable or
        the directory is not a git working tree.
    """
    return _git(["rev-parse", "HEAD"], root or _REPO_ROOT) or UNKNOWN_SHA


def working_tree_dirty(root: Path | None = None) -> bool | None:
    """Return whether the working tree has uncommitted changes.

    Returns
    -------
    bool | None
        ``True`` / ``False`` when git can report status, ``None`` when the state
        is unknowable (git absent or not a repository) — a distinct third value
        so callers never mistake "unknown" for "clean".
    """
    status = _git(["status", "--porcelain"], root or _REPO_ROOT)
    if status is None:
        return None
    return bool(status)


def provenance_block(
    root: Path | None = None, *, git_sha: str | None = None
) -> dict[str, Any]:
    """Assemble the reproducibility envelope for the current run.

    Parameters
    ----------
    root:
        Directory to resolve git state against; defaults to the repository root.
    git_sha:
        Explicit commit to record instead of resolving ``HEAD`` — used by remote
        runners that stamp the exact commit they cloned.

    Returns
    -------
    dict
        ``git_sha``, ``git_short``, ``git_dirty``, ``python``, ``platform`` and
        an ISO-8601 UTC ``generated_at`` timestamp.
    """
    root = root or _REPO_ROOT
    sha = git_sha or resolve_git_sha(root)
    return {
        "git_sha": sha,
        "git_short": sha[:12] if sha != UNKNOWN_SHA else UNKNOWN_SHA,
        "git_dirty": working_tree_dirty(root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "generated_at": datetime.now(UTC).isoformat(),
    }


def stamp(
    payload: dict[str, Any], root: Path | None = None, *, git_sha: str | None = None
) -> dict[str, Any]:
    """Attach a ``provenance`` block to *payload* in place and return it.

    Parameters
    ----------
    payload:
        Artefact mapping to annotate; a ``provenance`` key is set (or replaced).
    root, git_sha:
        Forwarded to :func:`provenance_block`.

    Returns
    -------
    dict
        The same *payload* object, for fluent use at a ``return`` site.
    """
    payload["provenance"] = provenance_block(root, git_sha=git_sha)
    return payload


def _has_per_sample_rows(
    payload: Mapping[str, Any], sample_keys: Sequence[str]
) -> bool:
    """Return whether *payload* carries a non-empty per-sample collection."""
    for key in sample_keys:
        value = payload.get(key)
        if (
            isinstance(value, (Sequence, Mapping))
            and not isinstance(value, (str, bytes))
            and len(value) > 0
        ):
            return True
    return False


def assert_reproducible(
    payload: Mapping[str, Any],
    *,
    sample_keys: Sequence[str] = DEFAULT_SAMPLE_KEYS,
    require_clean: bool = False,
) -> None:
    """Fail closed unless *payload* is a reproducible release artefact.

    A reproducible artefact must (1) carry a resolved source commit under
    ``provenance.git_sha`` (not :data:`UNKNOWN_SHA`) and (2) include at least one
    non-empty per-sample collection so the headline aggregate can be
    independently re-derived. With *require_clean*, a dirty working tree also
    fails — appropriate for a tagged release where the tree must match a commit.

    Parameters
    ----------
    payload:
        Artefact mapping, already :func:`stamp`-ed.
    sample_keys:
        Keys under which per-sample rows may appear.
    require_clean:
        When true, reject artefacts generated from a dirty working tree.

    Raises
    ------
    ProvenanceError
        If the provenance block, resolved commit, clean-tree requirement, or
        per-sample rows are missing.
    """
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ProvenanceError("artefact is missing a 'provenance' block")
    sha = provenance.get("git_sha")
    if not sha or sha == UNKNOWN_SHA:
        raise ProvenanceError(
            "artefact provenance has no resolved git_sha — the source commit is "
            "required for reproducibility (run inside the repo or pass --git-sha)"
        )
    if require_clean and provenance.get("git_dirty"):
        raise ProvenanceError("artefact was generated from a dirty working tree")
    if not _has_per_sample_rows(payload, sample_keys):
        raise ProvenanceError(
            "artefact carries no per-sample rows so the aggregate cannot be "
            f"re-derived (looked for: {', '.join(sample_keys)})"
        )
