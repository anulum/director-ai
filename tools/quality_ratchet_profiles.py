# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — native quality profile registry
"""Dispatch the maintained quality profiles to their actual native collectors."""

from __future__ import annotations

import subprocess
from argparse import Namespace
from pathlib import Path

from tools.quality_ratchet_contracts import Snapshot
from tools.quality_ratchet_golang import GO_CONFIGURATIONS, capture_go_documentation
from tools.quality_ratchet_julia_native import (
    JULIA_CONFIGURATIONS,
    capture_julia_documentation,
)
from tools.quality_ratchet_lean import LEAN_CONFIGURATIONS, capture_lean_documentation
from tools.quality_ratchet_native import capture_python_documentation
from tools.quality_ratchet_notebook import capture_notebook_typing
from tools.quality_ratchet_protobuf import capture_protobuf_documentation
from tools.quality_ratchet_rust import RUST_CONFIGURATIONS, capture_rust_documentation
from tools.quality_ratchet_typescript import capture_typescript_documentation
from tools.quality_ratchet_typing import (
    PYTHON_TYPING_CONFIGURATIONS,
    TYPING_CONFIGURATIONS,
    capture_python_typing,
)

PROFILE_CONFIGURATIONS: dict[str, tuple[str, ...]] = {
    "python-docs": ("pyproject.toml", "requirements/ci-lint.txt"),
    "python-types": PYTHON_TYPING_CONFIGURATIONS,
    "notebook-types": TYPING_CONFIGURATIONS,
    "go-docs": GO_CONFIGURATIONS,
    "typescript-docs": (),
    "protobuf-docs": (),
    "lean-docs": LEAN_CONFIGURATIONS,
    "rust-docs": RUST_CONFIGURATIONS,
    "julia-docs": JULIA_CONFIGURATIONS,
}
REQUIRED_PROFILES = tuple(PROFILE_CONFIGURATIONS)


def capture_profile(root: Path, args: Namespace, profile: str) -> Snapshot:
    """Run one native profile using its configured executable and deadline.

    Parameters
    ----------
    root : Path
        Actual Git checkout selected for native source inspection.
    args : argparse.Namespace
        Parsed executable paths and per-profile deadline from the maintained CLI.
    profile : str
        Named member of the maintained native profile registry.

    Returns
    -------
    Snapshot
        Complete source-bound native findings for this profile.
    """
    if profile == "go-docs":
        actual = capture_go_documentation(
            root, args.go, args.revive, timeout=args.timeout
        )
    elif profile == "typescript-docs":
        actual = capture_typescript_documentation(root, args.node, timeout=args.timeout)
    elif profile == "protobuf-docs":
        actual = capture_protobuf_documentation(root, args.protoc, timeout=args.timeout)
    elif profile == "rust-docs":
        actual = capture_rust_documentation(
            root, args.cargo, args.rustc, timeout=args.timeout
        )
    elif profile == "lean-docs":
        actual = capture_lean_documentation(root, args.lean, timeout=args.timeout)
    elif profile == "julia-docs":
        actual = capture_julia_documentation(root, args.julia, timeout=args.timeout)
    elif profile == "notebook-types":
        actual = capture_notebook_typing(root, args.python, timeout=args.timeout)
    elif profile in {"python-docs", "python-types"}:
        capture = (
            capture_python_typing
            if profile == "python-types"
            else capture_python_documentation
        )
        actual = capture(root, args.python, timeout=args.timeout)
    else:
        raise ValueError(f"Unknown native quality profile: {profile}")
    return actual


def capture_required_profiles(root: Path, args: Namespace) -> dict[str, Snapshot]:
    """Capture the full ordered profile set and identify any native refusal.

    Parameters
    ----------
    root : Path
        Actual Git checkout analysed by every required profile.
    args : argparse.Namespace
        Native executable paths and per-profile deadline.

    Returns
    -------
    dict of str and Snapshot
        Complete snapshots keyed by their required profile names.

    Raises
    ------
    ValueError
        If a native process cannot complete, including its profile name.
    """
    profiles: dict[str, Snapshot] = {}
    for name in REQUIRED_PROFILES:
        try:
            profiles[name] = capture_profile(root, args, name)
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            raise ValueError(f"{name}: {error}") from error
    return profiles
