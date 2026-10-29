# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — complete native Go documentation snapshot adapter
"""Combine native Revive conventions with actual Go declaration and directive analysis."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path

from tools.quality_ratchet_contracts import (
    SCHEMA,
    Snapshot,
    parse_snapshot,
    repository_path,
    snapshot_payload,
)
from tools.quality_ratchet_golang_protocol import (
    GO_VERSION as GO_VERSION,
)
from tools.quality_ratchet_golang_protocol import (
    REVIVE_VERSION as REVIVE_VERSION,
)
from tools.quality_ratchet_golang_protocol import (
    parse_go_parser_report,
    parse_go_tool_versions,
    parse_revive_report,
    qualify_go_build_report,
)
from tools.quality_ratchet_inventory import hash_sources, source_candidate, source_path

GO_ROOTS = ("gateway", "tools")
GENERATED_GO = (
    "gateway/go/proto/director/v1/director.pb.go",
    "gateway/go/proto/director/v1/director_grpc.pb.go",
)
GO_HELPER = ("main.go", "documentation.go", "go.mod")
GO_CONFIGURATIONS = ("gateway/go/go.mod", "gateway/go/go.sum")
REVIVE_CONFIGURATION = (
    'ignore-generated-header = true\nconfidence = 0.0\nseverity = "warning"\n'
    "[rule.exported]\n"
    'arguments = ["check-private-receivers", "check-public-interface"]\n'
    "[rule.package-comments]\n"
)


def go_sources(root: Path, *, deadline: float | None = None) -> tuple[str, ...]:
    """Inventory every maintained Git-visible Go source in its real repository.

    Parameters
    ----------
    root : Path
        Exact selected Git source root.
    deadline : float, optional
        Absolute monotonic deadline shared with the native collector. Standalone
        inventory calls retain a finite twenty-second limit per Git operation.

    Returns
    -------
    tuple[str, ...]
        Complete sorted regular-source scope, including native tests and tooling.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If scope, source, Git root or a nonempty inventory cannot qualify.
    """
    root = root.resolve(strict=True)
    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
        timeout=20 if deadline is None else min(20, _remaining(deadline)),
    )
    if Path(top.stdout.strip()).resolve() != root:
        raise ValueError("Go inventory requires the actual Git root")
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
        timeout=20 if deadline is None else min(20, _remaining(deadline)),
    )
    paths = set()
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        if not raw.endswith(b".go"):
            continue
        path = repository_path(raw.decode("utf-8"))
        if path in GENERATED_GO:
            continue
        if Path(path).parts[0] not in GO_ROOTS:
            raise ValueError("Go source lies outside the declared quality scope")
        candidate = source_candidate(root, path)
        if not candidate.exists():
            continue
        source_path(root, path)
        paths.add(path)
    if not paths:
        raise ValueError("Native Go source inventory is empty")
    return tuple(sorted(paths))


def _remaining(deadline: float) -> float:
    """Require finite remaining time before native execution and final publication."""
    remaining = deadline - time.monotonic()
    if not math.isfinite(remaining) or remaining <= 0:
        raise ValueError("Native Go documentation exhausted its whole deadline")
    return remaining


def _run(
    argv: list[str],
    root: Path,
    deadline: float,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    """Run one original native command within the remaining whole deadline."""
    return subprocess.run(
        argv,
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=_remaining(deadline),
    )


def capture_go_documentation(
    root: Path,
    go: str,
    revive: str,
    *,
    timeout: float = 120.0,
) -> Snapshot:
    """Check every maintained Go declaration using actual native source parsers.

    Parameters
    ----------
    root : Path
        Actual selected Git source root.
    go : str
        Native pinned Go executable; no installation occurs.
    revive : str
        Actual pinned Revive executable.
    timeout : float
        Positive finite whole-analysis deadline in seconds.

    Returns
    -------
    Snapshot
        Complete original source-bound declaration and convention diagnostics.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If native tools, source, configuration, scope or completion cannot qualify.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Native quality deadline must be positive and finite")
    deadline = time.monotonic() + timeout
    root = root.resolve(strict=True)
    environment = dict(os.environ)
    environment["GOTOOLCHAIN"] = "local"
    environment["GOFLAGS"] = "-mod=readonly"
    version = _run([go, "version"], root, deadline, environment)
    lint_version = _run([revive, "-version"], root, deadline, environment)
    native_version = parse_go_tool_versions(version, lint_version)
    paths = go_sources(root, deadline=deadline)
    sources = hash_sources(root, paths)
    helper = Path(__file__).resolve().parent / "quality_ratchet_go"
    helper_hashes = hash_sources(helper, GO_HELPER)
    modules = GO_CONFIGURATIONS
    configuration = hash_sources(root, modules)
    identity = hashlib.sha256(
        json.dumps(
            {
                "profile": "go-docs",
                "snapshot_schema": SCHEMA,
                "go": native_version,
                "revive": REVIVE_VERSION,
                "helper": helper_hashes,
                "configuration": configuration,
                "revive_configuration": REVIVE_CONFIGURATION,
                "roots": GO_ROOTS,
                "generated": GENERATED_GO,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    with tempfile.TemporaryDirectory(prefix="quality-go-parser-") as directory:
        binary = Path(directory) / "documentation"
        build = _run(
            [
                go,
                "build",
                "-o",
                str(binary),
                str(helper / "main.go"),
                str(helper / "documentation.go"),
            ],
            root,
            deadline,
            environment,
        )
        qualify_go_build_report(build)
        native = _run([str(binary), "--", *paths], root, deadline, environment)
    findings = list(parse_go_parser_report(native, sources, identity))
    with tempfile.TemporaryDirectory(prefix="quality-go-") as directory:
        config = Path(directory) / "revive.toml"
        config.write_text(REVIVE_CONFIGURATION)
        lint = _run(
            [
                revive,
                "-config",
                str(config),
                "-formatter",
                "json",
                "-set_exit_status",
                *paths,
            ],
            root,
            deadline,
            environment,
        )
        findings.extend(parse_revive_report(lint, sources, identity))
    if (
        go_sources(root, deadline=deadline) != paths
        or hash_sources(root, paths) != sources
        or hash_sources(root, modules) != configuration
        or hash_sources(helper, GO_HELPER) != helper_hashes
    ):
        raise ValueError("Go source or configuration changed during native analysis")
    _remaining(deadline)
    return parse_snapshot(
        snapshot_payload(
            Snapshot("go-docs", identity, sources, tuple(findings), True, ())
        )
    )
