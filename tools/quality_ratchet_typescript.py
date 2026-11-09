# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — complete native TypeScript documentation snapshot adapter
"""Bind actual TypeScript documentation findings to the full maintained source inventory."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

from tools.ci_workflow_inventory import object_mapping
from tools.quality_ratchet_contracts import (
    SCHEMA,
    Snapshot,
    decode_quality_json,
    parse_snapshot,
    repository_path,
)
from tools.quality_ratchet_inventory import hash_sources, source_candidate, source_path

NODE_VERSION = "26.13.0"
TYPESCRIPT_VERSION = "6.0.3"
TYPESCRIPT_ROOTS = (
    "gateway",
    "packages",
    "studio-web",
    "tools",
    "scripts",
    "benchmarks",
    "examples",
    "notebooks",
    "training",
    "demo",
    "docs",
)
TYPESCRIPT_SUFFIXES = frozenset(
    {".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs"}
)
TYPESCRIPT_HELPER = "packages/vercel-ai/tools/quality-ratchet.mjs"
TYPESCRIPT_ENGINE_INPUTS = (
    "tools/quality_ratchet_typescript.py",
    "tools/quality_ratchet_contracts.py",
    "tools/quality_ratchet_inventory.py",
    "tools/ci_workflow_inventory.py",
    TYPESCRIPT_HELPER,
    "packages/vercel-ai/package.json",
    "packages/vercel-ai/package-lock.json",
)
PARSER_CONFIGURATION = {
    "allowJs": True,
    "noEmit": True,
    "noLib": True,
    "noResolve": True,
    "jsx": 1,
    "target": 99,
}


def _remaining_seconds(deadline: float) -> float:
    """Return the time left in the caller's shared monotonic budget."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ValueError("Native TypeScript documentation exhausted its whole deadline")
    return remaining


def typescript_sources(root: Path, *, deadline: float | None = None) -> tuple[str, ...]:
    """Inventory all maintained Git-visible JavaScript and TypeScript sources.

    Parameters
    ----------
    root : Path
        Actual canonical Git source root.
    deadline : float | None
        Shared absolute monotonic deadline. Standalone inventories have a
        twenty-second policy ceiling, independent of host capacity.

    Returns
    -------
    tuple[str, ...]
        Complete sorted inventory, including tests, tooling and declarations.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If the Git root, source scope or regular nonempty inventory cannot qualify.
    """
    if deadline is None:
        deadline = time.monotonic() + 20.0
    if not math.isfinite(deadline):
        raise ValueError("Native TypeScript inventory deadline must be finite")
    root = root.resolve(strict=True)
    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
        timeout=_remaining_seconds(deadline),
    )
    if Path(top.stdout.strip()).resolve() != root:
        raise ValueError("TypeScript inventory requires the actual Git root")
    original = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
        timeout=_remaining_seconds(deadline),
    )
    paths = set()
    for raw in original.stdout.split(b"\0"):
        if not raw:
            continue
        if not any(
            raw.endswith(suffix.encode("ascii")) for suffix in TYPESCRIPT_SUFFIXES
        ):
            continue
        relative = repository_path(raw.decode("utf-8"))
        if Path(relative).parts[0] not in TYPESCRIPT_ROOTS:
            raise ValueError(
                "TypeScript source lies outside the declared quality scope"
            )
        candidate = source_candidate(root, relative)
        if not candidate.exists():
            continue
        source_path(root, relative)
        paths.add(relative)
    if not paths:
        raise ValueError("Native TypeScript source inventory is empty")
    _remaining_seconds(deadline)
    return tuple(sorted(paths))


def capture_typescript_documentation(
    root: Path,
    node: str,
    *,
    timeout: float = 120.0,
) -> Snapshot:
    """Measure source documentation with the actual pinned TypeScript compiler.

    Parameters
    ----------
    root : Path
        Actual selected Git source repository.
    node : str
        Pinned native Node executable; no installation is performed.
    timeout : float
        Positive finite whole-analysis deadline in seconds.

    Returns
    -------
    Snapshot
        Complete source-linked native declaration and masking findings.
        This parser profile does not claim semantic typing or import qualification.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If source, configuration, native tools or complete analysis cannot qualify.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Native quality deadline must be positive and finite")
    deadline = time.monotonic() + timeout
    root = root.resolve(strict=True)
    engine_root = Path(__file__).resolve().parents[1]
    helper = source_path(engine_root, TYPESCRIPT_HELPER)
    engine_sources = hash_sources(engine_root, TYPESCRIPT_ENGINE_INPUTS)

    def run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        """Execute the real native process within its remaining whole deadline."""
        return subprocess.run(
            [node, *arguments],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
            timeout=_remaining_seconds(deadline),
        )

    version = run(["--version"])
    metadata = run([str(helper), "--version"])
    expected = {
        "node": NODE_VERSION,
        "typescript": TYPESCRIPT_VERSION,
        "profile": "typescript-docs",
        "compilerOptions": PARSER_CONFIGURATION,
    }
    if (
        version.returncode != 0
        or version.stderr
        or version.stdout.strip() != "v" + NODE_VERSION
        or metadata.returncode != 0
        or metadata.stderr
        or json.dumps(decode_quality_json(metadata.stdout.encode()), sort_keys=True)
        != json.dumps(expected, sort_keys=True)
    ):
        raise ValueError(
            "Native Node or TypeScript profile differs from the qualified recipe"
        )
    paths = typescript_sources(root, deadline=deadline)
    sources = hash_sources(root, paths)
    native = run([str(helper), str(root), *paths])
    if native.returncode not in (0, 1) or native.stderr:
        raise ValueError("Native TypeScript parser did not completely analyse source")
    report = object_mapping(decode_quality_json(native.stdout.encode()))
    if (
        set(report) != {"schema", "complete", "sources", "findings"}
        or report["schema"] != "quality-ratchet.ts-ast.v1"
        or report["complete"] is not True
        or report["sources"] != sources
        or not isinstance(report["findings"], list)
    ):
        raise ValueError(
            "Native TypeScript report is not bound to complete source bytes"
        )
    findings = []
    for raw in report["findings"]:
        item = object_mapping(raw)
        if set(item) != {"path", "rule", "line", "column", "message"} or item.get(
            "rule"
        ) not in {"declaration-documentation", "source-directive"}:
            raise ValueError("Invalid original native TypeScript documentation finding")
        findings.append({**item, "cell": None})
    if bool(findings) != bool(native.returncode):
        raise ValueError("Native TypeScript status disagrees with its findings")
    if (
        typescript_sources(root, deadline=deadline) != paths
        or hash_sources(root, paths) != sources
        or hash_sources(engine_root, TYPESCRIPT_ENGINE_INPUTS) != engine_sources
    ):
        raise ValueError("TypeScript source or engine changed during native analysis")
    identity = hashlib.sha256(
        json.dumps(
            {
                "profile": "typescript-docs",
                "schema": SCHEMA,
                "native": expected,
                "engine": engine_sources,
                "roots": TYPESCRIPT_ROOTS,
                "suffixes": sorted(TYPESCRIPT_SUFFIXES),
            },
            sort_keys=True,
        ).encode(),
    ).hexdigest()
    snapshot = parse_snapshot(
        {
            "schema": SCHEMA,
            "profile": "typescript-docs",
            "identity": identity,
            "sources": sources,
            "findings": findings,
            "complete": True,
            "advisories": [],
        }
    )
    _remaining_seconds(deadline)
    return snapshot
