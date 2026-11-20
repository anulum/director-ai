# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — complete native Rust documentation snapshot adapter
"""Bind full authored Rust source to native syntax and explicit generated recipes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
import tomllib
from pathlib import Path

from tools.ci_workflow_inventory import object_mapping
from tools.quality_ratchet_contracts import (
    SCHEMA,
    Snapshot,
    parse_snapshot,
    repository_path,
)
from tools.quality_ratchet_inventory import hash_sources, source_path
from tools.quality_ratchet_rust_declarations import rust_documentation_findings

RUST_VERSION = "1.98.1"
RUST_COMMIT = "48a229ceaefd4985c50990b14116b6d856af0985"
RUST_ROOTS = ("backfire-kernel", "tools")
RUST_VENDOR = "backfire-kernel/vendor/bulletproofs/"
RUST_MEMBERS = (
    "backfire-types",
    "backfire-core",
    "backfire-physics",
    "backfire-observers",
    "backfire-ssgf",
    "backfire-ffi",
    "backfire-wasm",
)
RUST_CONFIGURATIONS = (
    "backfire-kernel/Cargo.toml",
    "backfire-kernel/Cargo.lock",
    *(f"backfire-kernel/crates/{name}/Cargo.toml" for name in RUST_MEMBERS),
)
RUST_HELPER = (
    "Cargo.toml",
    "Cargo.lock",
    "src/main.rs",
    "src/declarations.rs",
    "src/attributes.rs",
    "src/macros.rs",
)
RUST_BENCHMARKS = {
    "backfire-kernel/crates/backfire-core/benches/scoring_bench.rs": (
        "backfire-core",
        "scoring_bench",
    ),
    "backfire-kernel/crates/backfire-ssgf/benches/ssgf_bench.rs": (
        "backfire-ssgf",
        "ssgf_bench",
    ),
}


def rust_sources(root: Path) -> tuple[str, ...]:
    """Inventory all maintained original Rust source, including tests and benchmarks.

    Parameters
    ----------
    root : Path
        Actual selected Git repository root.

    Returns
    -------
    tuple[str, ...]
        Complete sorted source inventory without the exact dependency-owned vendor.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If source scope, generated boundary, regular files or Git identity refuse.
    """
    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    if Path(top.stdout.strip()).resolve() != root:
        raise ValueError("Rust inventory requires the actual Git root")
    config = tomllib.loads(source_path(root, RUST_CONFIGURATIONS[0]).read_text())
    workspace = object_mapping(config.get("workspace"))
    if workspace.get("exclude") != ["vendor/bulletproofs"] or workspace.get(
        "members"
    ) != [f"crates/{name}" for name in RUST_MEMBERS]:
        raise ValueError("Rust workspace source/vendor boundaries are not qualified")
    native = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
        timeout=20,
    )
    paths: set[str] = set()
    for raw in native.stdout.split(b"\0"):
        if not raw:
            continue
        relative = repository_path(raw.decode("utf-8"))
        if Path(relative).suffix != ".rs" or relative.startswith(RUST_VENDOR):
            continue
        if Path(relative).parts[0] not in RUST_ROOTS:
            raise ValueError("Rust source lies outside the declared native scope")
        candidate = root / relative
        if not candidate.exists() and not candidate.is_symlink():
            continue
        source_path(root, relative)
        paths.add(relative)
    if not paths:
        raise ValueError("Native Rust source inventory is empty")
    return tuple(sorted(paths))


def _criterion_sources(root: Path) -> frozenset[str]:
    """Bind generated benchmark registrations to actual Cargo declarations and lock."""
    workspace = object_mapping(
        tomllib.loads(source_path(root, RUST_CONFIGURATIONS[0]).read_text()).get(
            "workspace"
        )
    )
    criterion = object_mapping(
        object_mapping(workspace.get("dependencies")).get("criterion")
    )
    lock = tomllib.loads(source_path(root, RUST_CONFIGURATIONS[1]).read_text())
    packages = lock.get("package")
    if (
        criterion.get("version") != "0.5"
        or not isinstance(packages, list)
        or [
            object_mapping(package).get("version")
            for package in packages
            if object_mapping(package).get("name") == "criterion"
        ]
        != ["0.5.1"]
    ):
        raise ValueError("Criterion generated-source version is not qualified")
    for _relative, (member, name) in RUST_BENCHMARKS.items():
        manifest = tomllib.loads(
            source_path(root, f"backfire-kernel/crates/{member}/Cargo.toml").read_text()
        )
        dependency = object_mapping(
            object_mapping(manifest.get("dev-dependencies")).get("criterion")
        )
        benches = manifest.get("bench")
        if (
            dependency != {"workspace": True}
            or not isinstance(benches, list)
            or [
                object_mapping(bench)
                for bench in benches
                if object_mapping(bench).get("name") == name
            ]
            != [{"name": name, "harness": False}]
        ):
            raise ValueError("Criterion benchmark source recipe is not qualified")
    return frozenset(RUST_BENCHMARKS)


def _crate_roots(root: Path) -> frozenset[str]:
    """Resolve library target roots from the actual declared workspace members."""
    roots = {"tools/quality_ratchet_rust/src/main.rs"}
    for relative in RUST_CONFIGURATIONS[2:]:
        project = tomllib.loads(source_path(root, relative).read_text())
        library = object_mapping(project.get("lib", {}))
        source = repository_path(library.get("path", "src/lib.rs"))
        roots.add((Path(relative).parent / source).as_posix())
    return frozenset(roots)


def _run(
    argv: list[str],
    root: Path,
    environment: dict[str, str],
    deadline: float,
) -> subprocess.CompletedProcess[str]:
    """Run an original native process within the remaining whole-analysis deadline."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ValueError("Native Rust documentation exhausted its whole deadline")
    return subprocess.run(
        argv,
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=remaining,
    )


def _require_clean_cargo_configuration(root: Path, cargo_home: Path) -> None:
    """Refuse Cargo configuration outside the pinned reader recipe."""
    locations = (
        cargo_home,
        *((directory / ".cargo") for directory in (root, *root.parents)),
    )
    for directory in locations:
        for name in ("config", "config.toml"):
            path = directory / name
            try:
                path.lstat()
            except FileNotFoundError:
                continue
            raise ValueError(
                f"Ambient Cargo configuration requires qualification: {path}"
            )


def capture_rust_documentation(
    root: Path, cargo: str, rustc: str, *, timeout: float = 120.0
) -> Snapshot:
    """Analyse every original authored Rust declaration through the pinned native reader.

    Parameters
    ----------
    root : Path
        Actual selected Git source root.
    cargo : str
        Installed pinned Cargo executable; dependencies must already be cached.
    rustc : str
        Installed pinned compiler executable, used explicitly for the real build.
    timeout : float
        Positive finite whole-analysis deadline in seconds.

    Returns
    -------
    Snapshot
        Complete source-bound native documentation and directive findings.
        Columns preserve one-based native UTF8 character coordinates.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If versions, original source, generated recipes, native completion or
        configuration fail qualification.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Native quality deadline must be positive and finite")
    deadline = time.monotonic() + timeout
    root = root.resolve(strict=True)
    paths = rust_sources(root)
    benchmarks = _criterion_sources(root)
    crate_roots = _crate_roots(root)
    sources = hash_sources(root, paths)
    configuration = hash_sources(root, RUST_CONFIGURATIONS)
    helper = Path(__file__).resolve().parent / "quality_ratchet_rust"
    helper_hashes = hash_sources(helper, RUST_HELPER)
    adapter_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    decoder = Path(__file__).with_name("quality_ratchet_rust_declarations.py")
    decoder_hash = hashlib.sha256(decoder.read_bytes()).hexdigest()
    configured_home = os.environ.get("CARGO_HOME")
    cargo_home = (
        root / configured_home
        if configured_home is not None
        else Path.home() / ".cargo"
    ).resolve()
    _require_clean_cargo_configuration(root, cargo_home)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("CARGO_", "RUSTC_", "RUSTDOC_"))
        and key not in {"CARGO", "RUSTC", "RUSTFLAGS", "RUSTDOC", "RUSTDOCFLAGS"}
    }
    environment["CARGO_HOME"] = str(cargo_home)
    cargo_path, rustc_path = shutil.which(cargo), shutil.which(rustc)
    if cargo_path is None or rustc_path is None:
        raise ValueError("Native Rust executables are unavailable")
    environment["RUSTC"] = rustc_path
    version = _run([rustc_path, "--version", "--verbose"], root, environment, deadline)
    cargo_version = _run([cargo_path, "--version"], root, environment, deadline)
    if (
        version.returncode
        or version.stderr
        or not version.stdout.startswith(f"rustc {RUST_VERSION} ")
        or f"commit-hash: {RUST_COMMIT}\n" not in version.stdout
        or cargo_version.returncode
        or cargo_version.stderr
        or not cargo_version.stdout.startswith(f"cargo {RUST_VERSION} ")
    ):
        raise ValueError("Native Rust toolchain differs from the qualified profile")
    executables = {
        name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for name, path in (("cargo", cargo_path), ("rustc", rustc_path))
    }
    identity = hashlib.sha256(
        json.dumps(
            {
                "profile": "rust-docs",
                "snapshot_schema": SCHEMA,
                "rustc": version.stdout,
                "cargo": cargo_version.stdout,
                "executables": executables,
                "helper": helper_hashes,
                "adapter": adapter_hash,
                "decoder": decoder_hash,
                "configuration": configuration,
                "roots": RUST_ROOTS,
                "vendor": RUST_VENDOR,
                "criterion_benchmarks": sorted(benchmarks),
                "crate_roots": sorted(crate_roots),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    with tempfile.TemporaryDirectory(prefix="quality-rust-parser-") as temporary:
        target = Path(temporary)
        environment["CARGO_TARGET_DIR"] = str(target)
        build = _run(
            [
                cargo_path,
                "build",
                "--locked",
                "--offline",
                "--quiet",
                "--manifest-path",
                str(helper / "Cargo.toml"),
                "--target-dir",
                str(target),
                "--jobs",
                "2",
            ],
            root,
            environment,
            deadline,
        )
        if build.returncode or build.stdout or build.stderr:
            raise ValueError("Native Rust declaration reader did not build cleanly")
        argv = [str(target / "debug/director-quality-rust")]
        for path in paths:
            option = "--source"
            if path in benchmarks:
                option = "--criterion-source"
            elif path in crate_roots:
                option = "--crate-root"
            argv.extend([option, path])
        native = _run(argv, root, environment, deadline)
    if native.returncode not in (0, 1) or native.stderr:
        raise ValueError(
            "Native Rust reader did not completely analyse original source"
        )
    _require_clean_cargo_configuration(root, cargo_home)
    findings = rust_documentation_findings(native.stdout.encode(), sources, root)
    if bool(findings) != bool(native.returncode):
        raise ValueError("Native Rust status disagrees with original findings")
    if (
        rust_sources(root) != paths
        or hash_sources(root, paths) != sources
        or hash_sources(root, RUST_CONFIGURATIONS) != configuration
        or hash_sources(helper, RUST_HELPER) != helper_hashes
        or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != adapter_hash
        or hashlib.sha256(decoder.read_bytes()).hexdigest() != decoder_hash
        or {
            name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in (("cargo", cargo_path), ("rustc", rustc_path))
        }
        != executables
    ):
        raise ValueError("Rust source, reader, toolchain or configuration changed")
    return parse_snapshot(
        {
            "schema": SCHEMA,
            "profile": "rust-docs",
            "identity": identity,
            "sources": sources,
            "findings": findings,
            "complete": True,
            "advisories": [],
        }
    )
