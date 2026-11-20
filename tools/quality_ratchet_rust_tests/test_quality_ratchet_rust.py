# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — real native Rust documentation ratchet contracts
"""Exercise original Rust source, generated registrations and public capture/check."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools import quality_ratchet_rust
from tools.quality_ratchet_rust import (
    RUST_CONFIGURATIONS,
    RUST_HELPER,
    capture_rust_documentation,
    rust_sources,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def rust_tools() -> tuple[str, str]:
    """Resolve required native compiler tools without installing or skipping.

    Returns
    -------
    tuple[str, str]
        Actual Cargo and Rust compiler executables.
    """
    cargo = os.environ.get("DIRECTOR_QUALITY_CARGO") or shutil.which("cargo")
    rustc = os.environ.get("DIRECTOR_QUALITY_RUSTC") or shutil.which("rustc")
    assert cargo is not None and rustc is not None, "Cargo and Rustc are required"
    return cargo, rustc


@pytest.fixture(scope="session")
def rust_reader(
    tmp_path_factory: pytest.TempPathFactory, rust_tools: tuple[str, str]
) -> Path:
    """Build the actual production reader once for real process contract tests.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Native runner's owned isolated allocation.
    rust_tools : tuple[str, str]
        Actual installed compiler executables.

    Returns
    -------
    Path
        Real compiled production CLI, without substituted parser outputs.
    """
    directory = tmp_path_factory.mktemp("native-rust-reader")
    cargo, rustc = rust_tools
    environment = dict(os.environ)
    environment["RUSTC"] = rustc
    result = subprocess.run(
        [
            cargo,
            "build",
            "--locked",
            "--offline",
            "--quiet",
            "--manifest-path",
            str(ROOT / "tools/quality_ratchet_rust/Cargo.toml"),
            "--target-dir",
            str(directory),
            "--jobs",
            "2",
        ],
        env=environment,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0 and not result.stdout and not result.stderr
    return directory / "debug/director-quality-rust"


@pytest.fixture
def rust_repository(tmp_path: Path) -> Path:
    """Create an actual Git source object with immutable native Cargo configurations.

    Parameters
    ----------
    tmp_path : Path
        Native runner's isolated owned directory.

    Returns
    -------
    Path
        Original committed source and actual copied repository configuration.
    """
    root = tmp_path / "rust-quality"
    root.mkdir()
    for relative in RUST_CONFIGURATIONS:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    source = root / "backfire-kernel/crates/backfire-core/src/lib.rs"
    source.parent.mkdir(parents=True)
    source.write_text(
        "//! Source module.\n/// Return the current count.\nfn value() -> u8 { 1 }\n"
    )
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "add", *RUST_CONFIGURATIONS, str(source.relative_to(root))],
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Native fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-m",
            "original native Rust sources",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return root


@pytest.mark.parametrize(
    "attack", ("subdirectory", "workspace", "outside", "symlink", "empty")
)
def test_native_inventory_refuses_unqualified_original_scope(
    rust_repository: Path, tmp_path: Path, attack: str
) -> None:
    """Refuse real root, source, dependency and original-file scope changes."""
    root = rust_repository
    if attack == "subdirectory":
        root = root / "backfire-kernel"
    elif attack == "workspace":
        manifest = root / "backfire-kernel/Cargo.toml"
        manifest.write_text(
            manifest.read_text().replace('"vendor/bulletproofs"', '"vendor/other"')
        )
    elif attack == "outside":
        (root / "unqualified.rs").write_text("//! Original outside source.\n")
    elif attack == "symlink":
        outside = tmp_path / "external.rs"
        outside.write_text("//! Original outside source.\n")
        (root / "backfire-kernel/linked.rs").symlink_to(outside)
    else:
        (root / "backfire-kernel/crates/backfire-core/src/lib.rs").unlink()
    with pytest.raises(ValueError):
        rust_sources(root)


def test_native_inventory_retains_survivors_and_actual_vendor_boundary(
    rust_repository: Path,
) -> None:
    """Inventory maintained native source while handling a real tracked deletion."""
    original = rust_repository / "backfire-kernel/crates/backfire-core/src/lib.rs"
    original.unlink()
    survivor = rust_repository / "tools/retained.rs"
    survivor.parent.mkdir()
    survivor.write_text("//! Retained authored module.\n")
    vendor = rust_repository / "backfire-kernel/vendor/bulletproofs/unknown.rs"
    vendor.parent.mkdir(parents=True)
    vendor.write_text("dependency owned invalid syntax\n")
    assert rust_sources(rust_repository) == ("tools/retained.rs",)


@pytest.mark.parametrize("attack", ("criterion-version", "criterion-lock", "benchmark"))
def test_public_capture_requires_original_generated_recipe(
    rust_repository: Path, rust_tools: tuple[str, str], attack: str
) -> None:
    """Refuse actual generated benchmark dependency and harness configuration drift."""
    if attack == "criterion-version":
        manifest = rust_repository / "backfire-kernel/Cargo.toml"
        manifest.write_text(
            manifest.read_text().replace('version = "0.5"', 'version = "0.6"')
        )
    elif attack == "criterion-lock":
        lock = rust_repository / "backfire-kernel/Cargo.lock"
        lock.write_text(
            lock.read_text().replace('version = "0.5.1"', 'version = "0.5.2"')
        )
    else:
        manifest = rust_repository / "backfire-kernel/crates/backfire-core/Cargo.toml"
        manifest.write_text(
            manifest.read_text().replace("harness = false", "harness = true")
        )
    with pytest.raises(ValueError):
        capture_rust_documentation(rust_repository, *rust_tools)


@pytest.mark.parametrize("deadline", (0.0, -1.0, float("inf"), float("nan"), 1e-12))
def test_public_capture_requires_positive_finite_remaining_deadline(
    rust_repository: Path, rust_tools: tuple[str, str], deadline: float
) -> None:
    """Refuse actual invalid or exhausted whole-analysis time bounds before native work."""
    with pytest.raises(ValueError):
        capture_rust_documentation(rust_repository, *rust_tools, timeout=deadline)


@pytest.mark.parametrize("missing", ("cargo", "rustc"))
def test_public_capture_refuses_missing_native_executables(
    rust_repository: Path, rust_tools: tuple[str, str], missing: str
) -> None:
    """Require both actual native executable identities instead of a clean missing profile."""
    cargo, rustc = rust_tools
    with pytest.raises(ValueError):
        capture_rust_documentation(
            rust_repository,
            "unavailable-native-cargo" if missing == "cargo" else cargo,
            "unavailable-native-rustc" if missing == "rustc" else rustc,
        )


@pytest.mark.parametrize("wrong", ("cargo", "rustc"))
def test_public_capture_refuses_actual_unqualified_executable(
    rust_repository: Path, rust_tools: tuple[str, str], wrong: str
) -> None:
    """Refuse actual executable output that does not prove the pinned native toolchain."""
    cargo, rustc = rust_tools
    with pytest.raises(ValueError):
        capture_rust_documentation(
            rust_repository,
            sys.executable if wrong == "cargo" else cargo,
            sys.executable if wrong == "rustc" else rustc,
        )


def test_public_capture_discards_caller_compiler_wrappers_and_flags(
    rust_repository: Path, rust_tools: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Build the real reader despite invalid caller-provided compiler overrides."""
    for key in (
        "RUSTFLAGS",
        "CARGO_ENCODED_RUSTFLAGS",
        "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER",
        "CARGO_BUILD_RUSTC",
    ):
        monkeypatch.setenv(key, "unavailable-caller-compiler")
    monkeypatch.setenv("CARGO_BUILD_TARGET", "wasm32-unknown-unknown")
    monkeypatch.setenv("CARGO_BUILD_RUSTFLAGS", "--unqualified-option")
    snapshot = capture_rust_documentation(rust_repository, *rust_tools)
    assert snapshot.complete and snapshot.findings == ()


@pytest.mark.parametrize("location", ("source", "parent", "cache"))
@pytest.mark.parametrize("name", ("config", "config.toml"))
def test_public_capture_refuses_ambient_cargo_configuration(
    rust_repository: Path,
    rust_tools: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
    location: str,
    name: str,
) -> None:
    """Refuse ordinary Cargo configuration in each supported lookup location."""
    cache = rust_repository.parent / "cargo-cache"
    monkeypatch.setenv("CARGO_HOME", os.path.relpath(cache, rust_repository))
    directory = {
        "source": rust_repository / ".cargo",
        "parent": rust_repository.parent / ".cargo",
        "cache": cache,
    }[location]
    directory.mkdir()
    (directory / name).write_text("[build]\njobs = 1\n", encoding="utf-8")

    with pytest.raises(
        ValueError, match="Ambient Cargo configuration requires qualification"
    ):
        capture_rust_documentation(rust_repository, *rust_tools)


@pytest.mark.parametrize(
    "attack",
    (
        "syntax",
        "warning",
        "inventory",
        "configuration",
        "cargo-configuration",
        "helper",
        "adapter",
        "decoder",
        "toolchain",
        "status",
    ),
)
def test_public_capture_requires_complete_original_process_and_source_identity(
    rust_repository: Path,
    rust_tools: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    """Refuse genuine native build failures and actual changes after native execution."""
    isolated = tmp_path / "actual-reader"
    isolated.mkdir()
    for relative in ("quality_ratchet_rust.py", "quality_ratchet_rust_declarations.py"):
        (isolated / relative).write_bytes((ROOT / "tools" / relative).read_bytes())
    for relative in RUST_HELPER:
        target = isolated / "quality_ratchet_rust" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            (ROOT / "tools/quality_ratchet_rust" / relative).read_bytes()
        )
    main = isolated / "quality_ratchet_rust/src/main.rs"
    if attack == "syntax":
        main.write_text(main.read_text() + "\nfn incomplete(\n")
    elif attack == "warning":
        main.write_text(
            main.read_text()
            + "\n/// Deliberate native warning.\nfn unused_native_warning() {}\n"
        )
    cargo, rustc = rust_tools
    if attack == "toolchain":
        copied_cargo = isolated / "cargo"
        shutil.copyfile(cargo, copied_cargo)
        copied_cargo.chmod(0o700)
        cargo = str(copied_cargo)
    monkeypatch.setattr(
        quality_ratchet_rust, "__file__", str(isolated / "quality_ratchet_rust.py")
    )
    original_run = quality_ratchet_rust._run
    processes: list[subprocess.CompletedProcess[str]] = []

    def custody(
        argv: list[str], root: Path, environment: dict[str, str], deadline: float
    ) -> subprocess.CompletedProcess[str]:
        """Retain original native output unchanged before an actual owned-file mutation."""
        result = original_run(argv, root, environment, deadline)
        processes.append(result)
        record = tmp_path / f"original-process-{len(processes)}.json"
        record.write_text(
            json.dumps(
                {
                    "argv": argv,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )
        )
        if len(argv) > 1 and argv[1] == "build" and attack == "status":
            binary = (
                Path(argv[argv.index("--target-dir") + 1])
                / "debug/director-quality-rust"
            )
            original_binary = binary.with_name("director-quality-rust-original")
            binary.rename(original_binary)
            binary.write_text(
                "#!" + sys.executable + " -B\n"
                '"""Preserve actual native output while corrupting its transport exit status."""\n'
                "import sys,subprocess,json,base64\nfrom pathlib import Path\n"
                "result=subprocess.run(["
                + repr(str(original_binary))
                + ",*sys.argv[1:]],capture_output=True,timeout=20)\n"
                "Path("
                + repr(str(tmp_path / "original-inner-reader.json"))
                + ").write_text(json.dumps({'returncode':result.returncode,'stdout_b64':base64.b64encode(result.stdout).decode(),'stderr_b64':base64.b64encode(result.stderr).decode()}))\n"
                "sys.stdout.buffer.write(result.stdout);sys.stderr.buffer.write(result.stderr)\n"
                "raise SystemExit(1)\n"
            )
            binary.chmod(0o700)
        if Path(argv[0]).name == "director-quality-rust":
            if attack == "inventory":
                added = rust_repository / "tools/new.rs"
                added.parent.mkdir()
                added.write_text("//! Actual late source addition.\n")
            elif attack == "configuration":
                config = rust_repository / RUST_CONFIGURATIONS[0]
                config.write_text(
                    config.read_text() + "\n# Actual late configuration change.\n"
                )
            elif attack == "cargo-configuration":
                config = rust_repository / ".cargo/config.toml"
                config.parent.mkdir()
                config.write_text("[build]\njobs = 1\n", encoding="utf-8")
            elif attack in ("helper", "adapter", "decoder"):
                changed = {
                    "helper": main,
                    "adapter": isolated / "quality_ratchet_rust.py",
                    "decoder": isolated / "quality_ratchet_rust_declarations.py",
                }[attack]
                changed.write_text(changed.read_text() + "\n")
            elif attack == "toolchain":
                with Path(cargo).open("ab") as stream:
                    stream.write(b"\0")
        return result

    monkeypatch.setattr(quality_ratchet_rust, "_run", custody)
    message = (
        "Ambient Cargo configuration requires qualification"
        if attack == "cargo-configuration"
        else None
    )
    with pytest.raises(ValueError, match=message):
        capture_rust_documentation(rust_repository, cargo, rustc)
    assert processes
    if attack == "syntax":
        assert processes[-1].returncode != 0 and processes[-1].stderr
    elif attack == "warning":
        assert processes[-1].returncode == 0 and processes[-1].stderr
    elif attack == "status":
        assert processes[-1].returncode == 1 and not processes[-1].stderr
        original = json.loads((tmp_path / "original-inner-reader.json").read_text())
        assert (
            original["returncode"] == 0
            and original["stdout_b64"]
            and not original["stderr_b64"]
        )
    else:
        assert all(result.returncode == 0 and not result.stderr for result in processes)


def test_public_capture_keeps_the_production_rust_corpus(
    rust_tools: tuple[str, str],
) -> None:
    """Analyse the actual maintained crate roots, module files and benchmarks."""
    snapshot = capture_rust_documentation(ROOT, *rust_tools, timeout=180)
    assert snapshot.complete
    assert set(snapshot.sources) == set(rust_sources(ROOT))
    assert {
        "backfire-kernel/crates/backfire-core/src/lib.rs",
        "backfire-kernel/crates/backfire-core/src/kernel.rs",
        "backfire-kernel/crates/backfire-core/benches/scoring_bench.rs",
        "backfire-kernel/crates/backfire-ssgf/benches/ssgf_bench.rs",
        "tools/quality_ratchet_rust/src/main.rs",
    } <= set(snapshot.sources)


def test_public_capture_refuses_unresolved_native_macro_provenance(
    rust_repository: Path, rust_tools: tuple[str, str]
) -> None:
    """Expose a real native provenance refusal at the maintained capture API."""
    root = rust_repository / "backfire-kernel/crates/backfire-core/src"
    (root / "bridge.rs").write_text(
        "//! Module exporting a macro from an external dependency.\n"
        "pub use external_package::recipe as serde_json;\n"
    )
    source = root / "lib.rs"
    source.write_text(
        "//! Root declaration.\n"
        "/// External macro namespace.\nmod bridge;\n"
        "use self::bridge::*;\n"
        "/// Use the unresolved native macro.\n"
        "fn read() { serde_json::json!(); }\n"
    )
    with pytest.raises(ValueError, match="did not completely analyse original source"):
        capture_rust_documentation(rust_repository, *rust_tools)
