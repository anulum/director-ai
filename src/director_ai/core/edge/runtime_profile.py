# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — edge runtime readiness profile

"""Readiness contract for low-latency edge and mobile runtime paths.

The profile is deliberately evidence-oriented. It does not build models or
WASM bundles; it records which tracked contracts, source surfaces, build
artefacts, and smoke-test evidence exist in the current checkout. That keeps
the production boundary honest for browser, Worker, embedded, and local
low-latency deployments.
"""

from __future__ import annotations

import importlib
import importlib.util
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

EDGE_RUNTIME_READINESS_SCHEMA_VERSION = "director-ai.edge-runtime-readiness.v1"
RUST_ACCELERATOR_SYMBOLS = (
    "BackfireConfig",
    "RustCoherenceScorer",
    "RustStreamingKernel",
)

CheckStatus = Literal["pass", "pending", "missing"]


@dataclass(frozen=True)
class EdgeRuntimeCheck:
    """One auditable readiness check for an edge runtime surface."""

    name: str
    status: CheckStatus
    summary: str
    evidence: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        """Return whether this check is satisfied."""
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        """Serialise the check to stable JSON-safe data."""
        return {
            "name": self.name,
            "status": self.status,
            "passed": self.passed,
            "summary": self.summary,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class EdgeRuntimeReadiness:
    """Readiness report for one tracked edge/mobile runtime target."""

    schema_version: str
    target_id: str
    runtime: str
    wasm_target: str
    ready_for_local_trial: bool
    ready_for_release: bool
    checks: tuple[EdgeRuntimeCheck, ...]
    findings: tuple[str, ...]
    limits: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the readiness report to stable JSON-safe data."""
        return {
            "schema_version": self.schema_version,
            "target_id": self.target_id,
            "runtime": self.runtime,
            "wasm_target": self.wasm_target,
            "ready_for_local_trial": self.ready_for_local_trial,
            "ready_for_release": self.ready_for_release,
            "checks": [check.to_dict() for check in self.checks],
            "findings": list(self.findings),
            "limits": dict(self.limits),
        }

    def check_map(self) -> dict[str, EdgeRuntimeCheck]:
        """Return checks keyed by stable check name."""
        return {check.name: check for check in self.checks}


def probe_backfire_kernel_symbols() -> tuple[str, ...]:
    """Return available Python symbols from the Rust accelerator package."""
    if importlib.util.find_spec("backfire_kernel") is None:
        return ()
    module = importlib.import_module("backfire_kernel")
    return tuple(
        symbol for symbol in RUST_ACCELERATOR_SYMBOLS if hasattr(module, symbol)
    )


def build_edge_runtime_readiness(
    repo_root: str | Path,
    *,
    target_id: str = "browser-worker",
    quantised_model_path: str | Path | None = None,
    browser_smoke_evidence: str | Path | None = None,
    mobile_smoke_evidence: str | Path | None = None,
    import_probe: bool = True,
) -> EdgeRuntimeReadiness:
    """Build a deterministic readiness profile for one edge runtime target."""
    repo = Path(repo_root).resolve()
    wasm_plan = _load_toml(repo / "requirements" / "wasm_release_plan.toml")
    onnx_plan = _load_toml(repo / "requirements" / "onnx_wheel_targets.toml")
    pyproject = _load_toml(repo / "pyproject.toml")
    target = _target_from_plan(wasm_plan, target_id)
    runtime = str(target.get("runtime", "")) if target else ""
    wasm_target = str(target.get("target", "")) if target else ""

    checks = (
        _check_wasm_plan(repo, wasm_plan),
        _check_wasm_target(target_id, target),
        _check_wasm_source_contract(repo, wasm_plan),
        _check_wasm_build_artefact(repo),
        _check_quantised_nli_contract(repo, pyproject, onnx_plan),
        _check_quantised_model_artefact(repo, quantised_model_path),
        _check_rust_source_contract(repo),
        _check_rust_python_accelerator(import_probe=import_probe),
        _check_edge_deployment_docs(repo),
        _check_latency_benchmark_surface(repo),
        _check_browser_smoke_evidence(repo, browser_smoke_evidence),
        _check_mobile_smoke_evidence(repo, mobile_smoke_evidence),
    )
    check_map = {check.name: check for check in checks}
    local_trial_checks = (
        "wasm_release_plan",
        "wasm_target_matrix",
        "wasm_source_contract",
        "quantised_nli_contract",
        "rust_kernel_source_contract",
        "edge_deployment_docs",
        "latency_benchmark_surface",
    )
    release_checks = local_trial_checks + (
        "wasm_build_artefact",
        "quantised_model_artefact",
        "rust_python_accelerator",
        "browser_worker_smoke",
        "mobile_device_smoke",
    )
    ready_for_local_trial = all(check_map[name].passed for name in local_trial_checks)
    ready_for_release = all(check_map[name].passed for name in release_checks)
    findings = tuple(
        f"{check.name}:{check.status}:{check.summary}"
        for check in checks
        if not check.passed
    )
    limits = {
        "local_only": True,
        "actual_wasm_build_included": check_map["wasm_build_artefact"].passed,
        "quantised_model_artefact_included": check_map[
            "quantised_model_artefact"
        ].passed,
        "rust_python_import_included": check_map["rust_python_accelerator"].passed,
        "browser_worker_smoke_included": check_map["browser_worker_smoke"].passed,
        "mobile_device_smoke_included": check_map["mobile_device_smoke"].passed,
        "package_publish_included": False,
    }
    return EdgeRuntimeReadiness(
        schema_version=EDGE_RUNTIME_READINESS_SCHEMA_VERSION,
        target_id=target_id,
        runtime=runtime,
        wasm_target=wasm_target,
        ready_for_local_trial=ready_for_local_trial,
        ready_for_release=ready_for_release,
        checks=checks,
        findings=findings,
        limits=limits,
    )


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _target_from_plan(plan: dict[str, Any], target_id: str) -> dict[str, Any]:
    for target in plan.get("targets", []):
        if str(target.get("id", "")) == target_id:
            return dict(target)
    return {}


def _check_wasm_plan(repo: Path, plan: dict[str, Any]) -> EdgeRuntimeCheck:
    plan_path = repo / "requirements" / "wasm_release_plan.toml"
    policy = plan.get("policy", {})
    active_phases = {
        str(phase.get("id", ""))
        for phase in plan.get("phases", [])
        if phase.get("status") == "active"
    }
    required = (
        plan_path.exists(),
        policy.get("crate") == "backfire-kernel/crates/backfire-wasm",
        "test-wasm" in policy.get("make_targets", []),
        "wasm-build" in policy.get("make_targets", []),
        "p0-local-artefact" in active_phases,
    )
    if all(required):
        return EdgeRuntimeCheck(
            "wasm_release_plan",
            "pass",
            "tracked WASM release plan declares active local artefact phase",
            (_rel(repo, plan_path), str(policy.get("workflow", ""))),
        )
    return EdgeRuntimeCheck(
        "wasm_release_plan",
        "missing",
        "WASM release plan is absent or incomplete",
        (_rel(repo, plan_path),),
    )


def _check_wasm_target(
    target_id: str,
    target: dict[str, Any],
) -> EdgeRuntimeCheck:
    if target and target.get("target"):
        return EdgeRuntimeCheck(
            "wasm_target_matrix",
            "pass",
            f"{target_id} is declared for {target['target']}",
            (str(target.get("runtime", "")), str(target.get("priority", ""))),
        )
    return EdgeRuntimeCheck(
        "wasm_target_matrix",
        "missing",
        f"{target_id} is not declared in the WASM target matrix",
        (),
    )


def _check_wasm_source_contract(
    repo: Path,
    plan: dict[str, Any],
) -> EdgeRuntimeCheck:
    crate = repo / str(
        plan.get("policy", {}).get("crate", "backfire-kernel/crates/backfire-wasm")
    )
    required = (
        crate / "Cargo.toml",
        crate / "src" / "lib.rs",
        crate / "tests" / "kernel.rs",
        crate / "README.md",
        crate / "example" / "index.html",
    )
    if all(path.exists() for path in required):
        return EdgeRuntimeCheck(
            "wasm_source_contract",
            "pass",
            "WASM halt kernel source, tests, README, and browser example are present",
            tuple(_rel(repo, path) for path in required),
        )
    missing = tuple(_rel(repo, path) for path in required if not path.exists())
    return EdgeRuntimeCheck(
        "wasm_source_contract",
        "missing",
        "WASM halt kernel source contract is incomplete",
        missing,
    )


def _check_wasm_build_artefact(repo: Path) -> EdgeRuntimeCheck:
    pkg = repo / "backfire-kernel" / "crates" / "backfire-wasm" / "pkg"
    required = (
        pkg / "backfire_wasm_bg.wasm",
        pkg / "backfire_wasm.js",
        pkg / "package.json",
    )
    if all(path.exists() for path in required):
        return EdgeRuntimeCheck(
            "wasm_build_artefact",
            "pass",
            "local wasm-pack build artefacts are present",
            tuple(_rel(repo, path) for path in required),
        )
    present = tuple(_rel(repo, path) for path in required if path.exists())
    return EdgeRuntimeCheck(
        "wasm_build_artefact",
        "pending",
        "local wasm-pack build artefacts are not present in this checkout",
        present,
    )


def _check_quantised_nli_contract(
    repo: Path,
    pyproject: dict[str, Any],
    onnx_plan: dict[str, Any],
) -> EdgeRuntimeCheck:
    optional = pyproject.get("project", {}).get("optional-dependencies", {})
    required_paths = (
        repo / "docs-site" / "guide" / "quantization.md",
        repo / "docs-site" / "deployment" / "onnx-artefacts.md",
        repo / "requirements" / "onnx_wheel_targets.toml",
    )
    has_extras = "quantize" in optional and "onnx" in optional
    has_cpu_target = any(
        target.get("extra") == "onnx"
        and target.get("provider") == "CPUExecutionProvider"
        for target in onnx_plan.get("targets", [])
    )
    if has_extras and has_cpu_target and all(path.exists() for path in required_paths):
        return EdgeRuntimeCheck(
            "quantised_nli_contract",
            "pass",
            "quantised and ONNX NLI deployment contracts are tracked",
            tuple(_rel(repo, path) for path in required_paths)
            + ("extras:quantize", "extras:onnx"),
        )
    return EdgeRuntimeCheck(
        "quantised_nli_contract",
        "missing",
        "quantised or ONNX NLI deployment contract is incomplete",
        tuple(_rel(repo, path) for path in required_paths if path.exists()),
    )


def _check_quantised_model_artefact(
    repo: Path,
    quantised_model_path: str | Path | None,
) -> EdgeRuntimeCheck:
    if quantised_model_path is None:
        return EdgeRuntimeCheck(
            "quantised_model_artefact",
            "pending",
            "no quantised model artefact path was supplied",
            ("MODELS/lite-scorer-v2/onnx/model_quantized.onnx",),
        )
    model_path = Path(quantised_model_path)
    if not model_path.is_absolute():
        model_path = repo / model_path
    if model_path.exists() and model_path.is_file():
        return EdgeRuntimeCheck(
            "quantised_model_artefact",
            "pass",
            "quantised model artefact exists",
            (_rel(repo, model_path),),
        )
    return EdgeRuntimeCheck(
        "quantised_model_artefact",
        "pending",
        "quantised model artefact is not present",
        (_rel(repo, model_path),),
    )


def _check_rust_source_contract(repo: Path) -> EdgeRuntimeCheck:
    required = (
        repo / "backfire-kernel" / "Cargo.toml",
        repo / "backfire-kernel" / "crates" / "backfire-core" / "src" / "kernel.rs",
        repo / "backfire-kernel" / "crates" / "backfire-ffi" / "src" / "lib.rs",
        repo / "requirements" / "backfire_kernel_release.toml",
    )
    if all(path.exists() for path in required):
        return EdgeRuntimeCheck(
            "rust_kernel_source_contract",
            "pass",
            "Rust halt kernel and PyO3 wrapper sources are present",
            tuple(_rel(repo, path) for path in required),
        )
    return EdgeRuntimeCheck(
        "rust_kernel_source_contract",
        "missing",
        "Rust halt kernel source contract is incomplete",
        tuple(_rel(repo, path) for path in required if path.exists()),
    )


def _check_rust_python_accelerator(*, import_probe: bool) -> EdgeRuntimeCheck:
    if not import_probe:
        return EdgeRuntimeCheck(
            "rust_python_accelerator",
            "pending",
            "backfire_kernel import probe was not run",
            RUST_ACCELERATOR_SYMBOLS,
        )
    symbols = probe_backfire_kernel_symbols()
    if set(symbols) == set(RUST_ACCELERATOR_SYMBOLS):
        return EdgeRuntimeCheck(
            "rust_python_accelerator",
            "pass",
            "backfire_kernel exposes the expected Python accelerator symbols",
            symbols,
        )
    return EdgeRuntimeCheck(
        "rust_python_accelerator",
        "pending",
        "backfire_kernel is not importable or does not expose every accelerator symbol",
        symbols,
    )


def _check_edge_deployment_docs(repo: Path) -> EdgeRuntimeCheck:
    docs = (
        repo / "docs-site" / "deployment" / "wasm-runtime.md",
        repo / "docs-site" / "deployment" / "onnx-artefacts.md",
        repo / "docs-site" / "guide" / "runtime-boundaries.md",
    )
    if all(path.exists() for path in docs):
        wasm_text = docs[0].read_text(encoding="utf-8")
        onnx_text = docs[1].read_text(encoding="utf-8")
        if "WasmStreamingKernel" in wasm_text and "DIRECTOR_ONNX_PATH" in onnx_text:
            return EdgeRuntimeCheck(
                "edge_deployment_docs",
                "pass",
                "deployment docs cover WASM halt and ONNX scorer boundaries",
                tuple(_rel(repo, path) for path in docs),
            )
    return EdgeRuntimeCheck(
        "edge_deployment_docs",
        "missing",
        "edge deployment docs do not cover both WASM halt and ONNX scoring",
        tuple(_rel(repo, path) for path in docs if path.exists()),
    )


def _check_latency_benchmark_surface(repo: Path) -> EdgeRuntimeCheck:
    benches = (
        repo / "benchmarks" / "wasm_overhead_bench.py",
        repo / "benchmarks" / "latency_bench.py",
    )
    if all(path.exists() for path in benches):
        return EdgeRuntimeCheck(
            "latency_benchmark_surface",
            "pass",
            "latency benchmark scripts exist for WASM/Rust overhead and NLI scoring",
            tuple(_rel(repo, path) for path in benches),
        )
    return EdgeRuntimeCheck(
        "latency_benchmark_surface",
        "missing",
        "latency benchmark scripts are incomplete",
        tuple(_rel(repo, path) for path in benches if path.exists()),
    )


def _check_browser_smoke_evidence(
    repo: Path,
    browser_smoke_evidence: str | Path | None,
) -> EdgeRuntimeCheck:
    return _check_optional_evidence(
        repo,
        name="browser_worker_smoke",
        evidence_path=browser_smoke_evidence,
        pending_summary="browser or Web Worker smoke evidence was not supplied",
        success_summary="browser or Web Worker smoke evidence exists",
    )


def _check_mobile_smoke_evidence(
    repo: Path,
    mobile_smoke_evidence: str | Path | None,
) -> EdgeRuntimeCheck:
    return _check_optional_evidence(
        repo,
        name="mobile_device_smoke",
        evidence_path=mobile_smoke_evidence,
        pending_summary="mobile or embedded-device smoke evidence was not supplied",
        success_summary="mobile or embedded-device smoke evidence exists",
    )


def _check_optional_evidence(
    repo: Path,
    *,
    name: str,
    evidence_path: str | Path | None,
    pending_summary: str,
    success_summary: str,
) -> EdgeRuntimeCheck:
    if evidence_path is None:
        return EdgeRuntimeCheck(name, "pending", pending_summary, ())
    path = Path(evidence_path)
    if not path.is_absolute():
        path = repo / path
    if path.exists() and path.is_file():
        return EdgeRuntimeCheck(name, "pass", success_summary, (_rel(repo, path),))
    return EdgeRuntimeCheck(name, "pending", pending_summary, (_rel(repo, path),))


def _rel(repo: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo).as_posix()
    except ValueError:
        return "external path not serialised"
