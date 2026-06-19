# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — edge runtime readiness tests

from __future__ import annotations

import json
from pathlib import Path

from director_ai.core.edge import runtime_profile
from director_ai.core.edge.runtime_profile import build_edge_runtime_readiness


def _write(path: Path, text: str = "ok\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_minimal_repo(repo: Path) -> None:
    _write(
        repo / "requirements" / "wasm_release_plan.toml",
        """
[policy]
crate = "backfire-kernel/crates/backfire-wasm"
workflow = ".github/workflows/wheels.yml"
make_targets = ["test-wasm", "wasm-build"]
package_name = "backfire-wasm"
scope = "halt decision only; host supplies one coherence score per token"

[[phases]]
id = "p0-local-artefact"
status = "active"
goal = "Ship browser artefacts."
required_gates = ["wasm-pack build"]

[[targets]]
id = "browser-worker"
runtime = "Browser Web Worker"
target = "web"
priority = "p0"
""",
    )
    _write(
        repo / "requirements" / "onnx_wheel_targets.toml",
        """
[policy]
doc = "docs-site/deployment/onnx-artefacts.md"

[[targets]]
id = "linux-cpu-x86_64"
extra = "onnx"
provider = "CPUExecutionProvider"
""",
    )
    _write(
        repo / "pyproject.toml",
        """
[project]
name = "director-ai"
version = "0.0.0"

[project.optional-dependencies]
onnx = ["onnxruntime"]
quantize = ["bitsandbytes"]
""",
    )
    for path in (
        "backfire-kernel/Cargo.toml",
        "backfire-kernel/crates/backfire-core/src/kernel.rs",
        "backfire-kernel/crates/backfire-ffi/src/lib.rs",
        "backfire-kernel/crates/backfire-wasm/Cargo.toml",
        "backfire-kernel/crates/backfire-wasm/src/lib.rs",
        "backfire-kernel/crates/backfire-wasm/tests/kernel.rs",
        "backfire-kernel/crates/backfire-wasm/README.md",
        "backfire-kernel/crates/backfire-wasm/example/index.html",
        "requirements/backfire_kernel_release.toml",
        "benchmarks/wasm_overhead_bench.py",
        "benchmarks/latency_bench.py",
    ):
        _write(repo / path)
    _write(
        repo / "docs-site" / "deployment" / "wasm-runtime.md",
        "WasmStreamingKernel\n",
    )
    _write(
        repo / "docs-site" / "deployment" / "onnx-artefacts.md",
        "DIRECTOR_ONNX_PATH\n",
    )
    _write(repo / "docs-site" / "guide" / "runtime-boundaries.md")
    _write(repo / "docs-site" / "guide" / "quantization.md")


def test_readiness_separates_local_trial_from_release_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    _write_minimal_repo(tmp_path)
    monkeypatch.setattr(
        runtime_profile,
        "probe_backfire_kernel_symbols",
        lambda: runtime_profile.RUST_ACCELERATOR_SYMBOLS,
    )

    profile = build_edge_runtime_readiness(tmp_path)
    checks = profile.check_map()

    assert profile.ready_for_local_trial is True
    assert profile.ready_for_release is False
    assert checks["wasm_build_artefact"].status == "pending"
    assert checks["quantised_model_artefact"].status == "pending"
    assert checks["browser_worker_smoke"].status == "pending"
    assert checks["mobile_device_smoke"].status == "pending"
    assert profile.limits["actual_wasm_build_included"] is False
    assert profile.limits["quantised_model_artefact_included"] is False


def test_readiness_marks_release_ready_when_artifacts_and_smokes_exist(
    tmp_path,
    monkeypatch,
) -> None:
    _write_minimal_repo(tmp_path)
    monkeypatch.setattr(
        runtime_profile,
        "probe_backfire_kernel_symbols",
        lambda: runtime_profile.RUST_ACCELERATOR_SYMBOLS,
    )
    for path in (
        "backfire-kernel/crates/backfire-wasm/pkg/backfire_wasm_bg.wasm",
        "backfire-kernel/crates/backfire-wasm/pkg/backfire_wasm.js",
        "backfire-kernel/crates/backfire-wasm/pkg/package.json",
        "MODELS/lite-scorer-v2/onnx/model_quantized.onnx",
        "benchmarks/results/browser-worker-smoke.json",
        "benchmarks/results/mobile-smoke.json",
    ):
        _write(tmp_path / path)

    profile = build_edge_runtime_readiness(
        tmp_path,
        quantised_model_path="MODELS/lite-scorer-v2/onnx/model_quantized.onnx",
        browser_smoke_evidence="benchmarks/results/browser-worker-smoke.json",
        mobile_smoke_evidence="benchmarks/results/mobile-smoke.json",
    )

    assert profile.ready_for_local_trial is True
    assert profile.ready_for_release is True
    assert not profile.findings
    assert profile.limits["actual_wasm_build_included"] is True
    assert profile.limits["mobile_device_smoke_included"] is True


def test_readiness_rejects_unknown_target(tmp_path) -> None:
    _write_minimal_repo(tmp_path)

    profile = build_edge_runtime_readiness(
        tmp_path,
        target_id="untracked-target",
        import_probe=False,
    )

    assert profile.ready_for_local_trial is False
    assert profile.check_map()["wasm_target_matrix"].status == "missing"
    assert "untracked-target" in profile.findings[0]


def test_readiness_reports_empty_checkout_as_not_trial_ready(tmp_path) -> None:
    profile = build_edge_runtime_readiness(tmp_path, import_probe=False)
    checks = profile.check_map()

    assert profile.ready_for_local_trial is False
    assert checks["wasm_release_plan"].status == "missing"
    assert checks["wasm_source_contract"].status == "missing"
    assert checks["rust_kernel_source_contract"].status == "missing"
    assert checks["edge_deployment_docs"].status == "missing"
    assert checks["latency_benchmark_surface"].status == "missing"


def test_readiness_records_incomplete_source_docs_and_benchmarks(tmp_path) -> None:
    _write_minimal_repo(tmp_path)
    (tmp_path / "backfire-kernel" / "crates" / "backfire-wasm" / "README.md").unlink()
    (
        tmp_path / "backfire-kernel" / "crates" / "backfire-core" / "src" / "kernel.rs"
    ).unlink()
    (tmp_path / "docs-site" / "deployment" / "wasm-runtime.md").write_text(
        "runtime boundary pending\n",
        encoding="utf-8",
    )
    (tmp_path / "benchmarks" / "latency_bench.py").unlink()

    profile = build_edge_runtime_readiness(tmp_path, import_probe=False)
    checks = profile.check_map()

    assert checks["wasm_source_contract"].status == "missing"
    assert (
        "backfire-kernel/crates/backfire-wasm/README.md"
        in checks["wasm_source_contract"].evidence
    )
    assert checks["rust_kernel_source_contract"].status == "missing"
    assert checks["edge_deployment_docs"].status == "missing"
    assert checks["latency_benchmark_surface"].status == "missing"


def test_readiness_records_missing_quantisation_contract(tmp_path) -> None:
    _write_minimal_repo(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "director-ai"
version = "0.0.0"

[project.optional-dependencies]
onnx = ["onnxruntime"]
""",
        encoding="utf-8",
    )

    profile = build_edge_runtime_readiness(tmp_path, import_probe=False)

    assert profile.ready_for_local_trial is False
    assert profile.check_map()["quantised_nli_contract"].status == "missing"


def test_probe_backfire_kernel_symbols_reports_available_accelerators(
    monkeypatch,
) -> None:
    class PartialBackfireKernel:
        BackfireConfig = object()
        RustStreamingKernel = object()

    monkeypatch.setattr(
        runtime_profile.importlib.util,
        "find_spec",
        lambda name: object() if name == "backfire_kernel" else None,
    )
    monkeypatch.setattr(
        runtime_profile.importlib,
        "import_module",
        lambda name: PartialBackfireKernel,
    )

    symbols = runtime_profile.probe_backfire_kernel_symbols()

    assert symbols == ("BackfireConfig", "RustStreamingKernel")


def test_probe_backfire_kernel_symbols_returns_empty_when_package_missing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runtime_profile.importlib.util,
        "find_spec",
        lambda name: None,
    )

    assert runtime_profile.probe_backfire_kernel_symbols() == ()


def test_readiness_records_partial_accelerator_and_missing_smoke_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    _write_minimal_repo(tmp_path)
    monkeypatch.setattr(
        runtime_profile,
        "probe_backfire_kernel_symbols",
        lambda: ("BackfireConfig",),
    )

    profile = build_edge_runtime_readiness(
        tmp_path,
        browser_smoke_evidence="benchmarks/results/missing-browser-smoke.json",
    )
    checks = profile.check_map()

    assert checks["rust_python_accelerator"].status == "pending"
    assert checks["rust_python_accelerator"].evidence == ("BackfireConfig",)
    assert checks["browser_worker_smoke"].status == "pending"
    assert checks["browser_worker_smoke"].evidence == (
        "benchmarks/results/missing-browser-smoke.json",
    )


def test_readiness_accepts_absolute_smoke_evidence(tmp_path) -> None:
    _write_minimal_repo(tmp_path)
    smoke = tmp_path / "absolute-smoke.json"
    _write(smoke, "{}\n")

    profile = build_edge_runtime_readiness(
        tmp_path,
        mobile_smoke_evidence=smoke,
        import_probe=False,
    )

    assert profile.check_map()["mobile_device_smoke"].status == "pass"
    assert profile.check_map()["mobile_device_smoke"].evidence == (
        "absolute-smoke.json",
    )


def test_readiness_serialises_external_paths_without_leaking_absolute_path(
    tmp_path,
) -> None:
    _write_minimal_repo(tmp_path)
    external = tmp_path.parent / "outside-model.onnx"

    profile = build_edge_runtime_readiness(
        tmp_path,
        quantised_model_path=external,
        import_probe=False,
    )
    serialised = json.dumps(profile.to_dict(), sort_keys=True)

    assert "external path not serialised" in serialised
    assert str(external) not in serialised
