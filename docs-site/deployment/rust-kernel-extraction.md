<!--
SPDX-License-Identifier: Apache-2.0
Commercial licence available
Concepts 1996-2026 Miroslav Sotek. All rights reserved.
Code 2020-2026 Miroslav Sotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Director-Class AI - Rust kernel extraction plan
-->

# Rust Kernel Extraction

This page turns the runtime extraction decision into an execution plan. The
current decision is conservative: keep `backfire-kernel/` and
`formal/HaltMonitor/` in this repository until the API, wheel, proof, and CI
contracts are fixed by tests.

The tracked source for this plan is
`requirements/rust_kernel_extraction_plan.toml`.

## Execution Status

The v3.15 extraction boundary is active. The release contract is tracked in
`requirements/backfire_kernel_release.toml`; release notes live in
`backfire-kernel/RELEASE_NOTES.md`; and the standalone crate CI entrypoint is
`backfire-kernel/ci/advanced_runtime_ci.sh`.

The current extracted Rust workspace version is `0.1.1`. `director-ai[rust]` accepts
`backfire-kernel>=0.1.0,<0.2`, while the default Python path stays independent
of the Rust wheel.

## Boundary

| Surface | Path | Contract |
|---------|------|----------|
| Rust workspace | `backfire-kernel/` | Owns accelerated halt, streaming, scoring, WASM, and advanced primitives. |
| Python wheel | `backfire-kernel/crates/backfire-ffi/` | Exports `backfire_kernel`; installable only through `director-ai[rust]`. |
| Python fallback | `src/director_ai/` | Must import and pass tests without Rust, WASM, or Lean installed. |
| WASM artefact | `backfire-kernel/crates/backfire-wasm/` | Uses the same halt contract as the Rust core for offline and edge hosts. |
| Proof root | `formal/HaltMonitor/` | Maps Lean theorems to runtime surfaces through `proof_manifest.toml`. |

## Standalone Crate API

| Crate | Public role | Extraction rule |
|-------|-------------|-----------------|
| `backfire-types` | Shared config, score, stream, and result types. | Stable fields; removals require a version bump and fixture update. |
| `backfire-core` | Halt, streaming, scoring, and in-memory knowledge primitives. | Python and Rust fixtures must agree on halt and score semantics. |
| `backfire-ffi` | PyO3 wheel boundary for `backfire_kernel`. | Default Python imports must never require this wheel. |
| `backfire-wasm` | Browser, Worker, and offline halt runtime. | Follows the same halt fixtures as `backfire-core`. |
| `backfire-physics` | Physical and phase-dynamics primitives. | Advanced runtime only; absent from default install. |
| `backfire-observers` | Observer and controller primitives. | Advanced runtime only; absent from default install. |
| `backfire-ssgf` | Stochastic geometry primitives. | Advanced runtime only; absent from default install. |

## Versioning Policy

Rust crates use SemVer. Until the first stable Rust release, breaking API
changes require a new minor version. After that point, breaking API changes
require a new major version.

`director-ai` pins an explicit supported `backfire-kernel` range for
`director-ai[rust]`. The first active external range is `>=0.1.0,<0.2`. The
Python fallback remains the supported default path and must pass without the
Rust package installed.

## Python Wheel Contract

The wheel contract is:

1. The wheel exports `backfire_kernel`.
2. Installing `director-ai` without extras does not install or import the wheel.
3. Installing `director-ai[rust]` installs a supported wheel range.
4. Missing, incompatible, or unloaded Rust wheels fall back to Python semantics.
5. Wheel builds cover Linux x86-64, Linux ARM64, macOS x86-64, macOS ARM64, and
   Windows x64 before extraction.

`.github/workflows/wheels.yml` is the current release workflow for wheels,
source packages, and the WASM artefact.

## Proof Artefact Boundary

`formal/HaltMonitor/proof_manifest.toml` records the theorem boundary:

| Theorem | Runtime claim |
|---------|---------------|
| `run_emitted_preserves_input` | Emitted streams preserve input token order. |
| `run_emitted_implies_all_pass` | Emitted streams contain only threshold-passing items. |
| `run_any_fail_implies_not_emitted` | Any failing score prevents an emitted result. |
| `run_any_fail_implies_halted` | Any failing score forces the halted result. |
| `passes_antitone` | Passing a stricter threshold implies passing any looser one. |
| `run_all_pass_emitted` | If every item passes, the run emits the whole input. |
| `run_threshold_monotone` | Lowering the threshold never turns an emitting stream into a halting one. |
| `CoherenceScore.score_nonneg` | The composite coherence score is never negative. |
| `CoherenceScore.score_le_one` | The composite coherence score never exceeds one. |
| `CoherenceScore.score_mem_unit` | The composite coherence score always lies in `[0,1]`. |
| `clampUnit_mem_unit` | The score clamp lands in `[0,1]` for any input, so the rescaling path cannot emit an out-of-range score. |

The first seven theorems model the streaming-halt loop
(`HaltMonitor.Core`/`Properties`/`Monotonicity`); the last four model the
coherence scorer (`HaltMonitor.CoherenceScore`), giving the abstract `Score` a
machine-checked unit-interval range that mirrors
`CoherenceScorer.calculate_coherence` in `core/scoring/scorer.py`.

The proof root stays outside user installation. It supports release evidence
and contract fixtures; it is not part of the first-run path.

## Release CI

| Gate | Command | Scope |
|------|---------|-------|
| Rust workspace | `cargo test --workspace` | Rust crates |
| Standalone crate CI | `backfire-kernel/ci/advanced_runtime_ci.sh` | Format, FFI check, workspace tests, optional wheel build |
| PyO3 boundary | `cargo check -p backfire-ffi` | Wheel boundary |
| Wheel build | `maturin build --release -m backfire-kernel/crates/backfire-ffi/Cargo.toml` | Python wheel |
| Python optional path | `pytest tests/test_ffi_bindings.py` | Optional Rust path |
| Docs | `mkdocs build --strict` | Deployment docs |
| Plan check | `pytest tests/test_rust_kernel_extraction_plan.py` | Plan and proof boundary |

## Phases

| Phase | Exit gate |
|-------|-----------|
| `p0-contract-freeze` | Plan, docs, proof manifest, and plan tests pass. |
| `p1-crate-split-readiness` | Python and Rust contract tests pass without default-path Rust imports. |
| `p2-release-ci-cutover` | Linux, macOS, Windows, source package, and WASM jobs pass from tags. |
| `p3-repo-extraction` | `director-ai` pins the external range and the Python fallback passes alone. |

## Links

- [Runtime Extraction Evaluation](../guide/runtime-extraction-evaluation.md)
- [Runtime Boundaries](../guide/runtime-boundaries.md)
- [Rust FFI](../guide/rust-ffi.md)
- [WASM Runtime](wasm-runtime.md)
