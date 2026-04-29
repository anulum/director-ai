<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- Director-Class AI - Runtime Extraction Evaluation -->

# Runtime Extraction Evaluation

This note records the v3.15 evaluation for whether the Rust kernel and formal
proof artefacts should move out of the core repository.

## Current State

| Component | Current Location | Current Public Surface |
|-----------|------------------|------------------------|
| Rust workspace | `backfire-kernel/` | Optional `director-ai[rust]` extra and `backfire-kernel` package |
| PyO3 bindings | `backfire-kernel/crates/backfire-ffi/` | `backfire_kernel` Python module |
| WASM crate | `backfire-kernel/crates/backfire-wasm/` | Deferred edge runtime artefact |
| Lean halt proof | `formal/HaltMonitor/` | Machine-checked model for halt-loop safety claims |
| Python fallback | `src/director_ai/` | Required supported default path |

The Python package already treats Rust as optional at runtime. The support
burden remains high because the monorepo CI, contributor docs, and release
surface still expose Rust, WASM, and Lean as first-class repo concerns.

## Decision

Do not physically split the repositories during v3.15. First create a stable
package boundary and extraction contract. A direct split before that boundary
exists would move complexity into release coordination without reducing user
setup risk.

The supported v3.15 boundary is:

| Boundary | Decision |
|----------|----------|
| User default path | Python + FastAPI + local Chroma only |
| Rust kernel | Optional acceleration package, installable only through explicit extra |
| Lean proofs | Research/formal verification artefact, not required for user install |
| CI | Keep Rust CI visible, but classify it as advanced runtime CI |
| Release | Publish Rust/WASM artefacts independently only after contract tests pass |
| Docs | Keep advanced runtime docs out of first-run setup flow |

## Extraction Contract

Before moving `backfire-kernel/` or `formal/` to a separate repository, these
contracts must be stable:

| Contract | Requirement |
|----------|-------------|
| Python API | `director-ai` imports and test suite pass without Rust, WASM, or Lean installed |
| Optional install | `pip install director-ai[rust]` installs a versioned `backfire-kernel` wheel |
| Fallback behavior | Every Rust-backed scorer path has a Python fallback with equivalent public semantics |
| Test vectors | Shared JSON fixtures cover scores, halt decisions, injection verdicts, and BM25 outputs |
| Release compatibility | `director-ai` pins a supported `backfire-kernel` range and rejects incompatible versions |
| Proof traceability | Formal models cite the Python function and Rust function they model |
| CI ownership | Core CI can run without building Rust; advanced runtime CI validates the Rust workspace |

## Recommended Extraction Plan

1. Keep the source in this repository for v3.15 while hiding it from onboarding
   docs and first-run setup.
2. Add cross-language contract tests that compare Python and Rust results using
   shared fixtures.
3. Move Rust and WASM release jobs behind an advanced runtime workflow label.
4. Add a proof manifest under `formal/` that maps each theorem to the production
   function and contract fixture it supports.
5. After two releases with stable contract fixtures, extract the Rust workspace
   and formal models into separately versioned packages or repositories.

## Go / No-Go

| Signal | Go | No-Go |
|--------|----|-------|
| User setup | Default install never invokes Rust or Lean | Any quickstart failure depends on Rust, WASM, or Lean |
| Packaging | Wheels exist for target platforms | Users must compile Rust for the common path |
| Semantics | Contract fixtures pass in Python and Rust | Rust fallback differs from Python semantics |
| Proofs | Theorem manifest maps to production behavior | Proofs drift from implementation with no traceability |
| Maintenance | Advanced runtime issues can be triaged separately | Core contributors must debug Rust/Lean to fix Python docs or server issues |

## Outcome

The extraction is worthwhile, but the first implementation step is not a repo
split. The next engineering item is cross-language contract testing. Once the
contract is stable, the Rust kernel and formal proof artefacts can move to a
separately versioned package/repository without weakening the "just works"
Python path.
