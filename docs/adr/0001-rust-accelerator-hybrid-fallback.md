<!--
SPDX-License-Identifier: Apache-2.0
Commercial license available
(c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
(c) Code 2020-2026 Miroslav Sotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# ADR-0001 — Rust accelerators: hybrid bit-exact fallback

- **Status:** Accepted (2026-07-07)
- **Deciders:** Miroslav Šotek (CEO)
- **Supersedes:** the undocumented "mandatory accelerator" behaviour in
  `core/mandatory.py` and the contradicting "additive / Python stands on its
  own" claim in `ARCHITECTURE.md`.

## Context

`backfire-kernel` (PyO3/Rust) accelerates hot-path compute across the codebase.
The intended contract, per `ARCHITECTURE.md`, was *additive*: "all language
components are additive — the Python path stands on its own without any of
them", with "a pure-Python fallback".

The code does **not** honour that contract. Across **71 modules** (~90 import
sites) the pattern is:

```python
try:
    from backfire_kernel import rust_sum_f64
    _RUST_X = True
except ImportError:
    _RUST_X = True              # <-- flag is True in BOTH branches

    def rust_sum_f64(_values):  # stub that only raises
        raise RuntimeError("backfire_kernel rust_sum_f64 is unavailable")
```

`core/mandatory.py` documents this deliberately: accelerators are "required
production capabilities … preventing silent fallback or degraded behaviour".
So the flag is always `True`, the pure-Python branch is unreachable, and a base
`pip install director-ai` **without** the `rust` extra raises `RuntimeError`
when any accelerated path runs. Two things are therefore wrong at once:

1. **Integrity defect** — `ARCHITECTURE.md` claims a Python fallback that does
   not exist.
2. **Functional defect** — a base install is partially non-functional.

This also conflicts with the fleet-wide **"Python floor"** rule (every compute
function keeps a working pure-Python baseline). A single project cannot exempt
itself from that rule without CEO sign-off.

## Decision

**Hybrid.** Provide a reachable pure-Python fallback **only where it is
bit-exact** with the Rust kernel; keep the accelerator **mandatory** (honest
`RuntimeError`, `rust` extra required) where a pure-Python path cannot reproduce
the Rust result bit-for-bit. This gives a working Python floor for the exactly
reproducible kernels **without** reintroducing the silent numerical degradation
`mandatory.py` was built to prevent.

Bit-exactness is **not assumed — it is proven empirically** against the actual
`backfire-kernel` binary (a parity test that runs both the Rust kernel and the
candidate Python fallback over many inputs and asserts identical IEEE-754 bits).

### Critical finding — bit-exactness is subtle even for "sum"

Measured against the installed `backfire-kernel` (2026-07-07, Python 3.12):

| Kernel | Candidate Python | Bit-exact? | Evidence |
|---|---|---|---|
| `rust_sum_i64` | `sum(values)` | **Yes** | 0 mismatches / 20 000 (integer addition is exact) |
| `rust_sum_f64` | `sum(values, 0.0)` | **No** | 11 737 / 20 000 — CPython 3.12 `sum()` uses Neumaier compensated summation, not naive folding |
| `rust_sum_f64` | naive left-to-right fold | **No** | 480 / 30 000 — Rust `iter().sum()` result depends on SIMD codegen, not portably reproducible |
| `rust_sum_f64` | `math.fsum` / `numpy.sum` | **No** | 16 007 / 12 278 mismatches — different algorithms entirely |

**Implication:** `rust_sum_f64` (used by ~40 modules) is **not** bit-exactly
reproducible in portable Python, so those modules stay mandatory. Integer and
other deterministic kernels are fallback-eligible. Never ship a float-sum
"fallback" on the assumption that "a sum is a sum" — it would be exactly the
silent degradation this ADR forbids.

## Classification (initial; refined per-kernel as parity is measured)

- **Fallback-eligible (bit-exact, integer/deterministic):** `rust_sum_i64` and
  the modules that use only it. Candidates pending parity proof:
  `rust_word_overlap`, `rust_has_suspicious_unicode`, `rust_split_sentences`
  (deterministic string/set ops — verify each).
- **Mandatory (not bit-exact reproducible):** all `rust_sum_f64` /
  `rust_mean` / `rust_softmax` / `rust_conformal_quantile` float kernels;
  transcendental/geometry (`rust_two_link_ik`, `cyber_physical` geometry);
  cryptographic (`rust_verify_reality_anchor_mac`, `zk_attestation`,
  `provenance` commitments); model backends (`RustCoherenceScorer`,
  `rust_lite_score`); and NLP surfaces whose Rust and Python paths differ
  (`PiiScanner`, `rust_detect_fallacies`, `rust_eval_arithmetic`).

## Consequences

- Base `pip install director-ai` gains a working Python floor for the
  bit-exact (integer) kernels; float/crypto/model features continue to require
  the `rust` extra and fail with a clear, honest error when it is absent.
- `ARCHITECTURE.md` and backend-tier docs are corrected to describe this hybrid
  reality instead of a universal fallback that never existed.
- No silent numerical degradation: a fallback ships only after a committed
  parity test proves bit-exactness against the Rust binary.

## Rollout (phased — tracked in the internal BACKLOG, WS-B/ADR-1 lane)

1. This ADR + the `ARCHITECTURE.md` correction (records the decision, fixes the
   integrity defect). — **done 2026-07-07**
2. A shared `core` fallback helper for the proven bit-exact kernels
   (`sum_i64` first), each covered by a Rust-parity test.
3. Per-module conversion of the fallback-eligible modules (except-branch
   delegates to the helper and sets `_RUST_* = False`), Rust-absent tested.
4. For mandatory modules: keep the honest raise, add a one-line "rust extra
   required" note, and mark the `rust` extra required for those features in the
   backend-tier docs.

Until a module reaches step 3 it remains mandatory; the honest `RuntimeError`
is the correct interim behaviour (no silent fallback).
