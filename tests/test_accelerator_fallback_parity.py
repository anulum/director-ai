# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ADR-0001 accelerator-fallback bit-exactness parity
"""Bit-exactness parity for the pure-Python accelerator fallbacks (ADR-0001).

Two layers, no mocks:

* **Contract unit tests** run everywhere (including a base install without the
  compiled ``backfire_kernel``) and lock ``sum_i64``'s exact behaviour — empty
  sum, ordinary sums, the signed-64-bit boundaries, two's-complement wrap on
  overflow, and ``OverflowError`` on out-of-range inputs.
* **Parity tests** run the fallback and the real ``backfire_kernel.rust_sum_i64``
  over the same inputs and assert identical results (and identical rejection of
  out-of-range inputs). This is the empirical proof the ADR requires; skipped
  only if the compiled kernel is unavailable.
"""

from __future__ import annotations

import random

import pytest

from director_ai.core.accelerator_fallback import sum_i64

_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1

try:
    from backfire_kernel import rust_sum_i64 as _rust_sum_i64

    _HAS_RUST = True
except ImportError:  # pragma: no cover - exercised only in a no-kernel install
    _HAS_RUST = False

_needs_rust = pytest.mark.skipif(
    not _HAS_RUST, reason="backfire_kernel (compiled) not installed"
)


# ── Contract unit tests (always run) ──────────────────────────────────────────


def test_sum_i64_empty_is_zero() -> None:
    assert sum_i64([]) == 0


def test_sum_i64_ordinary_sum() -> None:
    assert sum_i64([1, 2, 3, -4]) == 2


def test_sum_i64_signed_64bit_boundaries() -> None:
    assert sum_i64([_I64_MAX]) == _I64_MAX
    assert sum_i64([_I64_MIN]) == _I64_MIN


def test_sum_i64_wraps_two_complement_on_overflow() -> None:
    # [i64::MAX, 1] -> i64::MIN; [i64::MAX, i64::MAX] -> -2; [i64::MIN, -1] -> i64::MAX
    assert sum_i64([_I64_MAX, 1]) == _I64_MIN
    assert sum_i64([_I64_MAX, _I64_MAX]) == -2
    assert sum_i64([_I64_MIN, -1]) == _I64_MAX


@pytest.mark.parametrize("bad", [2**63, -(2**63) - 1, 2**70, -(2**70)])
def test_sum_i64_rejects_out_of_range_inputs(bad: int) -> None:
    with pytest.raises(OverflowError):
        sum_i64([bad])


# ── Parity against the real compiled kernel ───────────────────────────────────


@_needs_rust
def test_parity_fixed_edge_cases() -> None:
    cases = [
        [],
        [0],
        [1, 2, 3],
        [-1, -2, -3],
        [_I64_MAX],
        [_I64_MIN],
        [_I64_MAX, 1],
        [_I64_MAX, _I64_MAX],
        [_I64_MIN, -1],
        [_I64_MIN, _I64_MIN],
        [_I64_MAX, _I64_MIN],
    ]
    for values in cases:
        assert sum_i64(values) == _rust_sum_i64(values), values


@_needs_rust
def test_parity_randomised_in_and_over_range() -> None:
    rng = random.Random(20260707)
    for _ in range(20_000):
        n = rng.randint(0, 32)
        # Mix small values and near-boundary values so some sums overflow and
        # exercise the wrap path in both the fallback and the kernel.
        values = [
            rng.choice(
                [
                    rng.randint(-1000, 1000),
                    rng.randint(_I64_MIN, _I64_MAX),
                ]
            )
            for _ in range(n)
        ]
        assert sum_i64(values) == _rust_sum_i64(values), values


@_needs_rust
@pytest.mark.parametrize("bad", [2**63, -(2**63) - 1, 2**70])
def test_parity_out_of_range_both_raise(bad: int) -> None:
    with pytest.raises(OverflowError):
        _rust_sum_i64([bad])
    with pytest.raises(OverflowError):
        sum_i64([bad])
