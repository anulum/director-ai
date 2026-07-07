# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Pure-Python fallbacks proven bit-exact with backfire-kernel accelerators.

Only kernels whose pure-Python implementation reproduces the Rust result
**bit-for-bit** live here (ADR-0001). A committed parity test
(``tests/test_accelerator_fallback_parity.py``) proves each against the actual
installed ``backfire-kernel`` binary. This gives a base ``pip install`` a working
Python floor for the exactly reproducible kernels without reintroducing the
silent numerical degradation the mandatory-accelerator pattern was built to
prevent.

Do **not** add a float-sum fallback here: ``rust_sum_f64`` is not portably
bit-exact across builds (its result depends on SIMD codegen; CPython 3.12
``sum`` uses compensated summation) — see ADR-0001. Integer and other provably
deterministic kernels are the only fallback-eligible ones.
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = ["sum_i64"]

# Signed 64-bit domain. ``rust_sum_i64`` takes a ``Vec<i64>`` (PyO3 rejects any
# input outside this range with OverflowError) and returns an ``i64`` whose
# ``iter().sum()`` wraps two's-complement on overflow in a release build.
_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1
_UINT64_MOD = 2**64
_UINT64_MASK = _UINT64_MOD - 1
_I64_SIGN_BIT = 2**63


def sum_i64(values: Iterable[int]) -> int:
    """Sum int64 values, bit-exact with ``backfire_kernel.rust_sum_i64``.

    Reproduces the Rust kernel exactly across the whole i64 domain, not merely
    for small inputs:

    * any value outside the signed 64-bit range raises ``OverflowError``, as
      PyO3 does when extracting the ``Vec<i64>`` (e.g. ``rust_sum_i64([2**63])``);
    * the accumulated total wraps two's-complement into the signed 64-bit range,
      matching ``values.iter().sum()`` on overflow (e.g. ``[i64::MAX, 1] ->
      i64::MIN``). Modular addition is associative, so wrapping once at the end
      yields the same i64 the Rust per-step wrap does.

    Within the non-overflow range this is ordinary exact integer addition, so it
    is identical to the accelerator (proven 0 mismatches, ADR-0001).
    """
    total = 0
    for value in values:
        if value < _I64_MIN or value > _I64_MAX:
            raise OverflowError(f"int64 value out of range: {value}")
        total += value
    total &= _UINT64_MASK
    if total >= _I64_SIGN_BIT:
        total -= _UINT64_MOD
    return total
