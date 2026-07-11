# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Task-Scoring Rust Accelerator Binding (backfire_kernel)
"""Canonical binding of the Rust task-scoring accelerators from ``backfire_kernel``.

Every task-scoring compute path that has a Rust fast lane (task-type
classification, sentence splitting, claim-coverage reduction, integer
summation) resolves it through this module: consumers read
``_task_accel._RUST_TASK`` and call ``_task_accel.rust_*`` dynamically,
so forcing the pure-Python floor in a test means patching exactly one
flag in exactly one place. This is the task-lane sibling of
:mod:`._nli_accel` — the two shared kernel functions (sentence splitting
and coverage reduction) are bound independently per lane so each lane's
floor can be forced without touching the other. When ``backfire_kernel``
is not installed, the flag is False and the stubs below keep the names
importable for the accelerated branch without ever being called.
"""

from __future__ import annotations

__all__ = [
    "_RUST_TASK",
    "rust_coverage_from_divergences",
    "rust_detect_task_type",
    "rust_split_sentences",
    "rust_sum_i64",
]

try:
    from backfire_kernel import (
        rust_coverage_from_divergences,
        rust_detect_task_type,
        rust_split_sentences,
        rust_sum_i64,
    )

    _RUST_TASK = True
except ImportError:
    # Rust unavailable → fall through to the pure-Python floor. The stubs keep
    # the names bound for the accelerated branch but are never called when False.
    _RUST_TASK = False

    def rust_coverage_from_divergences(
        _divergences: list[float],
        _support_threshold: float,
    ) -> tuple[float, int]:
        """Raise when the Rust claim coverage reducer is unavailable."""
        raise RuntimeError(
            "backfire_kernel rust_coverage_from_divergences is unavailable"
        )

    def rust_detect_task_type(_prompt: str, _response: str) -> str:
        """Raise when the Rust task classifier is unavailable."""
        raise RuntimeError("backfire_kernel rust_detect_task_type is unavailable")

    def rust_split_sentences(_text: str) -> list[str]:
        """Raise when the Rust sentence splitter accelerator is unavailable."""
        raise RuntimeError("backfire_kernel rust_split_sentences is unavailable")

    def rust_sum_i64(_values: list[int]) -> int:
        """Raise when the Rust integer summation accelerator is unavailable."""
        raise RuntimeError("backfire_kernel rust_sum_i64 is unavailable")
