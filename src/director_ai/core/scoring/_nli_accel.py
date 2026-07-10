# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — NLI Rust Accelerator Binding (backfire_kernel)
"""Canonical binding of the Rust NLI accelerators from ``backfire_kernel``.

Every NLI compute path that has a Rust fast lane (softmax, divergence and
confidence mapping, chunk aggregation, sentence splitting, chunk building,
claim-coverage reduction) resolves it through this module: consumers read
``_nli_accel._RUST_NLI`` and call ``_nli_accel.rust_*`` dynamically, so
forcing the pure-Python floor in a test means patching exactly one flag in
exactly one place. When ``backfire_kernel`` is not installed, the flag is
False and the stubs below keep the names importable for the accelerated
branch without ever being called.
"""

from __future__ import annotations

__all__ = [
    "_RUST_NLI",
    "rust_aggregate_chunk_scores",
    "rust_aggregate_chunk_scores_confidence_weighted",
    "rust_build_chunks",
    "rust_coverage_from_divergences",
    "rust_probs_to_confidence",
    "rust_probs_to_divergence",
    "rust_reduce_claim_attribution",
    "rust_softmax",
    "rust_split_sentences",
]

try:
    from backfire_kernel import (
        rust_aggregate_chunk_scores,
        rust_aggregate_chunk_scores_confidence_weighted,
        rust_build_chunks,
        rust_coverage_from_divergences,
        rust_probs_to_confidence,
        rust_probs_to_divergence,
        rust_reduce_claim_attribution,
        rust_softmax,
        rust_split_sentences,
    )

    _RUST_NLI = True
except ImportError:
    # Rust unavailable → fall through to the pure-Python floor. The stubs keep
    # the names bound for the accelerated branch but are never called when False.
    _RUST_NLI = False

    def rust_softmax(_flat: list[float], _cols: int) -> list[float]:
        """Raise when the Rust NLI softmax accelerator is unavailable."""
        raise RuntimeError("backfire_kernel rust_softmax is unavailable")

    def rust_probs_to_divergence(
        _flat: list[float],
        _ncols: int,
        _contradiction_idx: int,
        _neutral_idx: int,
    ) -> list[float]:
        """Raise when the Rust divergence accelerator is unavailable."""
        raise RuntimeError("backfire_kernel rust_probs_to_divergence is unavailable")

    def rust_probs_to_confidence(_flat: list[float], _ncols: int) -> list[float]:
        """Raise when the Rust confidence accelerator is unavailable."""
        raise RuntimeError("backfire_kernel rust_probs_to_confidence is unavailable")

    def rust_aggregate_chunk_scores(
        _flat_scores: list[float],
        _n_prem: int,
        _n_hyp: int,
        _inner_agg: str,
        _outer_agg: str,
    ) -> tuple[float, list[float]]:
        """Raise when Rust chunk aggregator accelerator is unavailable."""
        raise RuntimeError("backfire_kernel rust_aggregate_chunk_scores is unavailable")

    def rust_aggregate_chunk_scores_confidence_weighted(
        _flat_scores: list[float],
        _flat_confidences: list[float],
        _n_prem: int,
        _n_hyp: int,
        _inner_agg: str,
    ) -> tuple[float, list[float]]:
        """Raise when Rust weighted chunk aggregator is unavailable."""
        raise RuntimeError(
            "backfire_kernel rust_aggregate_chunk_scores_confidence_weighted is unavailable"
        )

    def rust_coverage_from_divergences(
        _divergences: list[float],
        _support_threshold: float,
    ) -> tuple[float, int]:
        """Raise when Rust claim coverage reducer is unavailable."""
        raise RuntimeError(
            "backfire_kernel rust_coverage_from_divergences is unavailable"
        )

    def rust_reduce_claim_attribution(
        _flat_divergences: list[float],
        _n_claims: int,
        _n_src: int,
    ) -> tuple[list[float], list[int]]:
        """Raise when Rust claim attribution reducer is unavailable."""
        raise RuntimeError(
            "backfire_kernel rust_reduce_claim_attribution is unavailable"
        )

    def rust_split_sentences(_text: str) -> list[str]:
        """Raise when the Rust sentence splitter accelerator is unavailable."""
        raise RuntimeError("backfire_kernel rust_split_sentences is unavailable")

    def rust_build_chunks(
        _sentences: list[str],
        _budget: int,
        _overlap_ratio: float,
    ) -> list[str]:
        """Raise when the Rust chunk builder accelerator is unavailable."""
        raise RuntimeError("backfire_kernel rust_build_chunks is unavailable")
