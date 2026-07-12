# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for retrieval run fusion strategies.

Covers: method validation, weighted RRF arithmetic, convex min-max
CombSUM, CombMNZ agreement multiplier, z-score fusion, normalisation
edge cases (empty, constant, zero-variance runs), weight and rrf_k
validation, row merge semantics, and ordering determinism.
"""

from __future__ import annotations

import pytest

from director_ai.core.retrieval.vector_store.fusion import (
    FUSION_METHODS,
    _minmax,
    _zscores,
    fuse_results,
    validate_fusion_method,
)


def _run(*pairs: tuple[str, float]) -> list[tuple[dict, float]]:
    """Build a scored run from (doc_id, score) pairs."""
    return [({"id": doc_id, "text": doc_id}, score) for doc_id, score in pairs]


class TestValidateFusionMethod:
    @pytest.mark.parametrize("method", FUSION_METHODS)
    def test_accepts_canonical_names(self, method):
        assert validate_fusion_method(method) == method

    def test_canonicalises_case_and_whitespace(self):
        assert validate_fusion_method("  RRF  ") == "rrf"
        assert validate_fusion_method("Convex") == "convex"

    @pytest.mark.parametrize("method", [None, 42, ["rrf"]])
    def test_rejects_non_string(self, method):
        with pytest.raises(ValueError, match="must be a string"):
            validate_fusion_method(method)

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="fusion_method must be one of"):
            validate_fusion_method("borda")


class TestFuseResultsValidation:
    def test_rejects_bool_rrf_k(self):
        with pytest.raises(ValueError, match="rrf_k must be an integer"):
            fuse_results("rrf", [], [], rrf_k=True)

    def test_rejects_small_rrf_k(self):
        with pytest.raises(ValueError, match="rrf_k must be at least 1"):
            fuse_results("rrf", [], [], rrf_k=0)

    @pytest.mark.parametrize("field", ["sparse_weight", "dense_weight"])
    def test_rejects_negative_weight(self, field):
        with pytest.raises(ValueError, match=f"{field} must be non-negative"):
            fuse_results("rrf", [], [], **{field: -0.5})

    @pytest.mark.parametrize("field", ["sparse_weight", "dense_weight"])
    def test_rejects_non_numeric_weight(self, field):
        with pytest.raises(ValueError, match=f"{field} must be numeric"):
            fuse_results("rrf", [], [], **{field: "1.0"})

    def test_rejects_bool_weight(self):
        with pytest.raises(ValueError, match="sparse_weight must be numeric"):
            fuse_results("rrf", [], [], sparse_weight=True)

    def test_rejects_all_zero_weights(self):
        with pytest.raises(ValueError, match="at least one fusion weight"):
            fuse_results("rrf", [], [], sparse_weight=0.0, dense_weight=0.0)


class TestRrfFusion:
    def test_matches_published_formula(self):
        """score(d) = Σ w/(k + rank) with 1-based ranks (SIGIR 2009)."""
        sparse = _run(("a", 9.0), ("b", 4.0))
        dense = _run(("b", 0.9), ("c", 0.8))

        fused = fuse_results("rrf", sparse, dense, rrf_k=60)

        # a: 1/61; b: 1/62 + 1/61; c: 1/62 → b > a > c
        assert [row["id"] for row in fused] == ["b", "a", "c"]

    def test_weights_scale_run_contributions(self):
        sparse = _run(("a", 9.0))
        dense = _run(("b", 0.9))

        fused = fuse_results("rrf", sparse, dense, sparse_weight=5.0)

        assert [row["id"] for row in fused] == ["a", "b"]

    def test_ignores_native_scores(self):
        """Identical rank patterns fuse identically whatever the scores."""
        low = fuse_results("rrf", _run(("a", 0.001)), _run(("b", 0.001)))
        high = fuse_results("rrf", _run(("a", 999.0)), _run(("b", 999.0)))

        assert [r["id"] for r in low] == [r["id"] for r in high]

    def test_empty_runs_fuse_to_empty(self):
        assert fuse_results("rrf", [], []) == []


class TestConvexFusion:
    def test_convex_combination_of_minmax_scores(self):
        """Weights normalise to a convex combination over min-max scores."""
        sparse = _run(("a", 10.0), ("b", 5.0), ("c", 0.0))
        dense = _run(("c", 0.9), ("b", 0.5), ("a", 0.1))

        fused = fuse_results(
            "convex",
            sparse,
            dense,
            sparse_weight=1.0,
            dense_weight=1.0,
        )

        # a: 0.5·1.0 + 0.5·0.0 = 0.5; b: 0.5·0.5 + 0.5·0.5 = 0.5 (tie,
        # insertion order keeps a first); c: 0.5·0.0 + 0.5·1.0 = 0.5.
        # All tie at 0.5 → insertion order a, b, c.
        assert [row["id"] for row in fused] == ["a", "b", "c"]

    def test_dense_weight_dominates(self):
        sparse = _run(("a", 10.0), ("b", 1.0))
        dense = _run(("b", 0.9), ("a", 0.1))

        fused = fuse_results(
            "convex",
            sparse,
            dense,
            sparse_weight=0.1,
            dense_weight=0.9,
        )

        assert fused[0]["id"] == "b"

    def test_missing_doc_contributes_zero(self):
        sparse = _run(("a", 10.0), ("b", 8.0))
        dense = _run(("a", 0.9), ("c", 0.8))

        fused = fuse_results("convex", sparse, dense)

        assert fused[0]["id"] == "a"

    def test_constant_run_counts_presence(self):
        """A constant run min-maxes to 1.0 — presence still matters."""
        sparse = _run(("a", 3.0), ("b", 3.0))
        dense = _run(("b", 0.5))

        fused = fuse_results("convex", sparse, dense)

        assert fused[0]["id"] == "b"


class TestCombmnzFusion:
    def test_agreement_multiplier_rewards_overlap(self):
        """A doc in both runs beats a single-run doc with equal sum."""
        sparse = _run(("both", 5.0), ("solo", 10.0))
        dense = _run(("both", 0.5))

        fused = fuse_results("combmnz", sparse, dense)

        # solo: minmax 1.0 (weight 0.5) × 1 hit = 0.5
        # both: (0.5·0.0 + 0.5·1.0) × 2 hits = 1.0
        assert fused[0]["id"] == "both"

    def test_single_run_reduces_to_convex_order(self):
        sparse = _run(("a", 9.0), ("b", 4.0), ("c", 1.0))

        combmnz = fuse_results("combmnz", sparse, [])
        convex = fuse_results("convex", sparse, [])

        assert [r["id"] for r in combmnz] == [r["id"] for r in convex]


class TestZscoreFusion:
    def test_standardised_sum_orders_by_joint_evidence(self):
        sparse = _run(("a", 10.0), ("b", 6.0), ("c", 2.0))
        dense = _run(("b", 0.9), ("a", 0.6), ("c", 0.3))

        fused = fuse_results("zscore", sparse, dense)

        # a: z=+1.22, z=0 → 0.61; b: z=0, z=+1.22 → 0.61; c: −1.22 twice.
        assert {fused[0]["id"], fused[1]["id"]} == {"a", "b"}
        assert fused[2]["id"] == "c"

    def test_zero_variance_run_contributes_nothing(self):
        """Documented: constant runs carry no signal under z-scoring."""
        sparse = _run(("a", 5.0), ("b", 5.0))
        dense = _run(("b", 0.9), ("a", 0.1))

        fused = fuse_results("zscore", sparse, dense)

        assert fused[0]["id"] == "b"


class TestRowMergeSemantics:
    def test_dense_row_wins_for_shared_doc(self):
        """Last-write-wins mirrors the historical HybridBackend merge."""
        sparse = [({"id": "a", "text": "sparse row", "distance": 0.4}, 1.5)]
        dense = [({"id": "a", "text": "dense row", "distance": 0.2}, 0.8)]

        fused = fuse_results("rrf", sparse, dense)

        assert len(fused) == 1
        assert fused[0]["text"] == "dense row"

    def test_returns_full_ranking_without_truncation(self):
        sparse = _run(*[(f"s{i}", 10.0 - i) for i in range(5)])
        dense = _run(*[(f"d{i}", 1.0 - i / 10) for i in range(5)])

        fused = fuse_results("rrf", sparse, dense)

        assert len(fused) == 10


class TestNormalisationHelpers:
    def test_minmax_empty(self):
        assert _minmax([]) == []

    def test_minmax_constant(self):
        assert _minmax([2.0, 2.0, 2.0]) == [1.0, 1.0, 1.0]

    def test_minmax_spans_unit_interval(self):
        assert _minmax([1.0, 3.0, 5.0]) == [0.0, 0.5, 1.0]

    def test_zscores_empty(self):
        assert _zscores([]) == []

    def test_zscores_constant(self):
        assert _zscores([4.0, 4.0]) == [0.0, 0.0]

    def test_zscores_standardises(self):
        values = _zscores([2.0, 4.0, 6.0])

        assert values[1] == pytest.approx(0.0)
        assert values[0] == pytest.approx(-values[2])
        assert values[2] == pytest.approx(1.2247, abs=1e-4)
