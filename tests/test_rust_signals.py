# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Rust-Accelerated Signals and BM25 Tests (STRONG)
"""Multi-angle tests for Rust signal functions and BM25 pipeline (STRONG).

Runs against the Python fallback when backfire_kernel is not installed,
and against the Rust implementation when it is.
"""

from __future__ import annotations

from director_ai.core.runtime.streaming import _trend_drop
from director_ai.core.scoring.verified_scorer import (
    _entity_overlap,
    _negation_flip,
    _numerical_consistency,
    _traceability,
)


class TestEntityOverlap:
    def test_identical_entities(self):
        score = _entity_overlap(
            "Paris is the capital of France",
            "France has its capital in Paris",
        )
        assert score > 0.5

    def test_no_entities(self):
        assert _entity_overlap("hello world", "goodbye world") == 1.0

    def test_disjoint_entities(self):
        score = _entity_overlap(
            "Alice works at Google",
            "Bob works at Microsoft",
        )
        assert score < 0.5

    def test_empty_strings(self):
        assert _entity_overlap("", "") == 1.0

    def test_one_sided_entities(self):
        score = _entity_overlap("Paris is nice", "the city is nice")
        assert score < 1.0


class TestNumericalConsistency:
    def test_matching_numbers(self):
        assert (
            _numerical_consistency("There are 46 chromosomes", "Humans have 46 total")
            is True
        )

    def test_mismatched_numbers(self):
        assert (
            _numerical_consistency(
                "Data retained for 90 days", "Data retained for 30 days"
            )
            is False
        )

    def test_no_numbers(self):
        assert _numerical_consistency("The sky is blue", "Blue sky") is None

    def test_one_sided_numbers(self):
        assert _numerical_consistency("There are 23 pairs", "Many pairs exist") is None

    def test_comma_numbers(self):
        result = _numerical_consistency(
            "Population is 1,000,000", "City has 1,000,000 people"
        )
        assert result is True


class TestNegationFlip:
    def test_flip_detected(self):
        assert _negation_flip(
            "Phone support is not available for Team plan",
            "Phone support is available for all paid plans",
        )

    def test_same_polarity(self):
        assert not _negation_flip(
            "Water boils at 100 degrees",
            "Water boils at 100 degrees Celsius",
        )

    def test_both_negated(self):
        assert not _negation_flip(
            "Support is not available on weekends",
            "Support is not provided on holidays",
        )

    def test_short_texts_no_flip(self):
        assert not _negation_flip("not here", "is here")


class TestTraceability:
    def test_high_traceability(self):
        score = _traceability(
            "Water boils at 100 degrees Celsius",
            "Water boils at 100 degrees Celsius at standard pressure",
        )
        assert score > 0.8

    def test_low_traceability(self):
        score = _traceability(
            "HIPAA and FedRAMP certified",
            "SOC 2 Type II and ISO 27001 certified",
        )
        assert score < 0.5

    def test_empty_claim(self):
        assert _traceability("the a an", "anything here") == 1.0

    def test_zero_traceability(self):
        score = _traceability(
            "quantum teleportation entanglement",
            "water boils temperature pressure",
        )
        assert score == 0.0


class TestTrendDrop:
    def test_flat_trend(self):
        assert abs(_trend_drop([0.5, 0.5, 0.5, 0.5, 0.5])) < 1e-10

    def test_declining_trend(self):
        assert _trend_drop([0.9, 0.7, 0.5, 0.3, 0.1]) > 0.5

    def test_rising_trend(self):
        assert _trend_drop([0.1, 0.3, 0.5, 0.7, 0.9]) < -0.5

    def test_single_value(self):
        assert _trend_drop([0.5]) == 0.0

    def test_two_values(self):
        drop = _trend_drop([0.8, 0.2])
        assert drop > 0.0

    def test_empty(self):
        assert _trend_drop([]) == 0.0


class TestRustDispatch:
    """Verify that the Rust dispatch flag exists and is consistent."""

    def test_verified_scorer_has_flag(self):
        from director_ai.core.scoring import verified_scorer

        assert hasattr(verified_scorer, "_RUST_SIGNALS")
        assert isinstance(verified_scorer._RUST_SIGNALS, bool)

    def test_streaming_has_flag(self):
        from director_ai.core.runtime import streaming

        assert hasattr(streaming, "_RUST_TREND")
        assert isinstance(streaming._RUST_TREND, bool)


class TestBM25RustFallback:
    """Test BM25 via the Python HybridBackend (Rust BM25 tested in cargo)."""

    def test_hybrid_backend_bm25(self):
        from director_ai.core.retrieval.vector_store import (
            HybridBackend,
            InMemoryBackend,
        )

        base = InMemoryBackend()
        hybrid = HybridBackend(base=base)
        hybrid.add("d1", "Water boils at 100 degrees Celsius")
        hybrid.add("d2", "The speed of light is 299792 km per second")
        hybrid.add("d3", "DNA has four nucleotide bases")

        results = hybrid.query("water boiling temperature", n_results=2)
        assert len(results) > 0
        assert (
            "water" in results[0]["text"].lower()
            or "boil" in results[0]["text"].lower()
        )

    def test_hybrid_empty_query(self):
        from director_ai.core.retrieval.vector_store import (
            HybridBackend,
            InMemoryBackend,
        )

        hybrid = HybridBackend(base=InMemoryBackend())
        hybrid.add("d1", "Some content here")
        results = hybrid.query("", n_results=3)
        assert isinstance(results, list)
