# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Backend Registry Tests (STRONG)
"""Multi-angle tests for ScorerBackend registry and LiteBackend.

Covers: registration, lookup, error handling, list enumeration,
LiteBackend scoring, batch scoring, score range invariants,
backend protocol compliance, entry point loading, parametrised
backend instantiation, and pipeline performance documentation.
"""

from __future__ import annotations

import pytest

from director_ai.core.backends import (
    LiteBackend,
    ScorerBackend,
    get_backend,
    list_backends,
    register_backend,
)

# ── Registration ──────────────────────────────────────────────────


class TestRegistration:
    """Test backend registration and lookup."""

    def _make_dummy(self):
        class Dummy(ScorerBackend):
            def score(self, p, h):
                return 0.5

            def score_batch(self, pairs):
                return [0.5] * len(pairs)

        return Dummy

    def test_register_and_get(self):
        Dummy = self._make_dummy()
        register_backend("test_dummy_reg", Dummy)
        assert get_backend("test_dummy_reg") is Dummy

    def test_register_non_subclass_raises(self):
        with pytest.raises(TypeError, match="ScorerBackend"):
            register_backend("bad_reg", str)

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown"):
            get_backend("nonexistent_backend_xyz_42")

    def test_list_backends_has_builtins(self):
        backends = list_backends()
        assert "lite" in backends
        assert "deberta" in backends

    def test_register_overwrites(self):
        D1 = self._make_dummy()
        D2 = self._make_dummy()
        register_backend("test_overwrite", D1)
        register_backend("test_overwrite", D2)
        assert get_backend("test_overwrite") is D2

    @pytest.mark.parametrize("builtin", ["lite", "deberta"])
    def test_builtin_backends_retrievable(self, builtin):
        cls = get_backend(builtin)
        assert issubclass(cls, ScorerBackend)


# ── LiteBackend ───────────────────────────────────────────────────


class TestLiteBackend:
    """Multi-angle tests for the lightweight heuristic backend."""

    @pytest.fixture
    def backend(self):
        return LiteBackend()

    def test_score_returns_float(self, backend):
        result = backend.score("sky is blue", "sky is blue")
        assert isinstance(result, float)

    def test_score_in_range(self, backend):
        result = backend.score("sky is blue", "sky is blue")
        assert 0.0 <= result <= 1.0

    @pytest.mark.parametrize(
        "premise,hypothesis",
        [
            ("sky is blue", "sky is blue"),
            ("", "response"),
            ("prompt", ""),
            ("", ""),
            ("🎉", "🎉"),
            ("a" * 10000, "b" * 10000),
        ],
    )
    def test_various_inputs(self, backend, premise, hypothesis):
        result = backend.score(premise, hypothesis)
        assert 0.0 <= result <= 1.0

    def test_score_batch(self, backend):
        results = backend.score_batch([("a", "b"), ("c", "d"), ("e", "f")])
        assert len(results) == 3
        assert all(0.0 <= r <= 1.0 for r in results)

    def test_score_batch_empty(self, backend):
        results = backend.score_batch([])
        assert results == []

    @pytest.mark.parametrize("batch_size", [1, 5, 10, 50])
    def test_score_batch_various_sizes(self, backend, batch_size):
        pairs = [("p", "h")] * batch_size
        results = backend.score_batch(pairs)
        assert len(results) == batch_size

    def test_deterministic(self, backend):
        s1 = backend.score("X", "Y")
        s2 = backend.score("X", "Y")
        assert s1 == s2


# ── Pipeline integration ─────────────────────────────────────────


class TestBackendPipelineIntegration:
    """Verify backends wire into CoherenceScorer."""

    @pytest.mark.parametrize("backend_name", ["lite", "deberta"])
    def test_scorer_accepts_backend(self, backend_name):
        from director_ai.core import CoherenceScorer

        scorer = CoherenceScorer(use_nli=False, scorer_backend=backend_name)
        approved, score = scorer.review("test", "test")
        assert isinstance(approved, bool)
        assert 0.0 <= score.score <= 1.0


# ── Performance documentation ────────────────────────────────────


class TestBackendPerformance:
    """Document backend performance characteristics."""

    def test_lite_backend_fast(self):
        """LiteBackend must be sub-millisecond per call."""
        import time

        be = LiteBackend()
        # Warmup
        for _ in range(10):
            be.score("warmup", "warmup")

        t0 = time.perf_counter()
        for _ in range(1000):
            be.score("What is AI?", "AI is intelligence.")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        per_call_ms = elapsed_ms / 1000
        assert per_call_ms < 1.0, (
            f"LiteBackend {per_call_ms:.3f}ms/call (expected <1ms)"
        )

    def test_list_backends_returns_dict(self):
        backends = list_backends()
        assert isinstance(backends, (list, dict, set))
