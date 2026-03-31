# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Backend Wrappers Tests (STRONG)
"""Multi-angle tests for built-in backend wrappers.

Covers: DeBERTaBackend, OnnxBackend, MiniCheckBackend instantiation
(use_model=False heuristic mode), scoring, batch scoring, score ranges,
entry point loading, parametrised backends, and pipeline performance.
"""

from __future__ import annotations

import pytest

from director_ai.core.backends import (
    DeBERTaBackend,
    MiniCheckBackend,
    OnnxBackend,
    _load_entry_points,
)

# ── Backend instantiation ────────────────────────────────────────


@pytest.fixture(
    params=[
        ("DeBERTa", DeBERTaBackend),
        ("Onnx", OnnxBackend),
        ("MiniCheck", MiniCheckBackend),
    ]
)
def backend(request):
    name, cls = request.param
    return name, cls(use_model=False)


class TestBackendInstantiation:
    """All backends must work in heuristic mode (use_model=False)."""

    def test_score_returns_numeric(self, backend):
        name, be = backend
        result = be.score("sky is blue", "sky is blue")
        assert isinstance(result, (float, int))

    def test_score_in_range(self, backend):
        name, be = backend
        result = be.score("sky is blue", "sky is blue")
        assert 0.0 <= result <= 1.0

    def test_batch_returns_list(self, backend):
        name, be = backend
        results = be.score_batch([("a", "b")])
        assert isinstance(results, list)
        assert len(results) == 1

    @pytest.mark.parametrize(
        "premise,hypothesis",
        [
            ("test", "test"),
            ("", ""),
            ("long " * 5000, "short"),
            ("🎉 emoji", "response 🎉"),
        ],
    )
    def test_various_inputs(self, backend, premise, hypothesis):
        _, be = backend
        result = be.score(premise, hypothesis)
        assert 0.0 <= result <= 1.0

    @pytest.mark.parametrize("batch_size", [0, 1, 5, 20])
    def test_batch_various_sizes(self, backend, batch_size):
        _, be = backend
        pairs = [("p", "h")] * batch_size
        results = be.score_batch(pairs)
        assert len(results) == batch_size

    def test_deterministic(self, backend):
        _, be = backend
        s1 = be.score("X", "Y")
        s2 = be.score("X", "Y")
        assert s1 == s2


# ── Entry points ──────────────────────────────────────────────────


class TestEntryPoints:
    """Entry point loading must be idempotent."""

    def test_load_entry_points_no_error(self):
        import director_ai.core.backends as bmod

        bmod._ENTRY_POINTS_LOADED = False
        _load_entry_points()
        assert bmod._ENTRY_POINTS_LOADED

    def test_load_entry_points_idempotent(self):
        import director_ai.core.backends as bmod

        _load_entry_points()
        _load_entry_points()
        assert bmod._ENTRY_POINTS_LOADED


# ── Performance documentation ────────────────────────────────────


class TestBackendWrapperPerformance:
    """Document heuristic backend latency."""

    @pytest.mark.parametrize("cls", [DeBERTaBackend, OnnxBackend, MiniCheckBackend])
    def test_heuristic_mode_fast(self, cls):
        """All backends in heuristic mode must be sub-millisecond."""
        import time

        be = cls(use_model=False)
        t0 = time.perf_counter()
        for _ in range(100):
            be.score("test", "test")
        per_call_ms = (time.perf_counter() - t0) / 100 * 1000
        assert per_call_ms < 1.0, f"{cls.__name__} took {per_call_ms:.3f}ms/call"
