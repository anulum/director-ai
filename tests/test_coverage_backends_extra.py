# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Coverage tests for backends.py — built-in backend wrappers."""

from __future__ import annotations

from director_ai.core.backends import (
    DeBERTaBackend,
    MiniCheckBackend,
    OnnxBackend,
    _load_entry_points,
)


class TestDeBERTaBackend:
    def test_instantiation(self):
        be = DeBERTaBackend(use_model=False)
        result = be.score("sky is blue", "sky is blue")
        assert 0.0 <= result <= 1.0

    def test_batch(self):
        be = DeBERTaBackend(use_model=False)
        results = be.score_batch([("a", "b")])
        assert len(results) == 1


class TestOnnxBackend:
    def test_instantiation(self):
        be = OnnxBackend(use_model=False)
        result = be.score("sky is blue", "sky is blue")
        assert 0.0 <= result <= 1.0

    def test_batch(self):
        be = OnnxBackend(use_model=False)
        results = be.score_batch([("a", "b")])
        assert len(results) == 1


class TestMiniCheckBackend:
    def test_instantiation(self):
        be = MiniCheckBackend(use_model=False)
        result = be.score("sky is blue", "sky is blue")
        assert 0.0 <= result <= 1.0

    def test_batch(self):
        be = MiniCheckBackend(use_model=False)
        results = be.score_batch([("a", "b")])
        assert len(results) == 1


class TestEntryPoints:
    def test_load_entry_points_no_error(self):
        import director_ai.core.backends as bmod

        bmod._ENTRY_POINTS_LOADED = False
        _load_entry_points()
        assert bmod._ENTRY_POINTS_LOADED
