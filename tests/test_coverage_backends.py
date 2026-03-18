# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Coverage tests for backends.py â€” ScorerBackend registry."""

from __future__ import annotations

import pytest

from director_ai.core.backends import (
    LiteBackend,
    ScorerBackend,
    get_backend,
    list_backends,
    register_backend,
)


class TestRegistration:
    def test_register_and_get(self):
        class Dummy(ScorerBackend):
            def score(self, p, h):
                return 0.5

            def score_batch(self, pairs):
                return [0.5] * len(pairs)

        register_backend("test_dummy", Dummy)
        assert get_backend("test_dummy") is Dummy

    def test_register_non_subclass(self):
        with pytest.raises(TypeError, match="ScorerBackend"):
            register_backend("bad", str)

    def test_get_unknown(self):
        with pytest.raises(KeyError, match="Unknown"):
            get_backend("nonexistent_backend_xyz")

    def test_list_backends(self):
        backends = list_backends()
        assert "lite" in backends
        assert "deberta" in backends


class TestLiteBackend:
    def test_score(self):
        be = LiteBackend()
        result = be.score("sky is blue", "sky is blue")
        assert 0.0 <= result <= 1.0

    def test_score_batch(self):
        be = LiteBackend()
        results = be.score_batch([("a", "b"), ("c", "d")])
        assert len(results) == 2
