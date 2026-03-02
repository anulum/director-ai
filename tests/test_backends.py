# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Backend Registry Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import pytest

from director_ai.core.backends import (
    DeBERTaBackend,
    LiteBackend,
    ScorerBackend,
    get_backend,
    list_backends,
    register_backend,
)


class TestScorerBackendABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            ScorerBackend()

    def test_subclass_must_implement_methods(self):
        class Incomplete(ScorerBackend):
            def score(self, premise, hypothesis):
                return 0.5

        with pytest.raises(TypeError):
            Incomplete()


class TestRegistry:
    def test_register_and_get(self):
        class Custom(ScorerBackend):
            def score(self, premise, hypothesis):
                return 0.42
            def score_batch(self, pairs):
                return [0.42] * len(pairs)

        register_backend("test_custom", Custom)
        assert get_backend("test_custom") is Custom

    def test_list_includes_builtins(self):
        backends = list_backends()
        assert "deberta" in backends
        assert "onnx" in backends
        assert "minicheck" in backends
        assert "lite" in backends

    def test_unknown_backend_raises(self):
        with pytest.raises(KeyError, match="nonexistent"):
            get_backend("nonexistent")

    def test_register_non_subclass_raises(self):
        with pytest.raises(TypeError):
            register_backend("bad", str)


class TestLiteBackend:
    def test_score_returns_float(self):
        b = LiteBackend()
        s = b.score("The sky is blue.", "The sky is blue.")
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0

    def test_score_batch(self):
        b = LiteBackend()
        results = b.score_batch([("a", "b"), ("c", "d")])
        assert len(results) == 2


class TestNLIScorerCustomBackend:
    def test_accepts_backend_instance(self):
        from director_ai.core.nli import NLIScorer

        class ConstBackend(ScorerBackend):
            def score(self, premise, hypothesis):
                return 0.33
            def score_batch(self, pairs):
                return [0.33] * len(pairs)

        scorer = NLIScorer(backend=ConstBackend())
        assert scorer.score("a", "b") == 0.33
        assert scorer.score_batch([("a", "b")]) == [0.33]

    def test_rejects_invalid_backend_type(self):
        from director_ai.core.nli import NLIScorer

        with pytest.raises(TypeError):
            NLIScorer(backend=42)

    def test_custom_backend_model_available(self):
        from director_ai.core.nli import NLIScorer

        class Dummy(ScorerBackend):
            def score(self, premise, hypothesis):
                return 0.5
            def score_batch(self, pairs):
                return [0.5] * len(pairs)

        scorer = NLIScorer(backend=Dummy())
        assert scorer.model_available is True


class TestEntryPointDiscovery:
    def test_entry_points_loaded_flag(self):
        import director_ai.core.backends as mod
        mod._load_entry_points()
        assert mod._ENTRY_POINTS_LOADED is True
