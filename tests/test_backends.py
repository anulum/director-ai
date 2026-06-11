# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Backend Registry Tests
"""Multi-angle tests for scorer backend registry and dispatch.

Covers: backend registration, lookup, listing, DeBERTa/Lite/ONNX/Rust
backends, batch scoring, score range invariants, pipeline integration
with CoherenceScorer, and performance documentation.
"""

import importlib
import sys
import types
from types import SimpleNamespace

import pytest

from director_ai.core.backends import (
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


class TestRustBackend:
    def test_rust_registration_conditional(self):
        backends = list_backends()
        # Entry-point always registers RustBackend; actual FFI availability
        # is checked at instantiation time, not registration.
        assert "rust" in backends
        assert "backfire" in backends

    def test_rust_backend_class_exists(self):
        from director_ai.core.backends import RustBackend

        assert issubclass(RustBackend, ScorerBackend)

    def test_rust_instantiation_without_backfire_raises(self):
        from unittest.mock import patch

        from director_ai.core.backends import RustBackend

        with (
            patch.dict("sys.modules", {"backfire_kernel": None}),
            pytest.raises((ImportError, ModuleNotFoundError)),
        ):
            RustBackend()

    def test_agent_falls_back_without_rust(self):
        from director_ai.core.agent import CoherenceAgent

        agent = CoherenceAgent()
        # Should not raise — falls back to Python scorer
        assert agent.scorer is not None

    def test_rust_backend_forwards_config_and_score_object(self, monkeypatch):
        from director_ai.core.backends import RustBackend

        created = {}

        class FakeBackfireConfig:
            def __init__(self, coherence_threshold):
                self.coherence_threshold = coherence_threshold

        class FakeRustCoherenceScorer:
            def __init__(self, config, knowledge_callback):
                created["threshold"] = config.coherence_threshold
                created["knowledge_callback"] = knowledge_callback

            def review(self, premise, hypothesis):
                created["review"] = (premise, hypothesis)
                return True, SimpleNamespace(score=0.73)

        fake_backfire = types.ModuleType("backfire_kernel")
        fake_backfire.BackfireConfig = FakeBackfireConfig
        fake_backfire.RustCoherenceScorer = FakeRustCoherenceScorer
        monkeypatch.setitem(sys.modules, "backfire_kernel", fake_backfire)

        callback = object()
        backend = RustBackend(threshold=0.82, knowledge_callback=callback)

        assert backend.score("claim", "evidence") == 0.73
        assert created == {
            "threshold": 0.82,
            "knowledge_callback": callback,
            "review": ("claim", "evidence"),
        }

    def test_rust_backend_batch_accepts_numeric_review_result(self, monkeypatch):
        from director_ai.core.backends import RustBackend

        class FakeBackfireConfig:
            def __init__(self, coherence_threshold):
                self.coherence_threshold = coherence_threshold

        class FakeRustCoherenceScorer:
            def __init__(self, config, knowledge_callback):
                self.calls = []

            def review(self, premise, hypothesis):
                self.calls.append((premise, hypothesis))
                return True, 0.61 + (0.1 * len(self.calls))

        fake_backfire = types.ModuleType("backfire_kernel")
        fake_backfire.BackfireConfig = FakeBackfireConfig
        fake_backfire.RustCoherenceScorer = FakeRustCoherenceScorer
        monkeypatch.setitem(sys.modules, "backfire_kernel", fake_backfire)

        backend = RustBackend()

        assert backend.score_batch([("p1", "h1"), ("p2", "h2")]) == [
            pytest.approx(0.71),
            pytest.approx(0.81),
        ]


class TestBackendWrappers:
    @pytest.mark.parametrize(
        ("wrapper_name", "expected_backend"),
        [
            ("DeBERTaBackend", "deberta"),
            ("OnnxBackend", "onnx"),
            ("MiniCheckBackend", "minicheck"),
        ],
    )
    def test_nli_wrappers_select_backend_and_forward_scores(
        self, monkeypatch, wrapper_name, expected_backend
    ):
        import director_ai.core.scoring.backends as backends_mod

        created = {}

        class FakeNLIScorer:
            def __init__(self, *, backend, **kwargs):
                created["backend"] = backend
                created["kwargs"] = kwargs

            def score(self, premise, hypothesis):
                created["score"] = (premise, hypothesis)
                return 0.44

            def score_batch(self, pairs):
                created["batch"] = pairs
                return [0.45 for _ in pairs]

        fake_nli = types.ModuleType("director_ai.core.scoring.nli")
        fake_nli.NLIScorer = FakeNLIScorer
        monkeypatch.setitem(sys.modules, "director_ai.core.scoring.nli", fake_nli)

        wrapper_cls = getattr(backends_mod, wrapper_name)
        wrapper = wrapper_cls(device="cpu", cache_dir="/tmp/director-ai")

        assert wrapper.score("premise", "hypothesis") == 0.44
        assert wrapper.score_batch([("p1", "h1"), ("p2", "h2")]) == [0.45, 0.45]
        assert created == {
            "backend": expected_backend,
            "kwargs": {"device": "cpu", "cache_dir": "/tmp/director-ai"},
            "score": ("premise", "hypothesis"),
            "batch": [("p1", "h1"), ("p2", "h2")],
        }

    def test_rules_wrapper_selects_rules_file_and_forwards_scores(self, monkeypatch):
        import director_ai.core.scoring.backends as backends_mod

        created = {}

        class FakeRulesBackend:
            def __init__(self, *, rules_file):
                created["rules_file"] = rules_file

            def score(self, premise, hypothesis):
                created["score"] = (premise, hypothesis)
                return 0.12

            def score_batch(self, pairs):
                created["batch"] = pairs
                return [0.13 for _ in pairs]

        fake_rules = types.ModuleType("director_ai.core.scoring.rules_scorer")
        fake_rules.RulesBackend = FakeRulesBackend
        monkeypatch.setitem(
            sys.modules,
            "director_ai.core.scoring.rules_scorer",
            fake_rules,
        )

        wrapper = backends_mod.RulesBackendWrapper(rules_file="rules.yaml")

        assert wrapper.score("claim", "policy") == 0.12
        assert wrapper.score_batch([("claim 1", "policy 1")]) == [0.13]
        assert created == {
            "rules_file": "rules.yaml",
            "score": ("claim", "policy"),
            "batch": [("claim 1", "policy 1")],
        }

    def test_embed_wrapper_selects_model_device_cache_and_forwards_scores(
        self, monkeypatch
    ):
        import director_ai.core.scoring.backends as backends_mod

        created = {}

        class FakeEmbedBackend:
            def __init__(self, *, model_name, device, cache_dir):
                created["init"] = (model_name, device, cache_dir)

            def score(self, premise, hypothesis):
                created["score"] = (premise, hypothesis)
                return 0.23

            def score_batch(self, pairs):
                created["batch"] = pairs
                return [0.24 for _ in pairs]

        fake_embed = types.ModuleType("director_ai.core.scoring.embed_scorer")
        fake_embed.EmbedBackend = FakeEmbedBackend
        monkeypatch.setitem(
            sys.modules,
            "director_ai.core.scoring.embed_scorer",
            fake_embed,
        )

        wrapper = backends_mod.EmbedBackendWrapper(
            model_name="local/embed",
            device="cuda",
            cache_dir="/cache",
        )

        assert wrapper.score("premise", "hypothesis") == 0.23
        assert wrapper.score_batch([("p", "h")]) == [0.24]
        assert created == {
            "init": ("local/embed", "cuda", "/cache"),
            "score": ("premise", "hypothesis"),
            "batch": [("p", "h")],
        }

    def test_distilled_wrapper_selects_artifact_mode_and_forwards_scores(
        self, monkeypatch
    ):
        import director_ai.core.scoring.backends as backends_mod

        created = {}

        class FakeDistilledNLIBackend:
            def __init__(self, *, model_path, use_onnx, device):
                created["init"] = (model_path, use_onnx, device)

            def score(self, premise, hypothesis):
                created["score"] = (premise, hypothesis)
                return 0.34

            def score_batch(self, pairs):
                created["batch"] = pairs
                return [0.35 for _ in pairs]

        fake_distilled = types.ModuleType("director_ai.core.scoring.distilled_scorer")
        fake_distilled.DistilledNLIBackend = FakeDistilledNLIBackend
        monkeypatch.setitem(
            sys.modules,
            "director_ai.core.scoring.distilled_scorer",
            fake_distilled,
        )

        wrapper = backends_mod.DistilledNLIBackendWrapper(
            model_path="/models/nli-lite",
            use_onnx=False,
            device="cuda",
        )

        assert wrapper.score("premise", "hypothesis") == 0.34
        assert wrapper.score_batch([("p", "h")]) == [0.35]
        assert created == {
            "init": ("/models/nli-lite", False, "cuda"),
            "score": ("premise", "hypothesis"),
            "batch": [("p", "h")],
        }

    def test_optional_backend_registration_skips_missing_dependencies(
        self, monkeypatch
    ):
        import director_ai.core.scoring.backends as backends_mod

        original_find_spec = importlib.util.find_spec

        def fake_find_spec(name, *args, **kwargs):
            if name in {"sentence_transformers", "backfire_kernel"}:
                return None
            return original_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
        spec = importlib.util.spec_from_file_location(
            "director_ai.core.scoring._backends_optional_dependency_test",
            backends_mod.__file__,
        )
        assert spec is not None
        assert spec.loader is not None
        isolated = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(isolated)

        # Inspect the import-time registry directly; list_backends() may add
        # separately installed entry-point backends after module import.
        assert "embed" not in isolated._REGISTRY
        assert "rust" not in isolated._REGISTRY
        assert "backfire" not in isolated._REGISTRY


class TestEntryPointDiscovery:
    def test_entry_points_loaded_flag(self):
        import director_ai.core.backends as mod

        mod._load_entry_points()
        assert mod._ENTRY_POINTS_LOADED is True


def test_backend_registry_allows_explicit_replacement() -> None:
    from director_ai.core.backends import ScorerBackend, get_backend, register_backend

    class FirstBackend(ScorerBackend):
        def score(self, premise: str, hypothesis: str) -> float:
            return 0.1

        def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [0.1] * len(pairs)

    class SecondBackend(ScorerBackend):
        def score(self, premise: str, hypothesis: str) -> float:
            return 0.2

        def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [0.2] * len(pairs)

    register_backend("test_replacement_contract", FirstBackend)
    register_backend("test_replacement_contract", SecondBackend)

    assert get_backend("test_replacement_contract") is SecondBackend
