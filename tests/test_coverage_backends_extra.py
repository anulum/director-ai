# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Backend Wrappers Tests
"""Multi-angle tests for built-in backend wrappers.

Covers: DeBERTaBackend, OnnxBackend, MiniCheckBackend instantiation
(use_model=False heuristic mode), scoring, batch scoring, score ranges,
entry point loading, parametrised backends, and pipeline performance.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from director_ai.core.backends import (
    DeBERTaBackend,
    DistilledNLIBackendWrapper,
    EmbedBackendWrapper,
    MiniCheckBackend,
    OnnxBackend,
    RulesBackendWrapper,
    RustBackend,
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


class TestOptionalBackendWrappers:
    def test_optional_auto_registration_branches_under_controlled_specs(
        self, monkeypatch
    ):
        import director_ai.core.scoring.backends as bmod

        original_find_spec = importlib.util.find_spec

        def no_optional_specs(name):
            if name in {"sentence_transformers", "backfire_kernel"}:
                return None
            return original_find_spec(name)

        monkeypatch.setattr(importlib.util, "find_spec", no_optional_specs)
        reloaded = importlib.reload(bmod)
        assert "embed" not in reloaded._REGISTRY
        assert "rust" not in reloaded._REGISTRY

        class BackfireConfig:
            pass

        class RustCoherenceScorer:
            pass

        fake_backfire = ModuleType("backfire_kernel")
        fake_backfire.BackfireConfig = BackfireConfig
        fake_backfire.RustCoherenceScorer = RustCoherenceScorer
        monkeypatch.setitem(sys.modules, "backfire_kernel", fake_backfire)

        def available_optional_specs(name):
            if name in {"sentence_transformers", "backfire_kernel"}:
                return SimpleNamespace(name=name)
            return original_find_spec(name)

        monkeypatch.setattr(importlib.util, "find_spec", available_optional_specs)
        reloaded = importlib.reload(bmod)

        assert reloaded._REGISTRY["embed"] is reloaded.EmbedBackendWrapper
        assert reloaded._REGISTRY["rust"] is reloaded.RustBackend
        assert reloaded._REGISTRY["backfire"] is reloaded.RustBackend

    def test_rust_backend_uses_threshold_callback_and_score_object(self, monkeypatch):
        captured = {}

        class BackfireConfig:
            def __init__(self, *, coherence_threshold):
                captured["threshold"] = coherence_threshold

        class RustCoherenceScorer:
            def __init__(self, *, config, knowledge_callback):
                captured["config"] = config
                captured["knowledge_callback"] = knowledge_callback

            def review(self, premise, hypothesis):
                captured["review"] = (premise, hypothesis)
                return True, SimpleNamespace(score=0.73)

        module = ModuleType("backfire_kernel")
        module.BackfireConfig = BackfireConfig
        module.RustCoherenceScorer = RustCoherenceScorer
        monkeypatch.setitem(sys.modules, "backfire_kernel", module)

        backend = RustBackend(threshold=0.42, knowledge_callback=lambda _: "fact")

        assert backend.score("premise", "hypothesis") == 0.73
        assert backend.score_batch([("a", "b"), ("c", "d")]) == [0.73, 0.73]
        assert captured["threshold"] == 0.42
        assert callable(captured["knowledge_callback"])
        assert captured["review"] == ("c", "d")

    def test_rust_backend_accepts_numeric_score_object(self, monkeypatch):
        class BackfireConfig:
            def __init__(self, *, coherence_threshold):
                self.coherence_threshold = coherence_threshold

        class RustCoherenceScorer:
            def __init__(self, *, config, knowledge_callback):
                self.config = config
                self.knowledge_callback = knowledge_callback

            def review(self, premise, hypothesis):
                return False, 0.25

        module = ModuleType("backfire_kernel")
        module.BackfireConfig = BackfireConfig
        module.RustCoherenceScorer = RustCoherenceScorer
        monkeypatch.setitem(sys.modules, "backfire_kernel", module)

        assert RustBackend().score("premise", "hypothesis") == 0.25

    def test_rules_backend_wrapper_delegates_rules_file_and_scores(self, monkeypatch):
        captured = {}

        class RulesBackend:
            def __init__(self, *, rules_file):
                captured["rules_file"] = rules_file

            def score(self, premise, hypothesis):
                return 0.11 if premise == "p" and hypothesis == "h" else 0.22

            def score_batch(self, pairs):
                return [self.score(p, h) for p, h in pairs]

        module = ModuleType("director_ai.core.scoring.rules_scorer")
        module.RulesBackend = RulesBackend
        monkeypatch.setitem(
            sys.modules, "director_ai.core.scoring.rules_scorer", module
        )

        backend = RulesBackendWrapper(rules_file="/tmp/rules.yaml")

        assert captured["rules_file"] == "/tmp/rules.yaml"
        assert backend.score("p", "h") == 0.11
        assert backend.score_batch([("p", "h"), ("x", "y")]) == [0.11, 0.22]

    def test_embed_backend_wrapper_delegates_constructor_and_scores(self, monkeypatch):
        captured = {}

        class EmbedBackend:
            def __init__(self, *, model_name, device, cache_dir):
                captured.update(
                    {"model_name": model_name, "device": device, "cache_dir": cache_dir}
                )

            def score(self, premise, hypothesis):
                return 0.31

            def score_batch(self, pairs):
                return [0.31 for _ in pairs]

        module = ModuleType("director_ai.core.scoring.embed_scorer")
        module.EmbedBackend = EmbedBackend
        monkeypatch.setitem(
            sys.modules, "director_ai.core.scoring.embed_scorer", module
        )

        backend = EmbedBackendWrapper(
            model_name="local-embed",
            device="cuda",
            cache_dir="/models",
        )

        assert captured == {
            "model_name": "local-embed",
            "device": "cuda",
            "cache_dir": "/models",
        }
        assert backend.score("p", "h") == 0.31
        assert backend.score_batch([("p", "h")]) == [0.31]

    def test_distilled_backend_wrapper_delegates_constructor_and_scores(
        self, monkeypatch
    ):
        captured = {}

        class DistilledNLIBackend:
            def __init__(self, *, model_path, use_onnx, device):
                captured.update(
                    {"model_path": model_path, "use_onnx": use_onnx, "device": device}
                )

            def score(self, premise, hypothesis):
                return 0.41

            def score_batch(self, pairs):
                return [0.41 for _ in pairs]

        module = ModuleType("director_ai.core.scoring.distilled_scorer")
        module.DistilledNLIBackend = DistilledNLIBackend
        monkeypatch.setitem(
            sys.modules, "director_ai.core.scoring.distilled_scorer", module
        )

        backend = DistilledNLIBackendWrapper(
            model_path="local-distilled",
            use_onnx=False,
            device="cuda",
        )

        assert captured == {
            "model_path": "local-distilled",
            "use_onnx": False,
            "device": "cuda",
        }
        assert backend.score("p", "h") == 0.41
        assert backend.score_batch([("p", "h"), ("a", "b")]) == [0.41, 0.41]
