# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real-surface coverage for the profile-wired embedding scorer."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Any, ClassVar, cast

import numpy as np
import pytest

from director_ai.core.config import DirectorConfig
from director_ai.core.retrieval.knowledge import GroundTruthStore
from director_ai.core.scoring.backends import EmbedBackendWrapper, register_backend
from director_ai.core.scoring.nli import NLIScorer
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


@dataclass(frozen=True)
class _EncodeCall:
    """A captured fake encoder invocation."""

    texts: tuple[str, ...]
    kwargs: dict[str, object]


class _FakeSentenceTransformer:
    """Deterministic local stand-in for sentence-transformers."""

    calls: ClassVar[list[_EncodeCall]] = []
    initialisations: ClassVar[list[dict[str, object]]] = []

    def __init__(
        self,
        model_name: str,
        *,
        device: str,
        cache_folder: str | None,
    ) -> None:
        self.initialisations.append(
            {
                "model_name": model_name,
                "device": device,
                "cache_folder": cache_folder,
            }
        )

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray[Any, Any]:
        """Return normalised semantic vectors for the requested texts."""
        self.calls.append(_EncodeCall(tuple(texts), dict(kwargs)))
        vectors = np.array([_semantic_vector(text) for text in texts], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return cast(np.ndarray[Any, Any], vectors / norms)


def _semantic_vector(text: str) -> list[float]:
    """Map scorer text to a stable local embedding vector."""
    lowered = text.lower()
    return [
        1.0,
        1.0 if "refund" in lowered or "policy" in lowered else 0.0,
        1.0 if "signed" in lowered or "approval" in lowered else 0.0,
        1.0 if "billing" in lowered or "finance" in lowered else 0.0,
    ]


def _install_fake_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Install a local fake sentence-transformers module."""
    _FakeSentenceTransformer.calls = []
    _FakeSentenceTransformer.initialisations = []
    module = ModuleType("sentence_transformers")
    module.__spec__ = ModuleSpec("sentence_transformers", loader=None)
    fake_module = cast(Any, module)
    fake_module.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_embed_scorer_unit_guard_declares_this_companion() -> None:
    """The embed scorer guard must point at this production-surface companion."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_embed_scorer.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_embed_scorer_real_surface.py" in reason


def test_embed_profile_build_scorer_uses_registered_embedding_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public embed profile should review through the embed backend."""
    _install_fake_sentence_transformers(monkeypatch)
    register_backend("embed", EmbedBackendWrapper)

    cfg = DirectorConfig.from_profile("embed")
    cfg.w_logic = 0.6
    cfg.w_fact = 0.4
    scorer = cfg.build_scorer(store=GroundTruthStore())

    assert scorer.scorer_backend == "embed"
    assert scorer._nli is not None
    assert isinstance(scorer._nli, NLIScorer)
    assert scorer._nli.backend == "__custom__"

    approved, score = scorer.review(
        "Refund policy requires signed approval.",
        "Refund policy requires signed approval.",
    )

    assert approved is True
    assert score.approved is True
    assert score.h_logical == pytest.approx(0.0, abs=1e-6)
    assert score.h_factual == pytest.approx(0.5)
    assert score.score > cfg.coherence_threshold
    assert _FakeSentenceTransformer.initialisations == [
        {
            "model_name": "BAAI/bge-small-en-v1.5",
            "device": "cpu",
            "cache_folder": None,
        }
    ]
    assert _FakeSentenceTransformer.calls
    assert _FakeSentenceTransformer.calls[0].kwargs == {
        "normalize_embeddings": True,
        "show_progress_bar": False,
    }


def test_embed_profile_preserves_registry_divergence_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The embed wrapper should expose divergence rather than similarity."""
    _install_fake_sentence_transformers(monkeypatch)
    register_backend("embed", EmbedBackendWrapper)

    cfg = DirectorConfig.from_profile("embed")
    cfg.w_logic = 0.6
    cfg.w_fact = 0.4
    scorer = cfg.build_scorer(store=GroundTruthStore())

    related = scorer.calculate_logical_divergence(
        "Refund policy requires signed approval.",
        "Refund policy requires signed approval.",
    )
    unrelated = scorer.calculate_logical_divergence(
        "Refund policy requires signed approval.",
        "Finance billing desk owns the invoice queue.",
    )

    assert related == pytest.approx(0.0, abs=1e-6)
    assert unrelated > related
