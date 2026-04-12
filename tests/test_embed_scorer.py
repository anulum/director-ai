# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""STRONG tests for ``director_ai.core.scoring.embed_scorer``.

Covers construction, lazy loading, scoring, batch scoring, edge cases,
and backend registry integration. Uses mocked SentenceTransformer to
avoid downloading real models in CI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from director_ai.core.scoring.embed_scorer import DEFAULT_EMBED_MODEL, EmbedBackend

# ── Mock helpers ────────────────────────────────────────────────────────


def _mock_st(similarity: float = 0.8):
    """Create a mock SentenceTransformer returning controlled embeddings."""
    mock_model = MagicMock()

    def encode_fn(texts, **kwargs):
        # Return normalised vectors with controlled cosine similarity
        n = len(texts)
        vecs = np.zeros((n, 10), dtype=np.float32)
        for i in range(n):
            vecs[i, 0] = 1.0  # base direction
            if i % 2 == 1:
                # Rotate second vector to get desired similarity
                angle = np.arccos(np.clip(similarity, -1, 1))
                vecs[i, 0] = np.cos(angle)
                vecs[i, 1] = np.sin(angle)
            # Normalise
            norm = np.linalg.norm(vecs[i])
            if norm > 0:
                vecs[i] /= norm
        return vecs

    mock_model.encode.side_effect = encode_fn
    return mock_model


# ── Construction ────────────────────────────────────────────────────────


class TestConstruction:
    def test_default_model_name(self):
        b = EmbedBackend()
        assert b._model_name == DEFAULT_EMBED_MODEL

    def test_custom_model(self):
        b = EmbedBackend(model_name="custom/model")
        assert b._model_name == "custom/model"

    def test_lazy_no_import_at_init(self):
        """Model is NOT loaded at __init__ time."""
        b = EmbedBackend()
        assert b._model is None

    def test_missing_sentence_transformers_raises(self):
        b = EmbedBackend()
        with (
            patch.dict("sys.modules", {"sentence_transformers": None}),
            pytest.raises(ImportError, match="sentence-transformers"),
        ):
            b._ensure_model()


# ── Scoring ─────────────────────────────────────────────────────────────


class TestScoring:
    def _backend(self, similarity=0.8):
        b = EmbedBackend()
        b._model = _mock_st(similarity)
        return b

    def test_score_returns_float(self):
        b = self._backend()
        s = b.score("premise", "hypothesis")
        assert isinstance(s, float)

    def test_score_range(self):
        b = self._backend(0.5)
        s = b.score("a", "b")
        assert 0.0 <= s <= 1.0

    def test_high_similarity(self):
        b = self._backend(0.95)
        s = b.score("Water boils at 100.", "Water boils at 100.")
        assert s > 0.9

    def test_low_similarity(self):
        b = self._backend(0.1)
        s = b.score("alpha", "omega")
        assert s < 0.3

    def test_identical_texts_high(self):
        b = self._backend(1.0)
        s = b.score("same", "same")
        assert s > 0.99


# ── Batch scoring ───────────────────────────────────────────────────────


class TestBatchScoring:
    def test_empty_batch(self):
        b = EmbedBackend()
        b._model = _mock_st()
        assert b.score_batch([]) == []

    def test_batch_length(self):
        b = EmbedBackend()
        b._model = _mock_st(0.7)
        scores = b.score_batch([("a", "b"), ("c", "d"), ("e", "f")])
        assert len(scores) == 3

    def test_batch_scores_in_range(self):
        b = EmbedBackend()
        b._model = _mock_st(0.6)
        scores = b.score_batch([("a", "b")] * 5)
        for s in scores:
            assert 0.0 <= s <= 1.0


# ── Backend registry ───────────────────────────────────────────────────


class TestRegistry:
    def test_embed_backend_registered(self):
        """embed backend should be registered if sentence-transformers is installed."""
        from director_ai.core.scoring.backends import list_backends

        backends = list_backends()
        # sentence-transformers is installed in our venv
        assert "embed" in backends

    def test_embed_backend_wraps_correctly(self):
        from director_ai.core.scoring.backends import get_backend

        cls = get_backend("embed")
        assert cls is not None
