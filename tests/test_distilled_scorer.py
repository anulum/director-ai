# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""STRONG tests for ``director_ai.core.scoring.distilled_scorer``.

Covers construction, lazy loading, ONNX/PyTorch inference paths,
softmax utility, batch scoring, and backend registry integration.
Uses mocks to avoid downloading real models.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from director_ai.core.scoring.distilled_scorer import (
    DEFAULT_DISTILLED_MODEL,
    DistilledNLIBackend,
    _softmax,
)

# ── _softmax utility ───────────────────────────────────────────────────


class TestSoftmax:
    def test_uniform(self):
        result = _softmax(np.array([0.0, 0.0]))
        np.testing.assert_allclose(result, [0.5, 0.5])

    def test_dominant(self):
        result = _softmax(np.array([10.0, 0.0]))
        assert result[0] > 0.99
        assert result[1] < 0.01

    def test_sums_to_one(self):
        result = _softmax(np.array([1.0, 2.0, 3.0]))
        assert abs(result.sum() - 1.0) < 1e-6

    def test_negative_logits(self):
        result = _softmax(np.array([-10.0, -5.0]))
        assert result.sum() - 1.0 < 1e-6
        assert result[1] > result[0]


# ── Construction ────────────────────────────────────────────────────────


class TestConstruction:
    def test_default_model(self):
        b = DistilledNLIBackend()
        assert b._model_path == DEFAULT_DISTILLED_MODEL

    def test_custom_model(self):
        b = DistilledNLIBackend(model_path="/tmp/my-model")
        assert b._model_path == "/tmp/my-model"

    def test_lazy_no_load_at_init(self):
        b = DistilledNLIBackend()
        assert not b._ready
        assert b._session is None
        assert b._model is None

    def test_use_onnx_default(self):
        b = DistilledNLIBackend()
        assert b._use_onnx is True


# ── ONNX inference path (mocked) ───────────────────────────────────────


class TestOnnxPath:
    def _mock_backend(self):
        b = DistilledNLIBackend()
        # Mock ONNX session
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [
            MagicMock(name="input_ids"),
            MagicMock(name="attention_mask"),
        ]
        # Return logits [entailment=2.0, contradiction=-1.0] → P(ent)≈0.95
        mock_session.run.return_value = [np.array([[2.0, -1.0]])]
        b._session = mock_session

        # Mock tokeniser
        mock_tok = MagicMock()
        mock_tok.return_value = {
            "input_ids": np.array([[1, 2, 3]]),
            "attention_mask": np.array([[1, 1, 1]]),
        }
        b._tokeniser = mock_tok
        b._ready = True
        return b

    def test_score_returns_float(self):
        b = self._mock_backend()
        s = b.score("premise", "hypothesis")
        assert isinstance(s, float)

    def test_score_high_entailment(self):
        b = self._mock_backend()
        s = b.score("x", "y")
        assert s > 0.9  # logits [2, -1] → softmax ≈ [0.95, 0.05]

    def test_score_range(self):
        b = self._mock_backend()
        s = b.score("a", "b")
        assert 0.0 <= s <= 1.0

    def test_batch(self):
        b = self._mock_backend()
        scores = b.score_batch([("a", "b"), ("c", "d")])
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)

    def test_empty_batch(self):
        b = self._mock_backend()
        assert b.score_batch([]) == []


# ── PyTorch fallback path (mocked) ─────────────────────────────────────


class TestPyTorchPath:
    def _mock_backend(self):
        import torch

        b = DistilledNLIBackend(use_onnx=False)
        mock_model = MagicMock()
        mock_model.return_value = MagicMock(logits=torch.tensor([[2.0, -1.0]]))
        b._model = mock_model
        b._torch = torch

        mock_tok = MagicMock()
        mock_tok.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        b._tokeniser = mock_tok
        b._ready = True
        b._device = "cpu"
        return b

    def test_score_pytorch(self):
        b = self._mock_backend()
        s = b.score("premise", "hypothesis")
        assert isinstance(s, float)
        assert s > 0.9


# ── Backend registry ───────────────────────────────────────────────────


class TestRegistry:
    def test_nli_lite_registered(self):
        from director_ai.core.scoring.backends import list_backends

        assert "nli-lite" in list_backends()
