# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle coverage for NLI model scoring paths pipeline (STRONG)."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from director_ai.core.nli import NLIScorer


class TestModelLoading:
    def test_load_nli_model_success(self):
        mock_torch = MagicMock()
        mock_torch.float16 = "fp16"
        mock_torch.bfloat16 = "bf16"
        mock_torch.float32 = "fp32"
        mock_transformers = MagicMock()
        tok = MagicMock()
        model = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = tok
        auto_cls = mock_transformers.AutoModelForSequenceClassification
        auto_cls.from_pretrained.return_value = model

        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            from director_ai.core.nli import _load_nli_model

            _load_nli_model.cache_clear()
            t, m = _load_nli_model("test-model", device="cpu")
            assert t is tok
            assert m is not None

    def test_load_nli_model_with_dtype(self):
        mock_torch = MagicMock()
        mock_torch.float16 = "fp16"
        mock_torch.bfloat16 = "bf16"
        mock_torch.float32 = "fp32"
        mock_transformers = MagicMock()
        tok = MagicMock()
        model = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = tok
        auto_cls = mock_transformers.AutoModelForSequenceClassification
        auto_cls.from_pretrained.return_value = model

        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            from director_ai.core.nli import _load_nli_model

            _load_nli_model.cache_clear()
            t, m = _load_nli_model("test-model-fp16", torch_dtype="float16")
            assert t is tok

    def test_load_nli_model_import_error(self):
        with patch.dict(sys.modules, {"torch": None, "transformers": None}):
            from director_ai.core.nli import _load_nli_model

            _load_nli_model.cache_clear()
            t, m = _load_nli_model("no-model")
            assert t is None
            assert m is None


class TestModelScoring:
    def _scorer_with_mock_model(self):
        scorer = NLIScorer(use_model=True, backend="deberta")
        mock_tok = MagicMock()
        mock_model = MagicMock()

        import torch

        mock_param = torch.zeros(1)
        mock_model.parameters.return_value = iter([mock_param])

        logits_3class = torch.tensor([[0.1, 0.3, 0.6]])
        mock_model.return_value = SimpleNamespace(logits=logits_3class)
        mock_tok.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

        scorer._tokenizer = mock_tok
        scorer._model = mock_model
        scorer._model_name = "test/model"
        return scorer

    @pytest.mark.skipif(
        "torch" not in sys.modules and not any("torch" in p for p in sys.path),
        reason="torch not installed",
    )
    def test_model_score_3class(self):
        scorer = self._scorer_with_mock_model()
        result = scorer._model_score("premise", "hypothesis")
        assert 0.0 <= result <= 1.0

    @pytest.mark.skipif(
        "torch" not in sys.modules and not any("torch" in p for p in sys.path),
        reason="torch not installed",
    )
    def test_model_score_2class(self):
        import torch

        scorer = self._scorer_with_mock_model()
        logits_2class = torch.tensor([[0.3, 0.7]])
        scorer._model.return_value = SimpleNamespace(logits=logits_2class)
        result = scorer._model_score("premise", "hypothesis")
        assert 0.0 <= result <= 1.0

    @pytest.mark.skipif(
        "torch" not in sys.modules and not any("torch" in p for p in sys.path),
        reason="torch not installed",
    )
    def test_model_score_batch(self):
        scorer = self._scorer_with_mock_model()
        import torch

        logits = torch.tensor([[0.1, 0.3, 0.6], [0.8, 0.1, 0.1]])
        scorer._model.return_value = SimpleNamespace(logits=logits)
        scorer._tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.tensor([[1, 1], [1, 1]]),
        }
        results = scorer._model_score_batch([("a", "b"), ("c", "d")])
        assert len(results) == 2

    @pytest.mark.skipif(
        "torch" not in sys.modules and not any("torch" in p for p in sys.path),
        reason="torch not installed",
    )
    def test_model_score_factcg(self):
        scorer = self._scorer_with_mock_model()
        scorer._model_name = "yaxili96/FactCG-DeBERTa"
        result = scorer._model_score("premise", "hypothesis")
        assert 0.0 <= result <= 1.0

    def test_model_score_no_model_raises(self):
        scorer = NLIScorer(use_model=False)
        scorer._tokenizer = None
        scorer._model = None
        with pytest.raises(RuntimeError, match="not loaded"):
            scorer._model_score("a", "b")

    def test_model_score_batch_no_model_raises(self):
        scorer = NLIScorer(use_model=False)
        scorer._tokenizer = None
        scorer._model = None
        with pytest.raises(RuntimeError, match="not loaded"):
            scorer._model_score_batch([("a", "b")])


class TestOnnxScoring:
    def test_onnx_score_batch_no_session_raises(self):
        scorer = NLIScorer(use_model=False, backend="onnx")
        scorer._tokenizer = None
        scorer._onnx_session = None
        with pytest.raises(RuntimeError, match="not loaded"):
            scorer._onnx_score_batch([("a", "b")])


class TestDecomposeClaims:
    def test_decompose_claims(self):
        scorer = NLIScorer(use_model=False)
        claims = scorer.decompose_claims("The sky is blue. The grass is green.")
        assert len(claims) == 2

    def test_score_decomposed_empty(self):
        scorer = NLIScorer(use_model=False)
        agg, per = scorer.score_decomposed("premise", "single claim")
        assert 0.0 <= agg <= 1.0
        assert len(per) == 1

    def test_score_decomposed_multi(self):
        scorer = NLIScorer(use_model=False)
        agg, per = scorer.score_decomposed(
            "The sky is blue.",
            "The sky is blue. Water is wet. Fire is hot.",
        )
        assert len(per) == 3


class TestEnsureMinicheck:
    def test_ensure_minicheck_runtime_error(self):
        """RuntimeError in MiniCheck() triggers ROCm manual loading fallback.

        When the fallback also fails (minicheck.inference unavailable),
        _ensure_minicheck must return False.
        """
        scorer = NLIScorer(use_model=False, backend="minicheck")
        scorer._minicheck_loaded = False

        class FakeMiniCheck:
            def __init__(self, **kwargs):
                raise RuntimeError("init fail")

        mock_minicheck_mod = MagicMock()
        mock_minicheck_mod.MiniCheck = FakeMiniCheck
        # Block minicheck.inference so the manual-loading fallback
        # triggers ImportError and _ensure_minicheck returns False.
        blocked = {
            "minicheck": mock_minicheck_mod,
            "minicheck.minicheck": mock_minicheck_mod,
            "minicheck.inference": None,
        }
        with patch.dict(sys.modules, blocked):
            result = scorer._ensure_minicheck()
            assert result is False
