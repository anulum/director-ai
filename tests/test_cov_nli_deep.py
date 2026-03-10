"""Deep coverage for nli.py — ONNX loading, quantize, FactCG, score routing."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np

from director_ai.core.nli import NLIScorer


class TestOnnxSessionLoading:
    def test_load_onnx_session_success(self, tmp_path):
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.SessionOptions.return_value = MagicMock()

        mock_transformers = MagicMock()
        mock_tok = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tok

        onnx_dir = tmp_path / "onnx_model"
        onnx_dir.mkdir()
        (onnx_dir / "model.onnx").write_bytes(b"fake")

        with patch.dict(
            sys.modules,
            {
                "onnxruntime": mock_ort,
                "transformers": mock_transformers,
            },
        ):
            from director_ai.core.nli import _load_onnx_session

            _load_onnx_session.cache_clear()
            tok, session = _load_onnx_session(str(onnx_dir))
            assert tok is mock_tok
            assert session is mock_session

    def test_load_onnx_session_cuda(self, tmp_path):
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["CUDAExecutionProvider"]
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.SessionOptions.return_value = MagicMock()

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = MagicMock()

        onnx_dir = tmp_path / "onnx_cuda"
        onnx_dir.mkdir()
        (onnx_dir / "model.onnx").write_bytes(b"fake")

        with patch.dict(
            sys.modules,
            {
                "onnxruntime": mock_ort,
                "transformers": mock_transformers,
            },
        ):
            from director_ai.core.nli import _load_onnx_session

            _load_onnx_session.cache_clear()
            tok, session = _load_onnx_session(str(onnx_dir), device="cuda:0")
            assert session is mock_session

    def test_load_onnx_session_trt(self, tmp_path):
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = [
            "TensorrtExecutionProvider",
            "CPUExecutionProvider",
        ]
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["TensorrtExecutionProvider"]
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.SessionOptions.return_value = MagicMock()

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = MagicMock()

        onnx_dir = tmp_path / "onnx_trt"
        onnx_dir.mkdir()
        (onnx_dir / "model.onnx").write_bytes(b"fake")

        with (
            patch.dict(
                sys.modules,
                {"onnxruntime": mock_ort, "transformers": mock_transformers},
            ),
            patch.dict(os.environ, {"DIRECTOR_ENABLE_TRT": "1"}),
        ):
            from director_ai.core.nli import _load_onnx_session

            _load_onnx_session.cache_clear()
            tok, session = _load_onnx_session(str(onnx_dir))
            assert session is mock_session

    def test_load_onnx_not_a_dir(self, tmp_path):
        bogus = str(tmp_path / "does_not_exist_abc123")
        with patch.dict(
            sys.modules,
            {
                "onnxruntime": MagicMock(),
                "transformers": MagicMock(),
            },
        ):
            from director_ai.core.nli import _load_onnx_session

            _load_onnx_session.cache_clear()
            tok, session = _load_onnx_session(bogus)
            assert tok is None
            assert session is None

    def test_load_onnx_fallback_file(self, tmp_path):
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.SessionOptions.return_value = MagicMock()

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = MagicMock()

        onnx_dir = tmp_path / "onnx_alt"
        onnx_dir.mkdir()
        (onnx_dir / "custom_model.onnx").write_bytes(b"fake")

        with patch.dict(
            sys.modules,
            {
                "onnxruntime": mock_ort,
                "transformers": mock_transformers,
            },
        ):
            from director_ai.core.nli import _load_onnx_session

            _load_onnx_session.cache_clear()
            tok, session = _load_onnx_session(str(onnx_dir))
            assert session is mock_session


class TestQuantize8bit:
    def test_load_with_quantize(self):
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

        bnb_config = MagicMock()
        mock_transformers.BitsAndBytesConfig.return_value = bnb_config

        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            from director_ai.core.nli import _load_nli_model

            _load_nli_model.cache_clear()
            t, m = _load_nli_model("quant-model", quantize_8bit=True)
            assert t is tok

    def test_load_quantize_no_bitsandbytes(self):
        mock_torch = MagicMock()
        mock_torch.float16 = "fp16"
        mock_torch.bfloat16 = "bf16"
        mock_torch.float32 = "fp32"
        mock_transformers = MagicMock()
        mock_transformers.BitsAndBytesConfig = MagicMock(
            side_effect=ImportError("no bnb")
        )
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
            t, m = _load_nli_model("quant-model-nobnb", quantize_8bit=True)
            assert t is tok


class TestScoreRouting:
    def test_score_onnx_routing(self):
        scorer = NLIScorer(use_model=True, backend="onnx")
        scorer._tokenizer = MagicMock()
        scorer._model = None
        scorer._onnx_session = MagicMock()

        input_info = MagicMock()
        input_info.name = "input_ids"
        scorer._onnx_session.get_inputs.return_value = [input_info]

        fake_logits = np.array([[0.1, 0.3, 0.6]], dtype=np.float32)
        scorer._onnx_session.run.return_value = [fake_logits]
        scorer._tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3]], dtype=np.int64),
        }

        result = scorer.score("premise", "hypothesis")
        assert 0.0 <= result <= 1.0

    def test_score_batch_onnx_routing(self):
        scorer = NLIScorer(use_model=True, backend="onnx")
        scorer._tokenizer = MagicMock()
        scorer._model = None
        scorer._onnx_session = MagicMock()

        input_info = MagicMock()
        input_info.name = "input_ids"
        scorer._onnx_session.get_inputs.return_value = [input_info]

        fake_logits = np.array([[0.1, 0.3, 0.6], [0.8, 0.1, 0.1]], dtype=np.float32)
        scorer._onnx_session.run.return_value = [fake_logits]
        scorer._tokenizer.return_value = {
            "input_ids": np.array([[1, 2], [3, 4]], dtype=np.int64),
        }

        results = scorer.score_batch([("a", "b"), ("c", "d")])
        assert len(results) == 2

    def test_minicheck_score_batch(self):
        scorer = NLIScorer(use_model=False, backend="minicheck")
        scorer._minicheck_loaded = True
        scorer._minicheck = MagicMock()
        scorer._minicheck.score.return_value = [0.8, 0.6]

        results = scorer.score_batch([("a", "b"), ("c", "d")])
        assert len(results) == 2


class TestOnnxScoreBatchFactCG:
    def test_onnx_factcg_batch(self):
        scorer = NLIScorer(use_model=True, backend="onnx")
        scorer._model_name = "yaxili96/FactCG-DeBERTa"
        scorer._tokenizer = MagicMock()
        scorer._onnx_session = MagicMock()

        input_info = MagicMock()
        input_info.name = "input_ids"
        scorer._onnx_session.get_inputs.return_value = [input_info]

        fake_logits = np.array([[0.2, 0.8]], dtype=np.float32)
        scorer._onnx_session.run.return_value = [fake_logits]
        scorer._tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3]], dtype=np.int64),
        }

        results = scorer._onnx_score_batch([("doc", "claim")])
        assert len(results) == 1


class TestNliAvailable:
    def test_nli_available_false(self):
        with patch.dict(sys.modules, {"torch": None, "transformers": None}):
            from director_ai.core.nli import nli_available

            assert nli_available() is False
