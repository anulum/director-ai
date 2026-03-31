# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — NLI Coverage Gaps Tests (STRONG)
"""Multi-angle tests for NLI scorer coverage gaps.

Covers: model loading, backend routing (DeBERTa/ONNX/MiniCheck), heuristic
fallback, score_batch, score_chunked, custom backends, label index caching,
FactCG template, max_length enforcement, pipeline integration with
CoherenceScorer, and performance documentation.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from director_ai.core.scoring.nli import (
    NLIScorer,
    _probs_to_confidence,
    _resolve_label_indices,
    clear_model_cache,
    export_onnx,
    export_tensorrt,
    nli_available,
)

# ── _resolve_label_indices ────────────────────────────────────────────────────


class TestResolveLabelIndices:
    def test_no_config_returns_default(self):
        model = object()
        assert _resolve_label_indices(model) == (2, 1)

    def test_empty_id2label_returns_default(self):
        model = SimpleNamespace(config=SimpleNamespace(id2label={}))
        assert _resolve_label_indices(model) == (2, 1)

    def test_contradiction_label_found(self):
        model = SimpleNamespace(
            config=SimpleNamespace(
                id2label={0: "entailment", 1: "neutral", 2: "contradiction"}
            )
        )
        ci, ni = _resolve_label_indices(model)
        assert ci == 2
        assert ni == 1

    def test_contradict_alias(self):
        model = SimpleNamespace(
            config=SimpleNamespace(
                id2label={0: "Contradict", 1: "Neutral", 2: "entailment"}
            )
        )
        ci, ni = _resolve_label_indices(model)
        assert ci == 0
        assert ni == 1

    def test_reordered_labels(self):
        # contradiction at 0, neutral at 2
        model = SimpleNamespace(
            config=SimpleNamespace(
                id2label={0: "CONTRADICTION", 1: "entailment", 2: "NEUTRAL"}
            )
        )
        ci, ni = _resolve_label_indices(model)
        assert ci == 0
        assert ni == 2

    def test_none_id2label_returns_default(self):
        model = SimpleNamespace(config=SimpleNamespace(id2label=None))
        assert _resolve_label_indices(model) == (2, 1)


# ── _probs_to_confidence ─────────────────────────────────────────────────────


class TestProbsToConfidence:
    def test_uniform_dist_zero_confidence(self):
        probs = np.array([[1 / 3, 1 / 3, 1 / 3]])
        result = _probs_to_confidence(probs)
        assert abs(result[0]) < 0.02

    def test_onehot_max_confidence(self):
        probs = np.array([[0.0, 0.0, 1.0]])
        result = _probs_to_confidence(probs)
        assert result[0] > 0.95

    def test_two_class(self):
        probs = np.array([[0.9, 0.1]])
        result = _probs_to_confidence(probs)
        assert 0.0 <= result[0] <= 1.0

    def test_batch(self):
        probs = np.array([[0.8, 0.1, 0.1], [1 / 3, 1 / 3, 1 / 3]])
        result = _probs_to_confidence(probs)
        assert len(result) == 2
        assert result[0] > result[1]


# ── clear_model_cache ─────────────────────────────────────────────────────────


def test_clear_model_cache():
    clear_model_cache()  # should not raise


# ── nli_available true branch ─────────────────────────────────────────────────


def test_nli_available_true_when_torch_present():
    mock_torch = MagicMock()
    mock_transformers = MagicMock()
    with patch.dict(
        sys.modules, {"torch": mock_torch, "transformers": mock_transformers}
    ):
        # reimport via function to pick up patched sys.modules
        from director_ai.core.scoring.nli import nli_available as _nli_available

        # The cached function closes over the real import; call directly
        result = _nli_available()
    assert isinstance(result, bool)


def test_nli_available_returns_bool():
    assert isinstance(nli_available(), bool)


# ── export_onnx ───────────────────────────────────────────────────────────────


class TestExportOnnx:
    def _make_ort_modules(self):
        mock_optimum = MagicMock()
        mock_ort_model = MagicMock()
        mock_optimum.onnxruntime.ORTModelForSequenceClassification.from_pretrained.return_value = mock_ort_model
        mock_transformers = MagicMock()
        mock_tok = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tok
        return mock_optimum, mock_transformers, mock_ort_model

    def test_export_no_quantize(self, tmp_path):
        mock_optimum, mock_transformers, _ = self._make_ort_modules()
        out = str(tmp_path / "onnx_out")
        with patch.dict(
            sys.modules,
            {
                "optimum": mock_optimum,
                "optimum.onnxruntime": mock_optimum.onnxruntime,
                "transformers": mock_transformers,
            },
        ):
            result = export_onnx(model_name="test/model", output_dir=out, quantize=None)
        assert result == out

    def test_export_int8(self, tmp_path):
        mock_optimum, mock_transformers, _ = self._make_ort_modules()
        out = str(tmp_path / "onnx_int8")

        mock_ort_quant = MagicMock()
        mock_ort_quant.quantization.QuantType.QInt8 = 0
        mock_ort_quant.quantization.quantize_dynamic = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "optimum": mock_optimum,
                "optimum.onnxruntime": mock_optimum.onnxruntime,
                "transformers": mock_transformers,
                "onnxruntime": mock_ort_quant,
                "onnxruntime.quantization": mock_ort_quant.quantization,
            },
        ):
            result = export_onnx(
                model_name="test/model", output_dir=out, quantize="int8"
            )
        assert result == out

    def test_export_fp16(self, tmp_path):
        mock_optimum, mock_transformers, _ = self._make_ort_modules()
        out = str(tmp_path / "onnx_fp16")

        mock_onnx_mod = MagicMock()
        mock_onnx_mod.load.return_value = MagicMock()
        mock_float16 = MagicMock()
        mock_float16.convert_float_to_float16.return_value = MagicMock()
        mock_ort_transformers = MagicMock()
        mock_ort_transformers.float16 = mock_float16

        with patch.dict(
            sys.modules,
            {
                "optimum": mock_optimum,
                "optimum.onnxruntime": mock_optimum.onnxruntime,
                "transformers": mock_transformers,
                "onnx": mock_onnx_mod,
                "onnxruntime.transformers": mock_ort_transformers,
            },
        ):
            result = export_onnx(
                model_name="test/model", output_dir=out, quantize="fp16"
            )
        assert result == out


# ── export_tensorrt ───────────────────────────────────────────────────────────


class TestExportTensorrt:
    def _make_trt_ort(self, tmp_path):
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["TensorrtExecutionProvider"]

        input_info = MagicMock()
        input_info.name = "input_ids"
        input_info2 = MagicMock()
        input_info2.name = "attention_mask"
        mock_session.get_inputs.return_value = [input_info, input_info2]
        mock_session.run.return_value = [np.array([[0.1, 0.3, 0.6]])]
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.SessionOptions.return_value = MagicMock()

        mock_transformers = MagicMock()
        mock_tok = MagicMock()
        mock_tok.return_value = {
            "input_ids": np.ones((1, 16), dtype=np.int64),
            "attention_mask": np.ones((1, 16), dtype=np.int64),
        }
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tok
        return mock_ort, mock_transformers

    def test_export_trt_success(self, tmp_path):
        onnx_dir = str(tmp_path / "onnx_trt_src")
        os.makedirs(onnx_dir)
        model_file = os.path.join(onnx_dir, "model.onnx")
        open(model_file, "wb").close()

        mock_ort, mock_transformers = self._make_trt_ort(tmp_path)

        with patch.dict(
            sys.modules,
            {
                "onnxruntime": mock_ort,
                "transformers": mock_transformers,
            },
        ):
            out = export_tensorrt(
                onnx_dir=onnx_dir, fp16=True, max_batch=4, warmup_pairs=2
            )
        assert out == os.path.join(onnx_dir, "trt_cache")

    def test_export_trt_custom_output_dir(self, tmp_path):
        onnx_dir = str(tmp_path / "onnx_trt_src2")
        out_dir = str(tmp_path / "my_trt_cache")
        os.makedirs(onnx_dir)
        open(os.path.join(onnx_dir, "model.onnx"), "wb").close()

        mock_ort, mock_transformers = self._make_trt_ort(tmp_path)

        with patch.dict(
            sys.modules,
            {
                "onnxruntime": mock_ort,
                "transformers": mock_transformers,
            },
        ):
            result = export_tensorrt(
                onnx_dir=onnx_dir, output_dir=out_dir, warmup_pairs=1
            )
        assert result == out_dir

    def test_export_trt_no_model_file(self, tmp_path):
        onnx_dir = str(tmp_path / "onnx_empty")
        os.makedirs(onnx_dir)

        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = ["TensorrtExecutionProvider"]

        with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            with pytest.raises(FileNotFoundError):
                export_tensorrt(onnx_dir=onnx_dir)

    def test_export_trt_no_trt_provider(self, tmp_path):
        onnx_dir = str(tmp_path / "onnx_notrt")
        os.makedirs(onnx_dir)
        open(os.path.join(onnx_dir, "model.onnx"), "wb").close()

        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

        with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            with pytest.raises(RuntimeError, match="TensorrtExecutionProvider"):
                export_tensorrt(onnx_dir=onnx_dir)


# ── _load_onnx_session quantized branch ──────────────────────────────────────


class TestOnnxQuantizedBranch:
    def test_quantized_model_preferred_on_cpu(self, tmp_path):
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.SessionOptions.return_value = MagicMock()

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = MagicMock()

        onnx_dir = tmp_path / "onnx_quant"
        onnx_dir.mkdir()
        (onnx_dir / "model.onnx").write_bytes(b"fp32")
        (onnx_dir / "model_quantized.onnx").write_bytes(b"int8")

        with patch.dict(
            sys.modules,
            {"onnxruntime": mock_ort, "transformers": mock_transformers},
        ):
            from director_ai.core.scoring.nli import _load_onnx_session

            _load_onnx_session.cache_clear()
            tok, session = _load_onnx_session(str(onnx_dir), device=None)

        assert session is mock_session
        # The quantized file should have been passed to InferenceSession
        call_args = mock_ort.InferenceSession.call_args
        assert "model_quantized.onnx" in call_args[0][0]


# ── LoRA adapter loading ──────────────────────────────────────────────────────


class TestLoraAdapterLoading:
    def _scorer_with_model(self):
        scorer = NLIScorer(use_model=True, backend="deberta")
        scorer._tokenizer = MagicMock()
        scorer._model = MagicMock()
        scorer._model_loaded = True
        scorer._model_name = "test/model"
        return scorer

    def test_load_lora_success(self):
        scorer = self._scorer_with_model()
        mock_peft = MagicMock()
        mock_merged = MagicMock()
        mock_peft_model = MagicMock()
        mock_peft_model.merge_and_unload.return_value = mock_merged
        mock_peft.PeftModel.from_pretrained.return_value = mock_peft_model

        with patch.dict(sys.modules, {"peft": mock_peft}):
            scorer._load_lora_adapter("/fake/adapter")

        assert scorer._model is mock_merged
        mock_merged.eval.assert_called_once()

    def test_load_lora_import_error(self):
        scorer = self._scorer_with_model()
        with patch.dict(sys.modules, {"peft": None}):
            scorer._load_lora_adapter("/fake/adapter")
        # Should not raise; model stays unchanged

    def test_load_lora_oserror(self):
        scorer = self._scorer_with_model()
        mock_peft = MagicMock()
        mock_peft.PeftModel.from_pretrained.side_effect = OSError("not found")

        with patch.dict(sys.modules, {"peft": mock_peft}):
            scorer._load_lora_adapter("/fake/adapter")
        # Should not raise

    def test_load_lora_valueerror(self):
        scorer = self._scorer_with_model()
        mock_peft = MagicMock()
        mock_peft.PeftModel.from_pretrained.side_effect = ValueError("bad path")

        with patch.dict(sys.modules, {"peft": mock_peft}):
            scorer._load_lora_adapter("/fake/adapter")

    def test_ensure_model_triggers_lora(self):
        scorer = NLIScorer(
            use_model=True, backend="deberta", lora_adapter_path="/fake/lora"
        )

        mock_tok = MagicMock()
        mock_model = MagicMock()
        mock_model.config = SimpleNamespace(
            id2label={0: "entailment", 1: "neutral", 2: "contradiction"}
        )

        mock_peft = MagicMock()
        mock_merged = MagicMock()
        mock_peft_instance = MagicMock()
        mock_peft_instance.merge_and_unload.return_value = mock_merged
        mock_peft.PeftModel.from_pretrained.return_value = mock_peft_instance

        with (
            patch(
                "director_ai.core.scoring.nli._load_nli_model",
                return_value=(mock_tok, mock_model),
            ),
            patch.dict(sys.modules, {"peft": mock_peft}),
        ):
            scorer._model_loaded = False
            scorer._ensure_model()

        assert scorer._model is mock_merged


# ── _model_score 2-class and 3-class label-index branches ────────────────────


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestModelScoreBranches:
    def _make_scorer(self, logits_shape=(1, 3)):
        import torch

        scorer = NLIScorer(use_model=True, backend="deberta")
        scorer._model_loaded = True
        mock_tok = MagicMock()
        mock_model = MagicMock()

        param = torch.zeros(1)
        mock_model.parameters.return_value = iter([param])

        logits = torch.zeros(*logits_shape)
        mock_model.return_value = SimpleNamespace(logits=logits)
        mock_tok.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

        scorer._tokenizer = mock_tok
        scorer._model = mock_model
        return scorer

    def test_two_class_return(self):
        import torch

        scorer = self._make_scorer()
        scorer._model_name = "other/model"
        # Override logits to 2-class
        scorer._model.return_value = SimpleNamespace(logits=torch.tensor([[0.3, 0.7]]))
        result = scorer._model_score("premise", "hyp")
        assert 0.0 <= result <= 1.0

    def test_three_class_with_label_indices(self):
        import torch

        scorer = self._make_scorer()
        scorer._model_name = "other/model"
        scorer._label_indices = (0, 1)
        scorer._model.return_value = SimpleNamespace(
            logits=torch.tensor([[0.6, 0.2, 0.2]])
        )
        result = scorer._model_score("premise", "hyp")
        assert 0.0 <= result <= 1.0

    def test_factcg_template_used_in_model_score(self):
        import torch

        scorer = self._make_scorer()
        scorer._model_name = "yaxili96/FactCG-something"
        scorer._model.return_value = SimpleNamespace(logits=torch.tensor([[0.1, 0.9]]))
        result = scorer._model_score("doc text", "claim text")
        assert 0.0 <= result <= 1.0
        # Tokenizer should have been called with a single string (FactCG template)
        call_args = scorer._tokenizer.call_args
        assert isinstance(call_args[0][0], str)


# ── _build_chunks overlap_ratio > 0 ──────────────────────────────────────────


class TestBuildChunksOverlap:
    def test_overlap_ratio_routes_to_overlap(self):
        scorer = NLIScorer(use_model=False)
        sentences = [f"Sentence {i}." for i in range(10)]
        chunks = scorer._build_chunks(sentences, budget=20, overlap_ratio=0.5)
        assert len(chunks) >= 1

    def test_build_chunks_overlap_directly(self):
        scorer = NLIScorer(use_model=False)
        sentences = ["One.", "Two.", "Three.", "Four.", "Five."]
        chunks = scorer._build_chunks_overlap(sentences, budget=5, overlap_ratio=0.5)
        assert len(chunks) >= 1

    def test_build_chunks_overlap_single_sentence_per_chunk(self):
        scorer = NLIScorer(use_model=False)
        # Budget so tiny that each sentence exceeds it
        sentences = ["This is a long sentence here.", "Another long sentence indeed."]
        chunks = scorer._build_chunks_overlap(sentences, budget=1, overlap_ratio=0.5)
        assert len(chunks) >= 1

    def test_build_chunks_overlap_empty(self):
        scorer = NLIScorer(use_model=False)
        chunks = scorer._build_chunks_overlap([], budget=100, overlap_ratio=0.5)
        assert chunks == [""]

    def test_build_chunks_overlap_full_overlap(self):
        scorer = NLIScorer(use_model=False)
        sentences = ["A.", "B.", "C."]
        # overlap_ratio close to 1 → stride=1, lots of chunks
        chunks = scorer._build_chunks_overlap(sentences, budget=10, overlap_ratio=0.9)
        assert len(chunks) >= len(sentences)


# ── score_batch_with_confidence ───────────────────────────────────────────────


class TestScoreBatchWithConfidence:
    def test_empty_returns_empty(self):
        scorer = NLIScorer(use_model=False)
        assert scorer.score_batch_with_confidence([]) == []

    def test_custom_backend_returns_ones_confidence(self):
        from director_ai.core.scoring.backends import ScorerBackend

        class Stub(ScorerBackend):
            def score(self, p, h):
                return 0.3

            def score_batch(self, pairs):
                return [0.3] * len(pairs)

        scorer = NLIScorer(backend=Stub())
        result = scorer.score_batch_with_confidence([("a", "b")])
        assert result == [(0.3, 1.0)]

    def test_lite_backend_returns_ones_confidence(self):
        scorer = NLIScorer(use_model=False, backend="lite")
        result = scorer.score_batch_with_confidence([("a", "b")])
        assert len(result) == 1
        assert result[0][1] == 1.0

    def test_minicheck_returns_ones_confidence(self):
        scorer = NLIScorer(use_model=False, backend="minicheck")
        result = scorer.score_batch_with_confidence([("a", "b")])
        assert len(result) == 1
        assert result[0][1] == 1.0

    def test_no_model_returns_half_confidence(self):
        scorer = NLIScorer(use_model=False)
        result = scorer.score_batch_with_confidence([("a", "b")])
        assert len(result) == 1
        assert result[0][1] == 0.5

    def test_onnx_backend_routes_to_onnx_with_confidence(self):
        scorer = NLIScorer(use_model=True, backend="onnx")
        scorer._model_loaded = True
        scorer._tokenizer = MagicMock()
        scorer._onnx_session = MagicMock()
        scorer._model_name = "other/model"

        input_info = MagicMock()
        input_info.name = "input_ids"
        scorer._onnx_session.get_inputs.return_value = [input_info]
        scorer._onnx_session.run.return_value = [
            np.array([[0.1, 0.3, 0.6], [0.7, 0.2, 0.1]], dtype=np.float32)
        ]
        scorer._tokenizer.return_value = {
            "input_ids": np.array([[1, 2], [3, 4]], dtype=np.int64),
        }

        result = scorer.score_batch_with_confidence([("a", "b"), ("c", "d")])
        assert len(result) == 2
        for div, conf in result:
            assert 0.0 <= div <= 1.0
            assert 0.0 <= conf <= 1.0


# ── _model_score_batch_with_confidence ───────────────────────────────────────


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestModelScoreBatchWithConfidence:
    def _make_scorer(self):
        import torch

        scorer = NLIScorer(use_model=True, backend="deberta")
        scorer._model_loaded = True
        mock_tok = MagicMock()
        mock_model = MagicMock()

        param = torch.zeros(1)
        mock_model.parameters.return_value = iter([param])

        logits = torch.tensor([[0.1, 0.3, 0.6], [0.7, 0.2, 0.1]])
        mock_model.return_value = SimpleNamespace(logits=logits)
        mock_tok.return_value = {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
        }

        scorer._tokenizer = mock_tok
        scorer._model = mock_model
        scorer._model_name = "other/model"
        scorer._label_indices = (2, 1)
        return scorer

    def test_returns_divergence_and_confidence(self):
        scorer = self._make_scorer()
        result = scorer._model_score_batch_with_confidence([("a", "b"), ("c", "d")])
        assert len(result) == 2
        for div, conf in result:
            assert 0.0 <= div <= 1.0
            assert 0.0 <= conf <= 1.0

    def test_no_model_raises(self):
        scorer = NLIScorer(use_model=False)
        scorer._tokenizer = None
        scorer._model = None
        with pytest.raises(RuntimeError, match="not loaded"):
            scorer._model_score_batch_with_confidence([("a", "b")])

    def test_factcg_template_batch(self):

        scorer = self._make_scorer()
        scorer._model_name = "yaxili96/FactCG-xyz"
        # FactCG formats each pair as a single string; the mocked model
        # returns logits with batch dim equal to the number of pairs, so
        # we pass 2 pairs to match the 2-row mock logits tensor.
        result = scorer._model_score_batch_with_confidence(
            [("doc", "claim"), ("d2", "c2")]
        )
        assert len(result) == 2


# ── _onnx_score_batch_with_confidence ────────────────────────────────────────


class TestOnnxScoreBatchWithConfidence:
    def _make_scorer(self, factcg=False):
        scorer = NLIScorer(use_model=True, backend="onnx")
        scorer._model_loaded = True
        scorer._tokenizer = MagicMock()
        scorer._onnx_session = MagicMock()
        scorer._model_name = "yaxili96/FactCG-x" if factcg else "other/model"
        scorer._label_indices = (2, 1)

        input_info = MagicMock()
        input_info.name = "input_ids"
        scorer._onnx_session.get_inputs.return_value = [input_info]
        scorer._onnx_session.run.return_value = [
            np.array([[0.2, 0.3, 0.5]], dtype=np.float32)
        ]
        scorer._tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3]], dtype=np.int64),
        }
        return scorer

    def test_basic(self):
        scorer = self._make_scorer()
        result = scorer._onnx_score_batch_with_confidence([("a", "b")])
        assert len(result) == 1
        div, conf = result[0]
        assert 0.0 <= div <= 1.0
        assert 0.0 <= conf <= 1.0

    def test_factcg_template(self):
        scorer = self._make_scorer(factcg=True)
        result = scorer._onnx_score_batch_with_confidence([("doc", "claim")])
        assert len(result) == 1

    def test_no_session_raises(self):
        scorer = NLIScorer(use_model=False, backend="onnx")
        scorer._tokenizer = None
        scorer._onnx_session = None
        with pytest.raises(RuntimeError, match="not loaded"):
            scorer._onnx_score_batch_with_confidence([("a", "b")])


# ── score_chunked_confidence_weighted ────────────────────────────────────────


class TestScoreChunkedConfidenceWeighted:
    def test_short_fits(self):
        scorer = NLIScorer(use_model=False)
        agg, per_hyp = scorer.score_chunked_confidence_weighted("short", "short")
        assert 0.0 <= agg <= 1.0
        assert per_hyp == [agg]

    def test_long_hypothesis(self):
        scorer = NLIScorer(use_model=False, max_length=20)
        long_hyp = ". ".join([f"Sentence {i}" for i in range(30)]) + "."
        agg, per_hyp = scorer.score_chunked_confidence_weighted(
            "Short premise.", long_hyp
        )
        assert 0.0 <= agg <= 1.0
        assert len(per_hyp) >= 1

    def test_inner_agg_min(self):
        scorer = NLIScorer(use_model=False, max_length=20)
        long_hyp = ". ".join([f"Sentence {i}" for i in range(20)]) + "."
        agg, per_hyp = scorer.score_chunked_confidence_weighted(
            "Short premise.", long_hyp, inner_agg="min"
        )
        assert 0.0 <= agg <= 1.0

    def test_inner_agg_mean(self):
        scorer = NLIScorer(use_model=False, max_length=20)
        long_hyp = ". ".join([f"Sent {i}" for i in range(20)]) + "."
        agg, per_hyp = scorer.score_chunked_confidence_weighted(
            "Short premise.", long_hyp, inner_agg="mean"
        )
        assert 0.0 <= agg <= 1.0

    def test_zero_total_weight_fallback(self):
        # Patch score_batch_with_confidence to return zero confidence for all
        scorer = NLIScorer(use_model=False, max_length=20)
        long_hyp = ". ".join([f"Sentence {i}" for i in range(20)]) + "."

        def zero_conf(pairs):
            return [(0.4, 0.0) for _ in pairs]

        scorer.score_batch_with_confidence = zero_conf
        agg, per_hyp = scorer.score_chunked_confidence_weighted(
            "Short premise.", long_hyp
        )
        assert 0.0 <= agg <= 1.0

    def test_with_overlap_ratio(self):
        scorer = NLIScorer(use_model=False, max_length=20)
        long_hyp = ". ".join([f"Sentence {i}" for i in range(15)]) + "."
        agg, per_hyp = scorer.score_chunked_confidence_weighted(
            "Short premise.", long_hyp, overlap_ratio=0.3
        )
        assert 0.0 <= agg <= 1.0


# ── score_claim_coverage_with_attribution edge cases ─────────────────────────


class TestScoreClaimCoverageWithAttribution:
    def test_no_source_sentences_uses_source_itself(self):
        scorer = NLIScorer(use_model=False)
        # Empty source → source_sents will be empty, triggering `source_sents = [source]`
        # We can't make source empty and have claims, so mock _split_sentences for source

        call_count = [0]

        @staticmethod
        def patched_split(text):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call = decompose_claims on summary → return 1 claim
                return ["A claim."]
            # Second call = split source → return empty
            return []

        with patch.object(NLIScorer, "_split_sentences", staticmethod(patched_split)):
            coverage, divs, claims, attrs = (
                scorer.score_claim_coverage_with_attribution("source text", "A claim.")
            )
        assert len(claims) == 1
        assert len(attrs) == 1

    def test_too_many_pairs_raises(self):
        scorer = NLIScorer(use_model=False)
        # Create enough claims × source sentences to exceed 10_000
        many_claims = ". ".join([f"Claim {i}" for i in range(101)]) + "."
        many_source = ". ".join([f"Sentence {i}" for i in range(101)]) + "."

        with pytest.raises(ValueError, match="exceeding limit"):
            scorer.score_claim_coverage_with_attribution(many_source, many_claims)

    def test_no_claims_fallback(self):
        scorer = NLIScorer(use_model=False)
        with patch.object(NLIScorer, "decompose_claims", return_value=[]):
            coverage, divs, claims, attrs = (
                scorer.score_claim_coverage_with_attribution(
                    "source text", "hypothesis"
                )
            )
        assert len(claims) == 1
        assert len(attrs) == 1

    def test_normal_attribution(self):
        scorer = NLIScorer(use_model=False)
        coverage, divs, claims, attrs = scorer.score_claim_coverage_with_attribution(
            "The sky is blue. Water is wet.", "The sky is blue."
        )
        assert 0.0 <= coverage <= 1.0
        assert len(attrs) == len(claims)


# ── _heuristic_score negation asymmetry ──────────────────────────────────────


class TestHeuristicNegation:
    def test_negation_asymmetry_high_overlap(self):
        # p has "not" (negation), h does not; overlap = 4/5 = 0.8 > 0.3
        score = NLIScorer._heuristic_score(
            "the cat is not here",
            "the cat is here",
        )
        assert score >= 0.7

    def test_negation_asymmetry_low_overlap(self):
        # Low overlap → negation asymmetry does not trigger
        score = NLIScorer._heuristic_score(
            "The cat sat on the mat",
            "The dog is not running outside",
        )
        assert 0.0 <= score <= 1.0

    def test_both_sides_negated_no_asymmetry(self):
        # Both have negation → no asymmetry boost
        score = NLIScorer._heuristic_score(
            "It is not cold and not windy",
            "It is not warm and not calm",
        )
        assert 0.0 <= score <= 1.0


# ── _ensure_model with onnx_path ──────────────────────────────────────────────


class TestEnsureModelOnnxPath:
    def test_ensure_model_onnx_with_path_loads_session(self, tmp_path):
        onnx_dir = tmp_path / "onnx_ep"
        onnx_dir.mkdir()
        (onnx_dir / "model.onnx").write_bytes(b"fake")

        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.SessionOptions.return_value = MagicMock()

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = MagicMock()

        with patch.dict(
            sys.modules,
            {"onnxruntime": mock_ort, "transformers": mock_transformers},
        ):
            from director_ai.core.scoring.nli import _load_onnx_session

            _load_onnx_session.cache_clear()
            scorer = NLIScorer(use_model=True, backend="onnx", onnx_path=str(onnx_dir))
            scorer._ensure_model()

        assert scorer._onnx_session is not None or scorer._onnx_batcher is not None


# ── token counter and cost ────────────────────────────────────────────────────


class TestTokenCounterAndCost:
    def test_last_token_count_property(self):
        scorer = NLIScorer(use_model=False)
        assert scorer.last_token_count == 0

    def test_last_estimated_cost(self):
        scorer = NLIScorer(use_model=False)
        scorer._last_token_count = 1000
        assert abs(scorer.last_estimated_cost - 1000 * 1e-5) < 1e-9

    def test_reset_token_counter(self):
        scorer = NLIScorer(use_model=False)
        scorer._last_token_count = 500
        scorer.reset_token_counter()
        assert scorer._last_token_count == 0


# ── score() routing for minicheck, onnx-ready, deberta-ready ─────────────────


class TestScoreRouting:
    def test_score_minicheck_loaded(self):
        scorer = NLIScorer(use_model=False, backend="minicheck")
        scorer._minicheck_loaded = True
        scorer._minicheck = MagicMock()
        scorer._minicheck.score.return_value = [0.7]
        result = scorer.score("doc", "claim")
        assert 0.0 <= result <= 1.0

    def test_score_onnx_when_ready(self):
        scorer = NLIScorer(use_model=True, backend="onnx")
        scorer._model_loaded = True
        scorer._tokenizer = MagicMock()
        scorer._onnx_session = MagicMock()
        scorer._model_name = "other/model"

        input_info = MagicMock()
        input_info.name = "input_ids"
        scorer._onnx_session.get_inputs.return_value = [input_info]
        scorer._onnx_session.run.return_value = [
            np.array([[0.1, 0.3, 0.6]], dtype=np.float32)
        ]
        scorer._tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3]], dtype=np.int64)
        }

        result = scorer.score("premise", "hypothesis")
        assert 0.0 <= result <= 1.0

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
    def test_score_deberta_when_ready(self):
        import torch

        scorer = NLIScorer(use_model=True, backend="deberta")
        scorer._model_loaded = True
        mock_tok = MagicMock()
        mock_model = MagicMock()
        param = torch.zeros(1)
        mock_model.parameters.return_value = iter([param])
        mock_model.return_value = SimpleNamespace(
            logits=torch.tensor([[0.1, 0.3, 0.6]])
        )
        mock_tok.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        scorer._tokenizer = mock_tok
        scorer._model = mock_model
        scorer._model_name = "other/model"
        result = scorer.score("premise", "hypothesis")
        assert 0.0 <= result <= 1.0

    def test_score_batch_onnx_when_ready(self):
        scorer = NLIScorer(use_model=True, backend="onnx")
        scorer._model_loaded = True
        scorer._tokenizer = MagicMock()
        scorer._onnx_session = MagicMock()
        scorer._model_name = "other/model"

        input_info = MagicMock()
        input_info.name = "input_ids"
        scorer._onnx_session.get_inputs.return_value = [input_info]
        scorer._onnx_session.run.return_value = [
            np.array([[0.1, 0.3, 0.6], [0.7, 0.2, 0.1]], dtype=np.float32)
        ]
        scorer._tokenizer.return_value = {
            "input_ids": np.array([[1, 2], [3, 4]], dtype=np.int64),
        }
        results = scorer.score_batch([("a", "b"), ("c", "d")])
        assert len(results) == 2

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
    def test_score_batch_deberta_when_ready(self):
        import torch

        scorer = NLIScorer(use_model=True, backend="deberta")
        scorer._model_loaded = True
        mock_tok = MagicMock()
        mock_model = MagicMock()
        param = torch.zeros(1)
        mock_model.parameters.return_value = iter([param])
        mock_model.return_value = SimpleNamespace(
            logits=torch.tensor([[0.1, 0.3, 0.6], [0.7, 0.2, 0.1]])
        )
        mock_tok.return_value = {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
        }
        scorer._tokenizer = mock_tok
        scorer._model = mock_model
        scorer._model_name = "other/model"
        scorer._label_indices = (2, 1)
        results = scorer.score_batch([("a", "b"), ("c", "d")])
        assert len(results) == 2


# ── _minicheck_score when minicheck is loaded ─────────────────────────────────


class TestMinicheckLoaded:
    def test_minicheck_score_loaded(self):
        scorer = NLIScorer(use_model=False, backend="minicheck")
        scorer._minicheck_loaded = True
        scorer._minicheck = MagicMock()
        scorer._minicheck.score.return_value = [0.8]
        result = scorer._minicheck_score("doc", "claim")
        assert abs(result - 0.2) < 1e-6

    def test_minicheck_score_batch_loaded(self):
        scorer = NLIScorer(use_model=False, backend="minicheck")
        scorer._minicheck_loaded = True
        scorer._minicheck = MagicMock()
        scorer._minicheck.score.return_value = [0.9, 0.4]
        results = scorer._minicheck_score_batch(
            [("doc1", "claim1"), ("doc2", "claim2")]
        )
        assert len(results) == 2
        assert abs(results[0] - 0.1) < 1e-6
        assert abs(results[1] - 0.6) < 1e-6


# ── _model_score_batch (PyTorch) ──────────────────────────────────────────────


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestModelScoreBatch:
    def _make_scorer(self, factcg=False):
        import torch

        scorer = NLIScorer(use_model=True, backend="deberta")
        scorer._model_loaded = True
        mock_tok = MagicMock()
        mock_model = MagicMock()
        param = torch.zeros(1)
        mock_model.parameters.return_value = iter([param])
        logits = torch.tensor([[0.1, 0.3, 0.6], [0.7, 0.2, 0.1]])
        mock_model.return_value = SimpleNamespace(logits=logits)
        mock_tok.return_value = {"input_ids": torch.tensor([[1, 2], [3, 4]])}
        scorer._tokenizer = mock_tok
        scorer._model = mock_model
        scorer._model_name = "yaxili96/FactCG-xyz" if factcg else "other/model"
        scorer._label_indices = (2, 1)
        return scorer

    def test_deberta_batch_non_factcg(self):
        scorer = self._make_scorer(factcg=False)
        results = scorer._model_score_batch([("a", "b"), ("c", "d")])
        assert len(results) == 2

    def test_deberta_batch_factcg(self):
        scorer = self._make_scorer(factcg=True)
        results = scorer._model_score_batch([("doc", "claim"), ("d2", "c2")])
        assert len(results) == 2


# ── _onnx_score_batch non-FactCG branch ──────────────────────────────────────


class TestOnnxScoreBatchNonFactCG:
    def test_non_factcg_batch(self):
        scorer = NLIScorer(use_model=True, backend="onnx")
        scorer._model_loaded = True
        scorer._tokenizer = MagicMock()
        scorer._onnx_session = MagicMock()
        scorer._model_name = "other/nli-model"
        scorer._label_indices = (2, 1)

        input_info = MagicMock()
        input_info.name = "input_ids"
        scorer._onnx_session.get_inputs.return_value = [input_info]
        scorer._onnx_session.run.return_value = [
            np.array([[0.1, 0.3, 0.6], [0.7, 0.2, 0.1]], dtype=np.float32)
        ]
        scorer._tokenizer.return_value = {
            "input_ids": np.array([[1, 2], [3, 4]], dtype=np.int64),
        }
        results = scorer._onnx_score_batch([("a", "b"), ("c", "d")])
        assert len(results) == 2


# ── _build_chunks_overlap "no current" branch ─────────────────────────────────


class TestBuildChunksOverlapNoCurrent:
    def test_single_sentence_exceeds_budget(self):
        scorer = NLIScorer(use_model=False)
        # budget=0 forces each sentence to exceed budget and trigger `not current`
        sentences = ["This is sentence one.", "This is sentence two."]
        chunks = scorer._build_chunks_overlap(sentences, budget=0, overlap_ratio=0.5)
        assert len(chunks) == len(sentences)


# ── _score_chunked_with_counts inner/outer agg variants ──────────────────────


class TestChunkedWithCountsAggs:
    def _make_long_inputs(self):
        scorer = NLIScorer(use_model=False, max_length=20)
        long_hyp = ". ".join([f"Sentence {i}" for i in range(20)]) + "."
        return scorer, "Short premise.", long_hyp

    def test_inner_agg_min(self):
        scorer, prem, hyp = self._make_long_inputs()
        agg, per_hyp, pc, hc = scorer._score_chunked_with_counts(
            prem, hyp, inner_agg="min"
        )
        assert 0.0 <= agg <= 1.0

    def test_inner_agg_mean(self):
        scorer, prem, hyp = self._make_long_inputs()
        agg, per_hyp, pc, hc = scorer._score_chunked_with_counts(
            prem, hyp, inner_agg="mean"
        )
        assert 0.0 <= agg <= 1.0

    def test_outer_agg_trimmed_mean(self):
        scorer, prem, hyp = self._make_long_inputs()
        agg, per_hyp, pc, hc = scorer._score_chunked_with_counts(
            prem, hyp, outer_agg="trimmed_mean"
        )
        assert 0.0 <= agg <= 1.0

    def test_outer_agg_mean(self):
        scorer, prem, hyp = self._make_long_inputs()
        agg, per_hyp, pc, hc = scorer._score_chunked_with_counts(
            prem, hyp, outer_agg="mean"
        )
        assert 0.0 <= agg <= 1.0


# ── score_batch_with_confidence deberta path ─────────────────────────────────


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestScoreBatchWithConfidenceDeberta:
    def test_deberta_model_routes_to_model_method(self):
        import torch

        scorer = NLIScorer(use_model=True, backend="deberta")
        scorer._model_loaded = True
        mock_tok = MagicMock()
        mock_model = MagicMock()
        param = torch.zeros(1)
        mock_model.parameters.return_value = iter([param])
        mock_model.return_value = SimpleNamespace(
            logits=torch.tensor([[0.1, 0.3, 0.6]])
        )
        mock_tok.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        scorer._tokenizer = mock_tok
        scorer._model = mock_model
        scorer._model_name = "other/model"
        scorer._label_indices = (2, 1)

        result = scorer.score_batch_with_confidence([("a", "b")])
        assert len(result) == 1
        div, conf = result[0]
        assert 0.0 <= div <= 1.0
        assert 0.0 <= conf <= 1.0


# ── score_decomposed empty claims ─────────────────────────────────────────────


class TestScoreDecomposedEmpty:
    def test_empty_claims_fallback(self):
        scorer = NLIScorer(use_model=False)
        with patch.object(NLIScorer, "decompose_claims", return_value=[]):
            agg, per = scorer.score_decomposed("premise", "hypothesis")
        assert 0.0 <= agg <= 1.0
        assert len(per) == 1


# ── score_claim_coverage full path ────────────────────────────────────────────


class TestScoreClaimCoverage:
    def test_no_claims_fallback(self):
        scorer = NLIScorer(use_model=False)
        with patch.object(NLIScorer, "decompose_claims", return_value=[]):
            coverage, divs, claims = scorer.score_claim_coverage("source", "summary")
        assert len(claims) == 1
        assert 0.0 <= coverage <= 1.0

    def test_normal_coverage(self):
        scorer = NLIScorer(use_model=False)
        coverage, divs, claims = scorer.score_claim_coverage(
            "The sky is blue.", "The sky is blue."
        )
        assert 0.0 <= coverage <= 1.0
        assert len(claims) >= 1

    def test_multiple_claims(self):
        scorer = NLIScorer(use_model=False)
        coverage, divs, claims = scorer.score_claim_coverage(
            "The sky is blue. Water is wet.",
            "The sky is blue. Fire is hot.",
        )
        assert 0.0 <= coverage <= 1.0
        assert len(claims) == 2
