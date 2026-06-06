# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ONNX Backend Tests
"""Multi-angle tests for ONNX NLI backend and export.

Covers: heuristic fallback when ONNX unavailable, bad path handling,
batch scoring, export importability, native exporter guards, onnxruntime
provider check, parametrised inputs, score range invariants, pipeline
integration via CoherenceScorer, and performance documentation.
"""

from __future__ import annotations

import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from director_ai.core.nli import NLIScorer, export_onnx

_has_onnxruntime = False
try:
    import onnxruntime  # noqa: F401

    _has_onnxruntime = True
except ImportError:
    pass


# ── ONNX fallback ───────────────────────────────────────────────


class TestOnnxBackendFallback:
    """ONNX backend falls back to heuristic when session unavailable."""

    def test_no_onnx_path(self):
        scorer = NLIScorer(use_model=True, backend="onnx")
        assert scorer.model_available is False
        s = scorer.score("premise", "hypothesis")
        assert 0.0 <= s <= 1.0

    def test_bad_onnx_path(self):
        scorer = NLIScorer(use_model=True, backend="onnx", onnx_path="/no/such/dir")
        assert scorer.model_available is False

    def test_batch_fallback(self):
        scorer = NLIScorer(use_model=True, backend="onnx")
        results = scorer.score_batch([("a", "b"), ("c", "d")])
        assert len(results) == 2
        assert all(0.0 <= r <= 1.0 for r in results)

    @pytest.mark.parametrize(
        "premise,hypothesis",
        [
            ("The sky is blue", "The sky is blue"),
            ("", "empty premise"),
            ("test", ""),
            ("日本語", "response"),
        ],
    )
    def test_fallback_various_inputs(self, premise, hypothesis):
        scorer = NLIScorer(use_model=True, backend="onnx")
        s = scorer.score(premise, hypothesis)
        assert 0.0 <= s <= 1.0

    def test_fallback_deterministic(self):
        scorer = NLIScorer(use_model=True, backend="onnx")
        s1 = scorer.score("test", "test")
        s2 = scorer.score("test", "test")
        assert s1 == s2

    @pytest.mark.parametrize("batch_size", [1, 3, 5, 10])
    def test_batch_various_sizes(self, batch_size):
        scorer = NLIScorer(use_model=True, backend="onnx")
        pairs = [("p", "h")] * batch_size
        results = scorer.score_batch(pairs)
        assert len(results) == batch_size


# ── ONNX export ──────────────────────────────────────────────────


class TestExportOnnx:
    """export_onnx must be importable and use the native modern stack."""

    def test_importable(self):
        from director_ai.core import export_onnx as fn

        assert callable(fn)

    def test_rejects_unknown_quantize_mode(self, tmp_path):
        with pytest.raises(ValueError, match="quantize"):
            export_onnx(output_dir=str(tmp_path), quantize="nf4")

    def test_export_uses_torch_onnx_not_legacy_exporter(self, tmp_path, monkeypatch):
        fake_torch = ModuleType("torch")
        fake_torch.long = object()
        fake_torch.no_grad = MagicMock(return_value=nullcontext())
        fake_torch.ones = MagicMock(
            side_effect=lambda shape, dtype=None: {"shape": shape, "dtype": dtype},
        )
        fake_torch.nn = SimpleNamespace(Module=object)
        fake_torch.onnx = SimpleNamespace(export=MagicMock())
        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoTokenizer = SimpleNamespace(from_pretrained=MagicMock())
        fake_transformers.AutoModelForSequenceClassification = SimpleNamespace(
            from_pretrained=MagicMock()
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        model = MagicMock()
        model.config.save_pretrained = MagicMock()
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": fake_torch.ones((1, 4), dtype=fake_torch.long),
            "attention_mask": fake_torch.ones((1, 4), dtype=fake_torch.long),
        }

        with (
            patch.object(
                fake_transformers.AutoTokenizer,
                "from_pretrained",
                return_value=tokenizer,
            ) as tok_from_pretrained,
            patch.object(
                fake_transformers.AutoModelForSequenceClassification,
                "from_pretrained",
                return_value=model,
            ) as model_from_pretrained,
        ):
            result = export_onnx(
                model_name="test/model",
                output_dir=str(tmp_path),
                revision="abc123",
            )

        assert result == str(tmp_path)
        tok_from_pretrained.assert_called_once_with("test/model", revision="abc123")
        model_from_pretrained.assert_called_once_with("test/model", revision="abc123")
        fake_torch.onnx.export.assert_called_once()
        _, _, exported_path = fake_torch.onnx.export.call_args.args
        assert exported_path.endswith("model.onnx")

    def test_export_disables_dynamo_when_dynamic_axes_are_used(
        self,
        tmp_path,
        monkeypatch,
    ):
        fake_torch = ModuleType("torch")
        fake_torch.long = object()
        fake_torch.no_grad = MagicMock(return_value=nullcontext())
        fake_torch.ones = MagicMock(
            side_effect=lambda shape, dtype=None: {"shape": shape, "dtype": dtype},
        )
        fake_torch.nn = SimpleNamespace(Module=object)
        captured: dict[str, object] = {}

        def export(
            model,
            args,
            f,
            *,
            input_names,
            output_names,
            dynamic_axes,
            opset_version,
            do_constant_folding,
            dynamo,
        ):
            captured.update(
                {
                    "model": model,
                    "args": args,
                    "f": f,
                    "input_names": input_names,
                    "output_names": output_names,
                    "dynamic_axes": dynamic_axes,
                    "opset_version": opset_version,
                    "do_constant_folding": do_constant_folding,
                    "dynamo": dynamo,
                },
            )

        fake_torch.onnx = SimpleNamespace(export=export)
        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoTokenizer = SimpleNamespace(from_pretrained=MagicMock())
        fake_transformers.AutoModelForSequenceClassification = SimpleNamespace(
            from_pretrained=MagicMock()
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        model = MagicMock()
        model.config.save_pretrained = MagicMock()
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": fake_torch.ones((1, 4), dtype=fake_torch.long),
            "attention_mask": fake_torch.ones((1, 4), dtype=fake_torch.long),
        }

        with (
            patch.object(
                fake_transformers.AutoTokenizer,
                "from_pretrained",
                return_value=tokenizer,
            ),
            patch.object(
                fake_transformers.AutoModelForSequenceClassification,
                "from_pretrained",
                return_value=model,
            ),
        ):
            export_onnx(
                model_name="test/model",
                output_dir=str(tmp_path),
                revision="abc123",
            )

        assert captured["dynamo"] is False
        assert captured["dynamic_axes"]["input_ids"] == {0: "batch", 1: "sequence"}


# ── ONNX runtime ─────────────────────────────────────────────────


@pytest.mark.skipif(not _has_onnxruntime, reason="onnxruntime not installed")
class TestOnnxRuntimeAvailable:
    """Tests when onnxruntime is installed."""

    def test_providers_list(self):
        import onnxruntime as ort

        providers = ort.get_available_providers()
        assert "CPUExecutionProvider" in providers

    def test_ort_version(self):
        import onnxruntime as ort

        assert hasattr(ort, "__version__")


# ── Pipeline integration ─────────────────────────────────────────


class TestOnnxPipelineIntegration:
    """ONNX backend must integrate with CoherenceScorer."""

    def test_scorer_with_onnx_backend(self):
        from director_ai.core import CoherenceScorer

        scorer = CoherenceScorer(use_nli=False, scorer_backend="onnx")
        approved, score = scorer.review("test", "test")
        assert isinstance(approved, bool)
        assert 0.0 <= score.score <= 1.0


# ── Performance documentation ───────────────────────────────────


class TestOnnxPerformanceDoc:
    """Document ONNX backend performance characteristics."""

    def test_heuristic_fallback_fast(self):
        import time

        scorer = NLIScorer(use_model=True, backend="onnx")
        # Warmup
        for _ in range(10):
            scorer.score("warmup", "warmup")

        t0 = time.perf_counter()
        for _ in range(100):
            scorer.score("test", "test")
        per_call_ms = (time.perf_counter() - t0) / 100 * 1000
        assert per_call_ms < 1.0, (
            f"ONNX heuristic fallback took {per_call_ms:.3f}ms/call (expected <1ms)"
        )


class TestOnnxPathValidation:
    """ONNX path is resolved before use; a non-directory falls back safely (V8)."""

    def test_nonexistent_onnx_path_falls_back(self):
        scorer = NLIScorer(backend="onnx", onnx_path="/nonexistent/onnx/dir")
        # Graceful fallback (model unavailable) — no raw path handed downstream.
        assert scorer.model_available is False
        assert 0.0 <= scorer.score("premise", "hypothesis") <= 1.0

    def test_onnx_path_pointing_at_file_falls_back(self, tmp_path):
        not_a_dir = tmp_path / "model.onnx"
        not_a_dir.write_text("", encoding="utf-8")
        scorer = NLIScorer(backend="onnx", onnx_path=str(not_a_dir))
        assert scorer.model_available is False

    def test_relative_onnx_path_resolved_not_passed_through(self):
        # A relative non-existent path resolves and falls back rather than being
        # handed to the runtime verbatim.
        scorer = NLIScorer(backend="onnx", onnx_path="./does-not-exist-rel")
        assert scorer.model_available is False
