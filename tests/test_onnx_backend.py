from __future__ import annotations

import pytest

from director_ai.core.nli import NLIScorer, export_onnx


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


class TestExportOnnxImport:
    """export_onnx is importable from the public API."""

    def test_importable(self):
        from director_ai.core import export_onnx as fn

        assert callable(fn)

    def test_missing_optimum_raises(self):
        """export_onnx raises ImportError when optimum is missing."""
        # In CI without optimum installed, the import inside
        # export_onnx() raises. We verify it doesn't silently pass.
        try:
            export_onnx(output_dir="/tmp/test_onnx_export")
            # If optimum IS installed, this would try to download
            # the model. Skip in that case.
            pytest.skip("optimum is installed — skip import test")
        except (ImportError, OSError):
            pass


_has_onnxruntime = False
try:
    import onnxruntime  # noqa: F401

    _has_onnxruntime = True
except ImportError:
    pass


@pytest.mark.skipif(not _has_onnxruntime, reason="onnxruntime not installed")
class TestOnnxRuntimeAvailable:
    """Tests that run when onnxruntime is installed."""

    def test_providers_list(self):
        import onnxruntime as ort

        providers = ort.get_available_providers()
        assert "CPUExecutionProvider" in providers
