# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ONNX Backend Tests (STRONG)
"""Multi-angle tests for ONNX NLI backend and export.

Covers: heuristic fallback when ONNX unavailable, bad path handling,
batch scoring, export importability, missing optimum guard, onnxruntime
provider check, parametrised inputs, score range invariants, pipeline
integration via CoherenceScorer, and performance documentation.
"""

from __future__ import annotations

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
    """export_onnx must be importable and guard missing deps."""

    def test_importable(self):
        from director_ai.core import export_onnx as fn

        assert callable(fn)

    def test_missing_optimum_raises(self):
        try:
            export_onnx(output_dir="/tmp/test_onnx_export_guard")
            pytest.skip("optimum is installed — skip import test")
        except (ImportError, OSError):
            pass


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
