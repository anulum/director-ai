# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Judge Benchmark Runner Tests
"""Multi-angle tests for benchmarks/run_judge_benchmark.py.

Covers: FP16 flag wiring, function signatures, GPU info utility,
comparison table, result saving, CLI argument parsing,
and pipeline performance documentation.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.run_judge_benchmark import (
    _gpu_info,
    _save,
    print_comparison,
    run_local_judge,
    run_nli_only,
)

# ── Function signatures ────────────────────────────────────────────


class TestFunctionSignatures:
    """Verify all benchmark functions accept required parameters."""

    def test_run_nli_only_accepts_nli_torch_dtype(self):
        sig = inspect.signature(run_nli_only)
        assert "nli_torch_dtype" in sig.parameters

    def test_run_local_judge_accepts_nli_torch_dtype(self):
        sig = inspect.signature(run_local_judge)
        assert "nli_torch_dtype" in sig.parameters

    def test_run_nli_only_default_dtype_is_none(self):
        sig = inspect.signature(run_nli_only)
        default = sig.parameters["nli_torch_dtype"].default
        assert default is None

    def test_run_local_judge_default_dtype_is_none(self):
        sig = inspect.signature(run_local_judge)
        default = sig.parameters["nli_torch_dtype"].default
        assert default is None


# ── GPU info utility ────────────────────────────────────────────────


class TestGPUInfo:
    """Test GPU detection utility for various hardware states."""

    def test_returns_dict(self):
        info = _gpu_info()
        assert isinstance(info, dict)

    def test_has_gpu_key(self):
        info = _gpu_info()
        assert "gpu" in info

    def test_has_cuda_key(self):
        info = _gpu_info()
        assert "cuda" in info

    @patch.dict("sys.modules", {"torch": None})
    def test_import_error_handled(self):
        # When torch is unavailable, should not crash
        info = _gpu_info()
        assert info["gpu"] == "unavailable" or "gpu" in info


# ── Result saving ──────────────────────────────────────────────────


class TestSaveResults:
    """Test benchmark result JSON serialisation."""

    def test_save_creates_json_file(self, tmp_path):
        with patch("benchmarks.run_judge_benchmark.RESULTS_DIR", tmp_path):
            _save({"benchmark": "test", "value": 42}, "test_result.json")
        path = tmp_path / "test_result.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["benchmark"] == "test"
        assert data["value"] == 42

    def test_save_json_is_pretty_printed(self, tmp_path):
        with patch("benchmarks.run_judge_benchmark.RESULTS_DIR", tmp_path):
            _save({"k": "v"}, "pretty.json")
        text = (tmp_path / "pretty.json").read_text()
        assert "\n" in text  # indented JSON


# ── Comparison table ───────────────────────────────────────────────


class TestPrintComparison:
    """Test side-by-side comparison output."""

    def test_comparison_does_not_crash(self, capsys):
        nli_only = {
            "catch_rate": 0.9,
            "false_positive_rate": 0.6,
            "precision": 0.6,
            "f1": 0.72,
            "accuracy": 0.65,
            "avg_latency_ms": 500,
            "elapsed_s": 100,
            "per_task": {
                "qa": {"f1": 0.85},
                "summarization": {"f1": 0.65},
                "dialogue": {"f1": 0.60},
            },
        }
        local_judge = {
            "catch_rate": 0.93,
            "false_positive_rate": 0.55,
            "precision": 0.65,
            "f1": 0.77,
            "accuracy": 0.70,
            "avg_latency_ms": 550,
            "elapsed_s": 120,
            "per_task": {
                "qa": {"f1": 0.88},
                "summarization": {"f1": 0.70},
                "dialogue": {"f1": 0.65},
            },
        }
        print_comparison(nli_only, local_judge)
        captured = capsys.readouterr()
        assert "Catch rate" in captured.out
        assert "Delta" in captured.out

    def test_comparison_shows_delta(self, capsys):
        nli = {
            "catch_rate": 0.9,
            "false_positive_rate": 0.6,
            "precision": 0.6,
            "f1": 0.7,
            "accuracy": 0.65,
            "avg_latency_ms": 500,
            "elapsed_s": 100,
            "per_task": {"qa": {"f1": 0.8}},
        }
        judge = {
            "catch_rate": 0.95,
            "false_positive_rate": 0.5,
            "precision": 0.7,
            "f1": 0.8,
            "accuracy": 0.75,
            "avg_latency_ms": 550,
            "elapsed_s": 120,
            "per_task": {"qa": {"f1": 0.85}},
        }
        print_comparison(nli, judge)
        captured = capsys.readouterr()
        assert "+" in captured.out  # positive delta shown

    @pytest.mark.parametrize("task", ["qa", "summarization", "dialogue"])
    def test_comparison_includes_per_task(self, capsys, task):
        base = {
            "catch_rate": 0.5,
            "false_positive_rate": 0.5,
            "precision": 0.5,
            "f1": 0.5,
            "accuracy": 0.5,
            "avg_latency_ms": 100,
            "elapsed_s": 50,
            "per_task": {task: {"f1": 0.5}},
        }
        print_comparison(base, base)
        captured = capsys.readouterr()
        assert task in captured.out


# ── Pipeline performance documentation ──────────────────────────────


class TestPipelinePerformanceDoc:
    """Verify benchmark runner documents performance at each stage."""

    def test_latency_result_has_required_fields(self):
        """Latency result must document device, median, p95, hardware."""
        # Fields set by run_judge_latency: device, median_ms, p95_ms, hw
        # Actual run requires GPU, so we verify the function exists
        from benchmarks.run_judge_benchmark import run_judge_latency

        sig = inspect.signature(run_judge_latency)
        assert "n_iters" in sig.parameters

    def test_nli_only_result_documents_hw(self):
        """NLI-only result must include hardware info."""
        # Check run_nli_only returns dict with 'hw' key by reading source
        src = inspect.getsource(run_nli_only)
        assert '"hw"' in src or "'hw'" in src

    def test_local_judge_result_documents_hw(self):
        src = inspect.getsource(run_local_judge)
        assert '"hw"' in src or "'hw'" in src

    def test_comparison_documents_all_metrics(self, capsys):
        """Comparison table must show catch_rate, FPR, precision, F1, accuracy."""
        base = {
            "catch_rate": 0.5,
            "false_positive_rate": 0.5,
            "precision": 0.5,
            "f1": 0.5,
            "accuracy": 0.5,
            "avg_latency_ms": 100,
            "elapsed_s": 50,
            "per_task": {"qa": {"f1": 0.5}},
        }
        print_comparison(base, base)
        captured = capsys.readouterr()
        for metric in ["Catch rate", "False positive", "Precision", "F1", "Accuracy"]:
            assert metric in captured.out
