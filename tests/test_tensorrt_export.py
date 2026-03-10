# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — TensorRT Export Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


# ── export_tensorrt unit tests ──────────────────────────────────────


class TestExportTensorrt:
    def test_missing_onnx_dir_raises(self):
        from director_ai.core.nli import export_tensorrt

        with pytest.raises(FileNotFoundError, match="ONNX model not found"):
            export_tensorrt(onnx_dir="/nonexistent/path")

    def test_no_trt_provider_raises(self, tmp_path):
        onnx_dir = tmp_path / "onnx"
        onnx_dir.mkdir()
        (onnx_dir / "model.onnx").write_bytes(b"dummy")

        from director_ai.core.nli import export_tensorrt

        with patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]):
            with pytest.raises(RuntimeError, match="TensorrtExecutionProvider not available"):
                export_tensorrt(onnx_dir=str(onnx_dir))

    def test_creates_cache_dir(self, tmp_path):
        onnx_dir = tmp_path / "onnx"
        onnx_dir.mkdir()
        (onnx_dir / "model.onnx").write_bytes(b"dummy")
        cache_dir = tmp_path / "cache"

        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["TensorrtExecutionProvider"]
        mock_session.get_inputs.return_value = [
            MagicMock(name="input_ids"),
            MagicMock(name="attention_mask"),
        ]
        mock_session.run.return_value = [__import__("numpy").array([[0.1, 0.9]])]

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": __import__("numpy").ones((4, 10), dtype=__import__("numpy").int64),
            "attention_mask": __import__("numpy").ones((4, 10), dtype=__import__("numpy").int64),
        }

        with (
            patch("onnxruntime.get_available_providers", return_value=["TensorrtExecutionProvider", "CUDAExecutionProvider"]),
            patch("onnxruntime.InferenceSession", return_value=mock_session),
            patch("onnxruntime.SessionOptions"),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        ):
            from director_ai.core.nli import export_tensorrt

            result = export_tensorrt(
                onnx_dir=str(onnx_dir),
                output_dir=str(cache_dir),
                warmup_pairs=2,
            )
            assert result == str(cache_dir)
            assert cache_dir.exists()
            mock_session.run.assert_called_once()

    def test_fp16_default_true(self, tmp_path):
        onnx_dir = tmp_path / "onnx"
        onnx_dir.mkdir()
        (onnx_dir / "model.onnx").write_bytes(b"dummy")

        captured_opts = {}

        def capture_session(model_file, opts, providers):
            for p in providers:
                if isinstance(p, tuple) and p[0] == "TensorrtExecutionProvider":
                    captured_opts.update(p[1])
            mock = MagicMock()
            mock.get_providers.return_value = ["TensorrtExecutionProvider"]
            inp_mock = MagicMock()
            inp_mock.name = "input_ids"
            mock.get_inputs.return_value = [inp_mock]
            mock.run.return_value = [__import__("numpy").array([[0.1, 0.9]])]
            return mock

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": __import__("numpy").ones((4, 10), dtype=__import__("numpy").int64),
        }

        with (
            patch("onnxruntime.get_available_providers", return_value=["TensorrtExecutionProvider", "CUDAExecutionProvider"]),
            patch("onnxruntime.InferenceSession", side_effect=capture_session),
            patch("onnxruntime.SessionOptions", return_value=MagicMock()),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        ):
            from director_ai.core.nli import export_tensorrt

            export_tensorrt(onnx_dir=str(onnx_dir), warmup_pairs=1)

        assert captured_opts.get("trt_fp16_enable") is True


# ── _load_onnx_session TRT auto-detection ───────────────────────────


class TestTrtAutoDetection:
    def test_trt_cache_dir_triggers_trt_ep(self, tmp_path):
        onnx_dir = tmp_path / "onnx"
        onnx_dir.mkdir()
        (onnx_dir / "model.onnx").write_bytes(b"dummy")
        trt_cache = onnx_dir / "trt_cache"
        trt_cache.mkdir()

        captured_providers = []

        def capture_session(model_file, opts, providers):
            captured_providers.extend(providers)
            mock = MagicMock()
            mock.get_providers.return_value = ["TensorrtExecutionProvider"]
            return mock

        mock_tokenizer = MagicMock()

        with (
            patch("onnxruntime.get_available_providers", return_value=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]),
            patch("onnxruntime.InferenceSession", side_effect=capture_session),
            patch("onnxruntime.SessionOptions", return_value=MagicMock()),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        ):
            from director_ai.core.nli import _load_onnx_session

            _load_onnx_session.cache_clear()
            _load_onnx_session(str(onnx_dir), device="cuda")

        trt_in_providers = any(
            (isinstance(p, tuple) and p[0] == "TensorrtExecutionProvider")
            or p == "TensorrtExecutionProvider"
            for p in captured_providers
        )
        assert trt_in_providers

    def test_no_trt_cache_skips_trt_ep(self, tmp_path):
        onnx_dir = tmp_path / "onnx"
        onnx_dir.mkdir()
        (onnx_dir / "model.onnx").write_bytes(b"dummy")
        # no trt_cache dir

        captured_providers = []

        def capture_session(model_file, opts, providers):
            captured_providers.extend(providers)
            mock = MagicMock()
            mock.get_providers.return_value = ["CUDAExecutionProvider"]
            return mock

        mock_tokenizer = MagicMock()

        os.environ.pop("DIRECTOR_ENABLE_TRT", None)
        with (
            patch("onnxruntime.get_available_providers", return_value=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]),
            patch("onnxruntime.InferenceSession", side_effect=capture_session),
            patch("onnxruntime.SessionOptions", return_value=MagicMock()),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        ):
            from director_ai.core.nli import _load_onnx_session

            _load_onnx_session.cache_clear()
            _load_onnx_session(str(onnx_dir), device="cuda")

        trt_in_providers = any(
            (isinstance(p, tuple) and p[0] == "TensorrtExecutionProvider")
            or p == "TensorrtExecutionProvider"
            for p in captured_providers
        )
        assert not trt_in_providers


# ── CLI export subcommand ───────────────────────────────────────────


class TestCLIExport:
    def test_export_onnx_format(self):
        from director_ai.cli import main

        with patch("director_ai.core.nli.export_onnx") as mock_export:
            main(["export", "--format", "onnx", "--output", "/tmp/test_onnx"])

        mock_export.assert_called_once_with(
            model_name="yaxili96/FactCG-DeBERTa-v3-Large",
            output_dir="/tmp/test_onnx",
            quantize=None,
        )

    def test_export_onnx_with_quantize(self):
        from director_ai.cli import main

        with patch("director_ai.core.nli.export_onnx") as mock_export:
            main(["export", "--format", "onnx", "--quantize", "int8"])

        mock_export.assert_called_once_with(
            model_name="yaxili96/FactCG-DeBERTa-v3-Large",
            output_dir="factcg_onnx",
            quantize="int8",
        )

    def test_export_tensorrt_format(self):
        from director_ai.cli import main

        with patch("director_ai.core.nli.export_tensorrt") as mock_export:
            mock_export.return_value = "/tmp/trt_cache"
            main(["export", "--format", "tensorrt", "--onnx-dir", "/tmp/onnx"])

        mock_export.assert_called_once_with(
            onnx_dir="/tmp/onnx",
            output_dir="factcg_onnx",
            fp16=True,
        )

    def test_export_tensorrt_no_fp16(self):
        from director_ai.cli import main

        with patch("director_ai.core.nli.export_tensorrt") as mock_export:
            mock_export.return_value = "/tmp/cache"
            main(["export", "--format", "tensorrt", "--no-fp16"])

        mock_export.assert_called_once()
        assert mock_export.call_args.kwargs["fp16"] is False

    def test_export_unknown_format_exits(self):
        from director_ai.cli import main

        with pytest.raises(SystemExit, match="1"):
            main(["export", "--format", "badformat"])


# ── __all__ exports ─────────────────────────────────────────────────


class TestExportsUpdated:
    def test_export_tensorrt_in_all(self):
        from director_ai.core import nli

        assert "export_tensorrt" in nli.__all__
