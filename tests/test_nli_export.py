# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — NLI Export Tests
"""Module-specific tests for NLI ONNX export and loading helpers."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import director_ai.core.scoring._nli_export as nli_export


def test_resolve_onnx_model_file_falls_back_to_first_onnx_file(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    fallback = bundle / "custom-model.onnx"
    fallback.write_bytes(b"onnx")

    onnx_dir, model_file = nli_export._resolve_onnx_model_file(str(bundle))

    assert onnx_dir == bundle.resolve()
    assert model_file == fallback


def test_resolve_onnx_model_file_rejects_unapproved_root(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    (outside / "model.onnx").write_bytes(b"onnx")
    monkeypatch.setenv("DIRECTOR_ONNX_ALLOWED_DIRS", str(allowed))

    with pytest.raises(PermissionError, match="outside DIRECTOR_ONNX_ALLOWED_DIRS"):
        nli_export._resolve_onnx_model_file(str(outside))


def test_resolve_onnx_model_file_rejects_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="Not a directory"):
        nli_export._resolve_onnx_model_file(str(tmp_path / "missing"))


def test_resolve_onnx_model_file_rejects_symlink_escape(tmp_path):
    bundle = tmp_path / "bundle"
    outside = tmp_path / "outside"
    bundle.mkdir()
    outside.mkdir()
    (outside / "model.onnx").write_bytes(b"onnx")
    (bundle / "model.onnx").symlink_to(outside / "model.onnx")

    with pytest.raises(PermissionError, match="escapes model directory"):
        nli_export._resolve_onnx_model_file(str(bundle))


def test_resolve_onnx_model_file_rechecks_allowed_model_file(monkeypatch, tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.onnx").write_bytes(b"onnx")
    calls = []

    def fake_relative_to(path, root):
        calls.append((path, root))
        return len(calls) < 3

    monkeypatch.setenv("DIRECTOR_ONNX_ALLOWED_DIRS", str(bundle))
    monkeypatch.setattr(nli_export, "_is_relative_to", fake_relative_to)

    with pytest.raises(PermissionError, match="outside DIRECTOR_ONNX_ALLOWED_DIRS"):
        nli_export._resolve_onnx_model_file(str(bundle))


def test_configured_onnx_allowed_dirs_ignores_empty_segments(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "DIRECTOR_ONNX_ALLOWED_DIRS",
        f"{tmp_path}{nli_export.os.pathsep}{nli_export.os.pathsep} ",
    )

    assert nli_export._configured_onnx_allowed_dirs() == (tmp_path.resolve(),)


def test_load_onnx_session_prefers_cpu_quantized_model(tmp_path, monkeypatch):
    nli_export._load_onnx_session.cache_clear()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.onnx").write_bytes(b"fp32")
    quantized = bundle / "model_quantized.onnx"
    quantized.write_bytes(b"int8")
    captured = {}

    class FakeSession:
        def get_providers(self):
            return ["CPUExecutionProvider"]

    def inference_session(model_file, opts, providers):
        captured["model_file"] = model_file
        captured["providers"] = providers
        return FakeSession()

    tokenizer = object()
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=Mock(return_value=tokenizer),
    )
    fake_ort = ModuleType("onnxruntime")
    fake_ort.GraphOptimizationLevel = SimpleNamespace(ORT_ENABLE_ALL="all")
    fake_ort.get_available_providers = Mock(return_value=["CPUExecutionProvider"])
    fake_ort.SessionOptions = Mock(
        return_value=SimpleNamespace(
            graph_optimization_level=None,
            log_severity_level=None,
        ),
    )
    fake_ort.InferenceSession = Mock(side_effect=inference_session)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    loaded_tokenizer, session = nli_export._load_onnx_session(str(bundle), device="cpu")

    assert loaded_tokenizer is tokenizer
    assert isinstance(session, FakeSession)
    assert captured["model_file"] == str(quantized)
    assert captured["providers"] == ["CPUExecutionProvider"]


def test_load_onnx_session_returns_none_when_runtime_missing(monkeypatch, tmp_path):
    nli_export._load_onnx_session.cache_clear()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.onnx").write_bytes(b"onnx")
    monkeypatch.setitem(sys.modules, "onnxruntime", None)

    assert nli_export._load_onnx_session(str(bundle)) == (None, None)


def test_export_onnx_rejects_tokenizer_without_tensor_inputs(
    tmp_path,
    monkeypatch,
):
    fake_torch = _fake_torch()
    fake_transformers = _fake_transformers(tokenizer_result={"metadata": object()})
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    with pytest.raises(RuntimeError, match="ONNX-exportable tensor inputs"):
        nli_export.export_onnx(
            model_name="test/model",
            output_dir=str(tmp_path),
            revision="abc123",
        )


def test_export_onnx_int8_quantization_writes_quantized_model(tmp_path, monkeypatch):
    fake_torch = _fake_torch()
    fake_transformers = _fake_transformers()
    quantize_dynamic = Mock()
    quantization_module = ModuleType("onnxruntime.quantization")
    quantization_module.QuantType = SimpleNamespace(QInt8="qint8")
    quantization_module.quantize_dynamic = quantize_dynamic
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "onnxruntime.quantization", quantization_module)

    result = nli_export.export_onnx(
        model_name="test/model",
        output_dir=str(tmp_path),
        quantize="int8",
        revision="abc123",
    )

    assert result == str(tmp_path)
    quantize_dynamic.assert_called_once_with(
        str(tmp_path / "model.onnx"),
        str(tmp_path / "model_quantized.onnx"),
        weight_type="qint8",
    )


def test_export_onnx_fp16_quantization_converts_and_saves(tmp_path, monkeypatch):
    fake_torch = _fake_torch()
    fake_transformers = _fake_transformers()
    fake_onnx = ModuleType("onnx")
    fake_model = object()
    fake_onnx.load = Mock(return_value=fake_model)
    fake_onnx.save = Mock()
    fake_float16 = ModuleType("onnxruntime.transformers.float16")
    fake_fp16_model = object()
    fake_float16.convert_float_to_float16 = Mock(return_value=fake_fp16_model)
    fake_transformers_pkg = ModuleType("onnxruntime.transformers")
    fake_transformers_pkg.float16 = fake_float16
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)
    monkeypatch.setitem(sys.modules, "onnxruntime.transformers", fake_transformers_pkg)
    monkeypatch.setitem(sys.modules, "onnxruntime.transformers.float16", fake_float16)

    result = nli_export.export_onnx(
        model_name="test/model",
        output_dir=str(tmp_path),
        quantize="fp16",
        revision="abc123",
    )

    assert result == str(tmp_path)
    fake_onnx.load.assert_called_once_with(str(tmp_path / "model.onnx"))
    fake_float16.convert_float_to_float16.assert_called_once_with(fake_model)
    fake_onnx.save.assert_called_once_with(
        fake_fp16_model,
        str(tmp_path / "model_fp16.onnx"),
    )


def test_export_tensorrt_requires_available_provider(tmp_path, monkeypatch):
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    (onnx_dir / "model.onnx").write_bytes(b"onnx")
    fake_ort = ModuleType("onnxruntime")
    fake_ort.get_available_providers = Mock(return_value=["CPUExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    with pytest.raises(RuntimeError, match="TensorrtExecutionProvider not available"):
        nli_export.export_tensorrt(str(onnx_dir))


def test_export_tensorrt_builds_cache_with_int64_tokenizer_feed(
    tmp_path,
    monkeypatch,
):
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    (onnx_dir / "model.onnx").write_bytes(b"onnx")
    captured: dict[str, object] = {}

    class FakeInput:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeSession:
        def __init__(self, model_file, opts, providers):
            captured["model_file"] = model_file
            captured["providers"] = providers

        def get_providers(self):
            return ["TensorrtExecutionProvider", "CUDAExecutionProvider"]

        def get_inputs(self):
            return [FakeInput("input_ids"), FakeInput("attention_mask")]

        def run(self, outputs, feed):
            captured["outputs"] = outputs
            captured["feed"] = feed
            return [np.array([[0.1, 0.9]])]

    fake_ort = ModuleType("onnxruntime")
    fake_ort.GraphOptimizationLevel = SimpleNamespace(ORT_ENABLE_ALL="all")
    fake_ort.get_available_providers = Mock(
        return_value=[
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    fake_ort.SessionOptions = Mock(
        return_value=SimpleNamespace(
            graph_optimization_level=None,
            log_severity_level=None,
        ),
    )
    fake_ort.InferenceSession = Mock(side_effect=FakeSession)
    tokenizer = Mock(
        return_value={
            "input_ids": np.array([[1, 2]], dtype=np.int32),
            "attention_mask": np.array([[1, 1]], dtype=np.int64),
            "token_type_ids": np.array([[0, 0]], dtype=np.int32),
        },
    )
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=Mock(return_value=tokenizer),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    result = nli_export.export_tensorrt(
        str(onnx_dir),
        output_dir=str(tmp_path / "trt"),
        fp16=False,
        max_batch=8,
        max_seq_len=128,
        warmup_pairs=2,
    )

    assert result == str(tmp_path / "trt")
    assert captured["model_file"] == str(onnx_dir / "model.onnx")
    provider, options = captured["providers"][0]
    assert provider == "TensorrtExecutionProvider"
    assert options["trt_engine_cache_path"] == str(tmp_path / "trt")
    assert options["trt_fp16_enable"] is False
    assert options["trt_profile_opt_shapes"] == (
        "input_ids=4x128,attention_mask=4x128"
    )
    assert captured["outputs"] is None
    feed = captured["feed"]
    assert set(feed) == {"input_ids", "attention_mask"}
    assert feed["input_ids"].dtype == np.int64
    assert feed["attention_mask"].dtype == np.int64
    assert fake_transformers.AutoTokenizer.from_pretrained.call_args.kwargs == {
        "revision": "local-artifact",
        "local_files_only": True,
    }


def test_dynamic_batcher_empty_flush_is_noop():
    score_fn = Mock(return_value=[0.1])
    batcher = nli_export.OnnxDynamicBatcher(score_fn)

    assert batcher.flush() == []
    score_fn.assert_not_called()


def _fake_torch():
    fake_torch = ModuleType("torch")
    fake_torch.no_grad = Mock(return_value=nullcontext())

    class Module:
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    def export(model, args, path, **kwargs):
        assert isinstance(path, str)
        assert kwargs["input_names"] == ["input_ids", "attention_mask"]
        assert kwargs["output_names"] == ["logits"]
        assert kwargs["dynamic_axes"]["logits"] == {0: "batch"}
        assert model(*args) is _FAKE_LOGITS

    fake_torch.nn = SimpleNamespace(Module=Module)
    fake_torch.onnx = SimpleNamespace(export=export)
    return fake_torch


def _fake_transformers(tokenizer_result=None):
    fake_transformers = ModuleType("transformers")
    tokenizer_result = tokenizer_result or {
        "input_ids": object(),
        "attention_mask": object(),
    }
    tokenizer = Mock(return_value=tokenizer_result)
    tokenizer.save_pretrained = Mock()
    model = Mock()
    model.return_value = SimpleNamespace(logits=_FAKE_LOGITS)
    model.config.save_pretrained = Mock()
    model.eval = Mock()
    fake_transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=Mock(return_value=tokenizer),
    )
    fake_transformers.AutoModelForSequenceClassification = SimpleNamespace(
        from_pretrained=Mock(return_value=model),
    )
    return fake_transformers


_FAKE_LOGITS = object()
