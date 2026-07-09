# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — NLI Export Tests
"""Typed guard tests for NLI ONNX export and loading helpers."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Protocol, cast
from unittest.mock import Mock

import numpy as np
import pytest

import director_ai.core.scoring._nli_export as nli_export

_FAKE_LOGITS = object()


class _TorchModule(Protocol):
    """Typed subset of the torch module used by ONNX export tests."""

    long: object
    no_grad: Mock
    ones: Mock
    nn: SimpleNamespace
    onnx: SimpleNamespace


class _FakeTorchBaseModule:
    """Minimal ``torch.nn.Module`` protocol with callable forwarding."""

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Forward calls to the subclass ``forward`` method."""
        return self.forward(*args, **kwargs)

    def forward(self, *args: object, **kwargs: object) -> object:
        """Require subclasses to provide the fake model forward pass."""
        raise NotImplementedError


class _TransformersBundle:
    """Installed fake transformers module and its protocol objects."""

    def __init__(self, tokenizer_result: dict[str, object] | None = None) -> None:
        result = tokenizer_result or {
            "input_ids": object(),
            "attention_mask": object(),
        }
        self.tokenizer = Mock(return_value=result)
        self.tokenizer.save_pretrained = Mock()
        self.model = Mock()
        self.model.return_value = SimpleNamespace(logits=_FAKE_LOGITS)
        self.model.config.save_pretrained = Mock()
        self.model.eval = Mock()
        self.tokenizer_from_pretrained = Mock(return_value=self.tokenizer)
        self.model_from_pretrained = Mock(return_value=self.model)
        self.module = _module(
            "transformers",
            AutoTokenizer=SimpleNamespace(
                from_pretrained=self.tokenizer_from_pretrained,
            ),
            AutoModelForSequenceClassification=SimpleNamespace(
                from_pretrained=self.model_from_pretrained,
            ),
        )


def _module(name: str, **attrs: object) -> ModuleType:
    """Return a module object populated with dynamic protocol attributes."""
    module = ModuleType(name)
    for attr_name, value in attrs.items():
        module.__dict__[attr_name] = value
    return module


def _fake_torch(
    export_function: Callable[..., object] | None = None,
) -> _TorchModule:
    """Build a fake torch module for exercising the real export function."""

    def ones(shape: tuple[int, int], dtype: object | None = None) -> dict[str, object]:
        """Return a typed tensor placeholder for export inputs."""
        return {"shape": shape, "dtype": dtype}

    def export(
        model: Callable[..., object],
        args: tuple[object, ...],
        path: str,
        **kwargs: object,
    ) -> None:
        """Validate the export contract passed to ``torch.onnx.export``."""
        assert isinstance(path, str)
        assert kwargs["input_names"] == ["input_ids", "attention_mask"]
        assert kwargs["output_names"] == ["logits"]
        dynamic_axes = cast(dict[str, dict[int, str]], kwargs["dynamic_axes"])
        assert dynamic_axes["logits"] == {0: "batch"}
        assert model(*args) is _FAKE_LOGITS

    module = _module(
        "torch",
        long=object(),
        no_grad=Mock(return_value=nullcontext()),
        ones=Mock(side_effect=ones),
        nn=SimpleNamespace(Module=_FakeTorchBaseModule),
        onnx=SimpleNamespace(export=export_function or export),
    )
    return cast(_TorchModule, module)


def _fake_transformers(
    tokenizer_result: dict[str, object] | None = None,
) -> _TransformersBundle:
    """Build a fake transformers module bundle for export tests."""
    return _TransformersBundle(tokenizer_result)


def test_resolve_onnx_model_file_falls_back_to_first_onnx_file(
    tmp_path: Path,
) -> None:
    """A bundle without canonical names should use the first ONNX file."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    fallback = bundle / "custom-model.onnx"
    fallback.write_bytes(b"onnx")

    onnx_dir, model_file = nli_export._resolve_onnx_model_file(str(bundle))

    assert onnx_dir == bundle.resolve()
    assert model_file == fallback


def test_resolve_onnx_model_file_rejects_unapproved_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured ONNX roots should reject directories outside the allowlist."""
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    (outside / "model.onnx").write_bytes(b"onnx")
    monkeypatch.setenv("DIRECTOR_ONNX_ALLOWED_DIRS", str(allowed))

    with pytest.raises(PermissionError, match="outside DIRECTOR_ONNX_ALLOWED_DIRS"):
        nli_export._resolve_onnx_model_file(str(outside))


def test_resolve_onnx_model_file_rejects_missing_directory(tmp_path: Path) -> None:
    """Missing ONNX bundle directories should fail with a clear error."""
    with pytest.raises(FileNotFoundError, match="Not a directory"):
        nli_export._resolve_onnx_model_file(str(tmp_path / "missing"))


def test_resolve_onnx_model_file_rejects_symlink_escape(tmp_path: Path) -> None:
    """Symlinked model files must not escape their containing bundle."""
    bundle = tmp_path / "bundle"
    outside = tmp_path / "outside"
    bundle.mkdir()
    outside.mkdir()
    (outside / "model.onnx").write_bytes(b"onnx")
    (bundle / "model.onnx").symlink_to(outside / "model.onnx")

    with pytest.raises(PermissionError, match="escapes model directory"):
        nli_export._resolve_onnx_model_file(str(bundle))


def test_resolve_onnx_model_file_rechecks_allowed_model_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The resolved model file should be checked against the allowlist too."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.onnx").write_bytes(b"onnx")
    calls: list[tuple[Path, Path]] = []

    def fake_relative_to(path: Path, root: Path) -> bool:
        """Allow the directory check and reject the resolved model check."""
        calls.append((path, root))
        return len(calls) < 3

    monkeypatch.setenv("DIRECTOR_ONNX_ALLOWED_DIRS", str(bundle))
    monkeypatch.setattr(nli_export, "_is_relative_to", fake_relative_to)

    with pytest.raises(PermissionError, match="outside DIRECTOR_ONNX_ALLOWED_DIRS"):
        nli_export._resolve_onnx_model_file(str(bundle))


def test_configured_onnx_allowed_dirs_ignores_empty_segments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty path-list segments should not create bogus allowed roots."""
    monkeypatch.setenv(
        "DIRECTOR_ONNX_ALLOWED_DIRS",
        f"{tmp_path}{os.pathsep}{os.pathsep} ",
    )

    assert nli_export._configured_onnx_allowed_dirs() == (tmp_path.resolve(),)


def test_load_onnx_session_prefers_cpu_quantized_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU ONNX loading should prefer the quantized artefact when present."""
    nli_export._load_onnx_session.cache_clear()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.onnx").write_bytes(b"fp32")
    quantized = bundle / "model_quantized.onnx"
    quantized.write_bytes(b"int8")
    captured: dict[str, object] = {}

    class FakeSession:
        """Fake ONNX Runtime session exposing provider metadata."""

        def get_providers(self) -> list[str]:
            """Return the active provider order."""
            return ["CPUExecutionProvider"]

    def inference_session(
        model_file: str,
        opts: object,
        providers: list[str | tuple[str, dict[str, object]]],
    ) -> FakeSession:
        """Capture the ONNX session constructor arguments."""
        captured["model_file"] = model_file
        captured["providers"] = providers
        captured["opts"] = opts
        return FakeSession()

    tokenizer = object()
    auto_tokenizer = SimpleNamespace(from_pretrained=Mock(return_value=tokenizer))
    fake_transformers = _module("transformers", AutoTokenizer=auto_tokenizer)
    fake_ort = _module(
        "onnxruntime",
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        get_available_providers=Mock(return_value=["CPUExecutionProvider"]),
        SessionOptions=Mock(
            return_value=SimpleNamespace(
                graph_optimization_level=None,
                log_severity_level=None,
            ),
        ),
        InferenceSession=Mock(side_effect=inference_session),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    loaded_tokenizer, session = nli_export._load_onnx_session(str(bundle), device="cpu")

    assert loaded_tokenizer is tokenizer
    assert isinstance(session, FakeSession)
    assert captured["model_file"] == str(quantized)
    assert captured["providers"] == ["CPUExecutionProvider"]


def test_load_onnx_session_returns_none_when_runtime_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Missing ONNX Runtime should produce the documented fallback tuple."""
    nli_export._load_onnx_session.cache_clear()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.onnx").write_bytes(b"onnx")
    monkeypatch.setitem(sys.modules, "onnxruntime", cast(ModuleType, None))

    assert nli_export._load_onnx_session(str(bundle)) == (None, None)


def test_export_tensorrt_raises_when_model_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """TensorRT export should fail before provider work when model.onnx is absent."""
    fake_ort = _module(
        "onnxruntime",
        get_available_providers=Mock(return_value=["CPUExecutionProvider"]),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    with pytest.raises(FileNotFoundError, match="ONNX model not found"):
        nli_export.export_tensorrt(onnx_dir=str(bundle))


def test_export_onnx_rejects_tokenizer_without_tensor_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Export should reject tokenizers that omit tensor input names."""
    fake_torch = _fake_torch()
    fake_transformers = _fake_transformers(tokenizer_result={"metadata": object()})
    monkeypatch.setitem(sys.modules, "torch", cast(ModuleType, fake_torch))
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers.module)

    with pytest.raises(RuntimeError, match="ONNX-exportable tensor inputs"):
        nli_export.export_onnx(
            model_name="test/model",
            output_dir=str(tmp_path),
            revision="abc123",
        )


def test_export_onnx_int8_quantization_writes_quantized_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INT8 export should call ONNX Runtime dynamic quantization."""
    fake_torch = _fake_torch()
    fake_transformers = _fake_transformers()
    quantize_dynamic = Mock()
    quantization_module = _module(
        "onnxruntime.quantization",
        QuantType=SimpleNamespace(QInt8="qint8"),
        quantize_dynamic=quantize_dynamic,
    )
    monkeypatch.setitem(sys.modules, "torch", cast(ModuleType, fake_torch))
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers.module)
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


def test_export_onnx_fp16_quantization_converts_and_saves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FP16 export should convert and save the ONNX graph."""
    fake_torch = _fake_torch()
    fake_transformers = _fake_transformers()
    fake_model = object()
    onnx_load = Mock(return_value=fake_model)
    onnx_save = Mock()
    fake_onnx = _module("onnx", load=onnx_load, save=onnx_save)
    fake_fp16_model = object()
    convert_float_to_float16 = Mock(return_value=fake_fp16_model)
    fake_float16 = _module(
        "onnxruntime.transformers.float16",
        convert_float_to_float16=convert_float_to_float16,
    )
    fake_transformers_pkg = _module(
        "onnxruntime.transformers",
        float16=fake_float16,
    )
    monkeypatch.setitem(sys.modules, "torch", cast(ModuleType, fake_torch))
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers.module)
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
    onnx_load.assert_called_once_with(str(tmp_path / "model.onnx"))
    convert_float_to_float16.assert_called_once_with(fake_model)
    onnx_save.assert_called_once_with(
        fake_fp16_model,
        str(tmp_path / "model_fp16.onnx"),
    )


def test_export_tensorrt_requires_available_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TensorRT export should fail when ORT lacks the TensorRT provider."""
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    (onnx_dir / "model.onnx").write_bytes(b"onnx")
    fake_ort = _module(
        "onnxruntime",
        get_available_providers=Mock(return_value=["CPUExecutionProvider"]),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    with pytest.raises(RuntimeError, match="TensorrtExecutionProvider not available"):
        nli_export.export_tensorrt(str(onnx_dir))


def test_export_tensorrt_builds_cache_with_int64_tokenizer_feed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TensorRT warmup should pass int64 inputs for the active ONNX inputs."""
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    (onnx_dir / "model.onnx").write_bytes(b"onnx")
    captured: dict[str, object] = {}

    class FakeInput:
        """Fake ONNX input descriptor."""

        def __init__(self, name: str) -> None:
            self.name = name

    class FakeSession:
        """Fake TensorRT-capable ONNX Runtime session."""

        def __init__(
            self,
            model_file: str,
            opts: object,
            providers: list[str | tuple[str, dict[str, object]]],
        ) -> None:
            captured["model_file"] = model_file
            captured["providers"] = providers
            captured["opts"] = opts

        def get_providers(self) -> list[str]:
            """Return TensorRT as the active provider."""
            return ["TensorrtExecutionProvider", "CUDAExecutionProvider"]

        def get_inputs(self) -> list[FakeInput]:
            """Expose the model inputs consumed by the warmup feed."""
            return [FakeInput("input_ids"), FakeInput("attention_mask")]

        def run(
            self,
            outputs: None,
            feed: dict[str, np.ndarray],
        ) -> list[np.ndarray]:
            """Capture the warmup feed and return fake logits."""
            captured["outputs"] = outputs
            captured["feed"] = feed
            return [np.array([[0.1, 0.9]])]

    fake_ort = _module(
        "onnxruntime",
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        get_available_providers=Mock(
            return_value=[
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        ),
        SessionOptions=Mock(
            return_value=SimpleNamespace(
                graph_optimization_level=None,
                log_severity_level=None,
            ),
        ),
        InferenceSession=Mock(side_effect=FakeSession),
    )
    tokenizer = Mock(
        return_value={
            "input_ids": np.array([[1, 2]], dtype=np.int32),
            "attention_mask": np.array([[1, 1]], dtype=np.int64),
            "token_type_ids": np.array([[0, 0]], dtype=np.int32),
        },
    )
    auto_tokenizer = SimpleNamespace(from_pretrained=Mock(return_value=tokenizer))
    fake_transformers = _module("transformers", AutoTokenizer=auto_tokenizer)
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
    providers = cast(list[str | tuple[str, dict[str, object]]], captured["providers"])
    provider, options = cast(tuple[str, dict[str, object]], providers[0])
    assert provider == "TensorrtExecutionProvider"
    assert options["trt_engine_cache_path"] == str(tmp_path / "trt")
    assert options["trt_fp16_enable"] is False
    assert options["trt_profile_opt_shapes"] == ("input_ids=4x128,attention_mask=4x128")
    assert captured["outputs"] is None
    feed = cast(dict[str, np.ndarray], captured["feed"])
    assert set(feed) == {"input_ids", "attention_mask"}
    assert feed["input_ids"].dtype == np.int64
    assert feed["attention_mask"].dtype == np.int64
    assert auto_tokenizer.from_pretrained.call_args.kwargs == {
        "revision": "local-artifact",
        "local_files_only": True,
    }


def test_dynamic_batcher_empty_flush_is_noop() -> None:
    """Flushing an empty dynamic batcher should not call the scorer."""
    calls: list[list[tuple[str, str]]] = []

    def score_fn(pairs: list[tuple[str, str]]) -> list[float]:
        """Record unexpected scoring calls."""
        calls.append(pairs)
        return [0.1]

    batcher = nli_export.OnnxDynamicBatcher(score_fn)

    assert batcher.flush() == []
    assert calls == []


def test_dynamic_batcher_timeout_not_elapsed_before_first_submit() -> None:
    """An unstarted batch never counts as timed out under a non-zero timeout."""
    batcher = nli_export.OnnxDynamicBatcher(lambda pairs: [0.1], flush_timeout_ms=10.0)

    assert batcher._timeout_elapsed(1_000.0) is False
