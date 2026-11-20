# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — NLI Export Tests
"""Typed guard tests for NLI ONNX export and loading helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
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
        """Validate the export contract and write the graph file like torch."""
        assert isinstance(path, str)
        assert kwargs["input_names"] == ["input_ids", "attention_mask"]
        assert kwargs["output_names"] == ["logits"]
        dynamic_axes = cast(dict[str, dict[int, str]], kwargs["dynamic_axes"])
        assert dynamic_axes["logits"] == {0: "batch"}
        assert model(*args) is _FAKE_LOGITS
        Path(path).write_bytes(b"onnx-graph")

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
    quantize_dynamic = Mock(
        side_effect=lambda src, dst, **_kwargs: Path(dst).write_bytes(b"int8")
    )
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
        int8_reduce_range=True,
    )

    assert result == str(tmp_path)
    quantize_dynamic.assert_called_once_with(
        str(tmp_path / "model.onnx"),
        str(tmp_path / "model_quantized.onnx"),
        weight_type="qint8",
        reduce_range=True,
    )
    manifest = _manifest(tmp_path)
    assert manifest["model_file"] == "model_quantized.onnx"
    assert manifest["precision"] == "int8"
    assert manifest["int8_reduce_range"] is True
    assert manifest["sha256"] == hashlib.sha256(b"int8").hexdigest()


def test_export_onnx_fp16_quantization_converts_and_saves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FP16 export should convert and save the ONNX graph."""
    fake_torch = _fake_torch()
    fake_transformers = _fake_transformers()
    fake_model = object()
    onnx_load = Mock(return_value=fake_model)
    onnx_save = Mock(side_effect=lambda _model, dst: Path(dst).write_bytes(b"fp16"))
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
    manifest = _manifest(tmp_path)
    assert manifest["model_file"] == "model_fp16.onnx"
    assert manifest["precision"] == "fp16"
    assert manifest["int8_reduce_range"] is None
    onnx_dir, served = nli_export._resolve_onnx_model_file(str(tmp_path))
    assert served == onnx_dir / "model_fp16.onnx"


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


def _manifest(directory: Path) -> dict[str, object]:
    """Read the export manifest written into ``directory``."""
    payload = json.loads(
        (directory / nli_export.ONNX_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert isinstance(payload, dict)
    return payload


def _bind(
    directory: Path,
    precision: str,
    contents: bytes,
    *,
    int8_reduce_range: bool = True,
) -> Path:
    """Write one model file and a manifest that binds it."""
    directory.mkdir(parents=True, exist_ok=True)
    name = {"fp32": "model.onnx", "int8": "model_quantized.onnx"}.get(
        precision, "model_fp16.onnx"
    )
    (directory / name).write_bytes(contents)
    return nli_export._write_onnx_manifest(
        directory,
        precision=precision,
        source_model="test/model",
        source_revision="abc123",
        int8_reduce_range=int8_reduce_range if precision == "int8" else None,
    )


@pytest.mark.parametrize(
    ("quantize", "reduce_range", "message"),
    [
        ("int8", None, "needs int8_reduce_range"),
        (None, True, "applies only to int8"),
        ("fp16", False, "applies only to int8"),
    ],
)
def test_export_onnx_requires_an_explicit_int8_range_only_for_int8(
    tmp_path: Path,
    quantize: str | None,
    reduce_range: bool | None,
    message: str,
) -> None:
    """INT8 needs a range policy for the target CPU; other precisions refuse one."""
    with pytest.raises(ValueError, match=message):
        nli_export.export_onnx(
            output_dir=str(tmp_path),
            quantize=quantize,
            int8_reduce_range=reduce_range,
        )


def test_fp32_export_binds_model_onnx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A default export binds model.onnx with its digest."""
    monkeypatch.setitem(sys.modules, "torch", cast(ModuleType, _fake_torch()))
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers().module)

    nli_export.export_onnx(
        model_name="test/model", output_dir=str(tmp_path), revision="abc123"
    )

    manifest = _manifest(tmp_path)
    assert manifest["model_file"] == "model.onnx"
    assert manifest["precision"] == "fp32"
    assert manifest["source_revision"] == "abc123"
    assert (
        manifest["sha256"]
        == hashlib.sha256((tmp_path / "model.onnx").read_bytes()).hexdigest()
    )


def test_manifest_serves_its_file_instead_of_the_legacy_choice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A manifest-bound FP32 file is served on CPU even beside an INT8 file."""
    nli_export._load_onnx_session.cache_clear()
    bundle = tmp_path / "bundle"
    _bind(bundle, "fp32", b"fp32")
    (bundle / "model_quantized.onnx").write_bytes(b"int8")
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
        """Capture the served model file."""
        captured["model_file"] = model_file
        return FakeSession()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        _module(
            "transformers",
            AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=object())),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        _module(
            "onnxruntime",
            GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
            get_available_providers=Mock(return_value=["CPUExecutionProvider"]),
            SessionOptions=Mock(
                return_value=SimpleNamespace(
                    graph_optimization_level=None, log_severity_level=None
                )
            ),
            InferenceSession=Mock(side_effect=inference_session),
        ),
    )

    nli_export._load_onnx_session(str(bundle), device="cpu")

    assert captured["model_file"] == b"fp32"


def test_digest_mismatch_is_refused(tmp_path: Path) -> None:
    """A model changed after export is refused, never served."""
    bundle = tmp_path / "bundle"
    _bind(bundle, "int8", b"int8")
    (bundle / "model_quantized.onnx").write_bytes(b"tampered")

    with pytest.raises(PermissionError, match="digest does not match"):
        nli_export._resolve_onnx_model_file(str(bundle))


def test_policy_refusal_is_raised_even_without_the_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused path raises before the runtime import is attempted."""
    nli_export._load_onnx_session.cache_clear()
    bundle = tmp_path / "bundle"
    _bind(bundle, "fp32", b"fp32")
    (bundle / "model.onnx").write_bytes(b"tampered")
    monkeypatch.setitem(sys.modules, "onnxruntime", cast(ModuleType, None))

    with pytest.raises(PermissionError, match="digest does not match"):
        nli_export._load_onnx_session(str(bundle))


@pytest.mark.parametrize(
    "payload",
    [
        "{not json",
        json.dumps({"schema": "other"}),
        json.dumps(
            {
                "schema": "director-ai.onnx-export.v1",
                "model_file": "model.onnx",
                "precision": "int8",
                "sha256": "0" * 64,
            }
        ),
        json.dumps(
            {
                "schema": "director-ai.onnx-export.v1",
                "model_file": "../model.onnx",
                "precision": "fp32",
                "sha256": "0" * 64,
            }
        ),
        json.dumps(
            {
                "schema": "director-ai.onnx-export.v1",
                "model_file": "model.onnx",
                "precision": "fp32",
                "sha256": "short",
            }
        ),
        json.dumps(
            {
                "schema": "director-ai.onnx-export.v1",
                "model_file": "model.onnx",
                "precision": "fp32",
                "sha256": "0" * 64,
                "int8_reduce_range": True,
            }
        ),
    ],
)
def test_malformed_manifests_are_refused(tmp_path: Path, payload: str) -> None:
    """Unreadable, unknown or inconsistent manifests are refused."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.onnx").write_bytes(b"fp32")
    (bundle / nli_export.ONNX_MANIFEST_FILENAME).write_text(payload, encoding="utf-8")

    with pytest.raises(PermissionError, match="ONNX manifest"):
        nli_export._resolve_onnx_model_file(str(bundle))


@pytest.mark.parametrize("replacement", ["absent", "directory"])
def test_manifest_naming_a_missing_file_is_refused(
    tmp_path: Path, replacement: str
) -> None:
    """A bound file that is gone, or not a regular file, refuses the load."""
    bundle = tmp_path / "bundle"
    _bind(bundle, "fp32", b"fp32")
    (bundle / "model.onnx").unlink()
    if replacement == "directory":
        (bundle / "model.onnx").mkdir()

    with pytest.raises(PermissionError, match="names a missing model file"):
        nli_export._resolve_onnx_model_file(str(bundle))


def test_int8_manifest_needs_a_boolean_range_policy(tmp_path: Path) -> None:
    """An INT8 manifest without its range policy is malformed."""
    bundle = tmp_path / "bundle"
    manifest = _bind(bundle, "int8", b"int8")
    payload = _manifest(bundle)
    payload["int8_reduce_range"] = None
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PermissionError, match="ONNX manifest is malformed"):
        nli_export._resolve_onnx_model_file(str(bundle))


def test_session_loads_the_verified_bytes_and_refuses_a_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bytes checked against the manifest are the bytes that are loaded."""
    bundle = tmp_path / "bundle"
    _bind(bundle, "fp32", b"fp32")
    binding = nli_export._manifest_binding(bundle)
    assert binding is not None
    assert nli_export._verified_model_bytes(binding) == b"fp32"

    (bundle / "model.onnx").write_bytes(b"swapped")
    with pytest.raises(PermissionError, match="digest does not match"):
        nli_export._verified_model_bytes(binding)


def test_manifest_write_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed write keeps the previous manifest and leaves no temporary file."""
    bundle = tmp_path / "bundle"
    _bind(bundle, "fp32", b"fp32")
    before = _manifest(bundle)

    def fail_replace(source: str, target: Path) -> None:
        """Simulate a crash between writing and moving the manifest."""
        raise OSError("disk gone")

    monkeypatch.setattr(os, "replace", fail_replace)
    (bundle / "model.onnx").write_bytes(b"new")
    with pytest.raises(OSError, match="disk gone"):
        _bind(bundle, "fp32", b"new")

    assert _manifest(bundle) == before
    assert sorted(p.name for p in bundle.iterdir()) == [
        nli_export.ONNX_MANIFEST_FILENAME,
        "model.onnx",
    ]


def test_legacy_directory_warns_once_and_strict_mode_refuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A manifest-less export warns once; the strict setting refuses it."""
    bundle = tmp_path / "legacy"
    bundle.mkdir()
    (bundle / "model.onnx").write_bytes(b"fp32")
    monkeypatch.delenv("DIRECTOR_ONNX_REQUIRE_MANIFEST", raising=False)

    with caplog.at_level("WARNING", logger="DirectorAI.NLI"):
        nli_export._resolve_onnx_model_file(str(bundle))
        nli_export._resolve_onnx_model_file(str(bundle))
    warnings = [r for r in caplog.records if "no export manifest" in r.getMessage()]
    assert len(warnings) == 1
    assert "director-ai export --format onnx" in warnings[0].getMessage()

    monkeypatch.setenv("DIRECTOR_ONNX_REQUIRE_MANIFEST", "1")
    with pytest.raises(PermissionError, match="DIRECTOR_ONNX_REQUIRE_MANIFEST=1"):
        nli_export._resolve_onnx_model_file(str(bundle))


def test_cpu_flags_are_read_from_cpuinfo(tmp_path: Path) -> None:
    """The flags line is parsed; a missing line or file gives ``None``."""
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\nflags\t\t: sse4_2 avx avx512_vnni\n")
    assert nli_export._cpu_flags(cpuinfo) == frozenset({"sse4_2", "avx", "avx512_vnni"})
    cpuinfo.write_text("processor\t: 0\n")
    assert nli_export._cpu_flags(cpuinfo) is None
    assert nli_export._cpu_flags(tmp_path / "absent") is None


@pytest.mark.parametrize(
    ("machine", "flags", "expected"),
    [
        ("aarch64", None, True),
        ("x86_64", frozenset({"avx", "avx_vnni"}), True),
        ("AMD64", frozenset({"amx_int8"}), True),
        ("x86_64", frozenset({"avx", "avx2"}), False),
        ("x86_64", None, None),
    ],
)
def test_full_range_int8_support_follows_the_cpu(
    monkeypatch: pytest.MonkeyPatch,
    machine: str,
    flags: frozenset[str] | None,
    expected: bool | None,
) -> None:
    """ARM and VNNI or AMX x86 CPUs qualify; unreadable flags are unknown."""
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setattr(nli_export, "_cpu_flags", lambda: flags)

    assert nli_export._full_range_int8_supported() is expected


@pytest.mark.parametrize(
    ("supported", "reason"),
    [(False, "has no AVX512-VNNI"), (None, "could not be read")],
)
def test_full_range_int8_mismatch_warns_or_refuses_in_strict_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    supported: bool | None,
    reason: str,
) -> None:
    """A full-range INT8 model on an unconfirmed CPU warns, or refuses."""
    bundle = tmp_path / "bundle"
    _bind(bundle, "int8", b"int8", int8_reduce_range=False)
    monkeypatch.setattr(nli_export, "_full_range_int8_supported", lambda: supported)
    monkeypatch.delenv("DIRECTOR_ONNX_REQUIRE_MANIFEST", raising=False)

    with caplog.at_level("WARNING", logger="DirectorAI.NLI"):
        _, served = nli_export._resolve_onnx_model_file(str(bundle))
    assert served.name == "model_quantized.onnx"
    assert any(reason in r.getMessage() for r in caplog.records)

    monkeypatch.setenv("DIRECTOR_ONNX_REQUIRE_MANIFEST", "1")
    with pytest.raises(PermissionError, match="int8_reduce_range=True"):
        nli_export._resolve_onnx_model_file(str(bundle))


def test_full_range_int8_on_a_supporting_cpu_is_silent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A full-range INT8 model on a VNNI CPU loads without a warning."""
    bundle = tmp_path / "bundle"
    _bind(bundle, "int8", b"int8", int8_reduce_range=False)
    monkeypatch.setattr(nli_export, "_full_range_int8_supported", lambda: True)
    monkeypatch.setenv("DIRECTOR_ONNX_REQUIRE_MANIFEST", "1")

    with caplog.at_level("WARNING", logger="DirectorAI.NLI"):
        nli_export._resolve_onnx_model_file(str(bundle))
    assert not caplog.records


def test_legacy_fp16_beside_another_export_is_refused(tmp_path: Path) -> None:
    """Without a manifest an FP16 file next to FP32 is ambiguous and refused."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.onnx").write_bytes(b"fp32")
    (bundle / "model_fp16.onnx").write_bytes(b"fp16")

    with pytest.raises(PermissionError, match="FP16 export next to other models"):
        nli_export._resolve_onnx_model_file(str(bundle))
