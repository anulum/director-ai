# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - ONNX Backend Tests
"""Multi-angle tests for ONNX NLI backend and export.

Covers heuristic fallback when ONNX is unavailable, bad path handling, batch
scoring, public export importability, native exporter guards, onnxruntime
provider checks, parametrized inputs, score range invariants, CoherenceScorer
pipeline integration, and fallback performance documentation.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Literal, cast

import pytest

from director_ai.core.nli import NLIScorer, export_onnx

_has_onnxruntime = False
try:
    import onnxruntime  # noqa: F401

    _has_onnxruntime = True
except ImportError:
    pass


@dataclass(frozen=True, slots=True)
class _FakeTensor:
    """Small tensor placeholder passed into the fake ONNX exporter."""

    shape: tuple[int, int]
    dtype: object | None


@dataclass(frozen=True, slots=True)
class _FakeModelOutput:
    """Model output placeholder with the logits attribute export expects."""

    logits: object


@dataclass(frozen=True, slots=True)
class _OnnxExportCall:
    """Captured ``torch.onnx.export`` call metadata."""

    model: object
    args: tuple[object, ...]
    file_path: str
    input_names: list[str]
    output_names: list[str]
    dynamic_axes: dict[str, dict[int, str]]
    opset_version: int
    do_constant_folding: bool
    dynamo: bool | None


_FAKE_LOGITS = object()


class _NoGrad:
    """Context manager matching the subset of ``torch.no_grad`` used here."""

    def __enter__(self) -> None:
        """Enter the fake no-grad context."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> Literal[False]:
        """Leave the fake no-grad context without suppressing exceptions."""
        return False


class _NoGradFactory:
    """Callable no-grad factory that records exporter usage."""

    def __init__(self) -> None:
        """Initialise the call counter."""
        self.calls = 0

    def __call__(self) -> _NoGrad:
        """Return a fake no-grad context manager."""
        self.calls += 1
        return _NoGrad()


class _FakeOnes:
    """Callable tensor factory matching the export-time ``torch.ones`` calls."""

    def __init__(self) -> None:
        """Initialise captured tensor creation calls."""
        self.calls: list[_FakeTensor] = []

    def __call__(
        self,
        shape: tuple[int, int],
        *,
        dtype: object | None = None,
    ) -> _FakeTensor:
        """Return a fake tensor with the requested shape and dtype."""
        tensor = _FakeTensor(shape=shape, dtype=dtype)
        self.calls.append(tensor)
        return tensor


class _FakeTorchBaseModule:
    """Minimal ``torch.nn.Module`` base for the runtime export wrapper."""

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Forward calls to ``forward`` like a torch module."""
        return self.forward(*args, **kwargs)

    def forward(self, *args: object, **kwargs: object) -> object:
        """Require subclasses to implement forward."""
        raise NotImplementedError


class _FakeNnNamespace:
    """Typed torch.nn namespace for the export wrapper base class."""

    Module = _FakeTorchBaseModule


class _RecordingOnnxExporter:
    """Callable replacement for ``torch.onnx.export``."""

    def __init__(self) -> None:
        """Initialise captured export calls."""
        self.calls: list[_OnnxExportCall] = []

    def __call__(
        self,
        model: object,
        args: tuple[object, ...],
        f: str,
        *,
        input_names: list[str],
        output_names: list[str],
        dynamic_axes: dict[str, dict[int, str]],
        opset_version: int,
        do_constant_folding: bool,
        dynamo: bool | None = None,
    ) -> None:
        """Record a native ONNX export call and write the fake artifact."""
        wrapped = cast(Callable[..., object], model)
        assert wrapped(*args) is _FAKE_LOGITS
        self.calls.append(
            _OnnxExportCall(
                model=model,
                args=args,
                file_path=f,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=do_constant_folding,
                dynamo=dynamo,
            ),
        )
        Path(f).write_bytes(b"onnx")


class _FakeOnnxNamespace:
    """Typed torch.onnx namespace exposing the native export callable."""

    def __init__(self, exporter: _RecordingOnnxExporter) -> None:
        """Bind the recording export callable."""
        self.export = exporter


class _FakeTorch(ModuleType):
    """ModuleType-backed fake torch module with typed dynamic attributes."""

    long: object
    no_grad: _NoGradFactory
    ones: _FakeOnes
    nn: _FakeNnNamespace
    onnx: _FakeOnnxNamespace

    def __init__(self) -> None:
        """Build a fake torch module for export-only tests."""
        super().__init__("torch")
        self.long = object()
        self.no_grad = _NoGradFactory()
        self.ones = _FakeOnes()
        self.nn = _FakeNnNamespace()
        self.onnx = _FakeOnnxNamespace(_RecordingOnnxExporter())


class _FakeModelConfig:
    """Config protocol that writes an export artifact."""

    def __init__(self) -> None:
        """Initialise the saved output path."""
        self.saved_to: Path | None = None

    def save_pretrained(self, output_path: Path) -> None:
        """Write a visible config artifact."""
        self.saved_to = output_path
        (output_path / "config.json").write_text("{}", encoding="utf-8")


class _FakeModel:
    """Sequence-classification model protocol consumed by export_onnx."""

    def __init__(self) -> None:
        """Initialise model state used by assertions."""
        self.config = _FakeModelConfig()
        self.eval_called = False
        self.forward_inputs: list[dict[str, object]] = []

    def eval(self) -> None:
        """Record that export switched the model into eval mode."""
        self.eval_called = True

    def __call__(self, **inputs: object) -> _FakeModelOutput:
        """Return fake logits for the export wrapper."""
        assert set(inputs) == {"input_ids", "attention_mask"}
        self.forward_inputs.append(inputs)
        return _FakeModelOutput(logits=_FAKE_LOGITS)


class _FakeTokenizer:
    """Tokenizer protocol that emits exportable fake tensors."""

    def __init__(self, torch: _FakeTorch) -> None:
        """Bind the fake torch module used to create tensors."""
        self._torch = torch
        self.calls: list[dict[str, object]] = []
        self.saved_to: Path | None = None

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str,
        truncation: bool,
        max_length: int,
    ) -> dict[str, _FakeTensor]:
        """Return tensor placeholders for the ONNX export inputs."""
        self.calls.append(
            {
                "text": text,
                "return_tensors": return_tensors,
                "truncation": truncation,
                "max_length": max_length,
            },
        )
        return {
            "input_ids": self._torch.ones((1, 4), dtype=self._torch.long),
            "attention_mask": self._torch.ones((1, 4), dtype=self._torch.long),
        }

    def save_pretrained(self, output_path: Path) -> None:
        """Write a visible tokenizer artifact."""
        self.saved_to = output_path
        (output_path / "tokenizer.json").write_text("{}", encoding="utf-8")


class _TokenizerFactory:
    """Factory namespace matching ``AutoTokenizer.from_pretrained``."""

    def __init__(self, tokenizer: _FakeTokenizer) -> None:
        """Bind the tokenizer returned by the factory."""
        self._tokenizer = tokenizer
        self.calls: list[tuple[str, str | None]] = []

    def from_pretrained(
        self,
        model_name: str,
        *,
        revision: str | None = None,
    ) -> _FakeTokenizer:
        """Return the fake tokenizer and record revision pinning."""
        self.calls.append((model_name, revision))
        return self._tokenizer


class _ModelFactory:
    """Factory namespace matching model ``from_pretrained``."""

    def __init__(self, model: _FakeModel) -> None:
        """Bind the model returned by the factory."""
        self._model = model
        self.calls: list[tuple[str, str | None]] = []

    def from_pretrained(
        self,
        model_name: str,
        *,
        revision: str | None = None,
    ) -> _FakeModel:
        """Return the fake model and record revision pinning."""
        self.calls.append((model_name, revision))
        return self._model


class _FakeTransformers(ModuleType):
    """ModuleType-backed fake transformers module for public export tests."""

    AutoTokenizer: _TokenizerFactory
    AutoModelForSequenceClassification: _ModelFactory

    def __init__(self, tokenizer: _FakeTokenizer, model: _FakeModel) -> None:
        """Build the fake transformers module."""
        super().__init__("transformers")
        self.AutoTokenizer = _TokenizerFactory(tokenizer)
        self.AutoModelForSequenceClassification = _ModelFactory(model)


def _install_fake_export_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_FakeTorch, _FakeTransformers, _FakeTokenizer, _FakeModel]:
    """Install typed torch/transformers fakes into ``sys.modules``."""
    fake_torch = _FakeTorch()
    tokenizer = _FakeTokenizer(fake_torch)
    model = _FakeModel()
    fake_transformers = _FakeTransformers(tokenizer, model)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    return fake_torch, fake_transformers, tokenizer, model


class TestOnnxBackendFallback:
    """ONNX backend falls back to heuristic when session unavailable."""

    def test_no_onnx_path(self) -> None:
        """Missing ONNX artifact directory should use heuristic scoring."""
        scorer = NLIScorer(use_model=True, backend="onnx")

        assert scorer.model_available is False
        assert 0.0 <= scorer.score("premise", "hypothesis") <= 1.0

    def test_bad_onnx_path(self) -> None:
        """Nonexistent ONNX artifact directory should be unavailable."""
        scorer = NLIScorer(use_model=True, backend="onnx", onnx_path="/no/such/dir")

        assert scorer.model_available is False

    def test_batch_fallback(self) -> None:
        """Batch fallback should return one bounded score per pair."""
        scorer = NLIScorer(use_model=True, backend="onnx")

        results = scorer.score_batch([("a", "b"), ("c", "d")])

        assert len(results) == 2
        assert all(0.0 <= result <= 1.0 for result in results)

    @pytest.mark.parametrize(
        ("premise", "hypothesis"),
        [
            ("The sky is blue", "The sky is blue"),
            ("", "empty premise"),
            ("test", ""),
            ("日本語", "response"),
        ],
    )
    def test_fallback_various_inputs(self, premise: str, hypothesis: str) -> None:
        """Fallback scoring should be bounded for varied text inputs."""
        scorer = NLIScorer(use_model=True, backend="onnx")

        score = scorer.score(premise, hypothesis)

        assert 0.0 <= score <= 1.0

    def test_fallback_deterministic(self) -> None:
        """Fallback scoring should be deterministic for identical inputs."""
        scorer = NLIScorer(use_model=True, backend="onnx")

        score_one = scorer.score("test", "test")
        score_two = scorer.score("test", "test")

        assert score_one == score_two

    @pytest.mark.parametrize("batch_size", [1, 3, 5, 10])
    def test_batch_various_sizes(self, batch_size: int) -> None:
        """Batch fallback should preserve requested batch size."""
        scorer = NLIScorer(use_model=True, backend="onnx")
        pairs = [("p", "h")] * batch_size

        results = scorer.score_batch(pairs)

        assert len(results) == batch_size


class TestExportOnnx:
    """export_onnx must be importable and use the native modern stack."""

    def test_importable(self) -> None:
        """The public core package should expose export_onnx."""
        from director_ai.core import export_onnx as fn

        assert callable(fn)

    def test_rejects_unknown_quantize_mode(self, tmp_path: Path) -> None:
        """Unsupported quantization modes should fail before model import."""
        with pytest.raises(ValueError, match="quantize"):
            export_onnx(output_dir=str(tmp_path), quantize="nf4")

    def test_export_uses_torch_onnx_not_legacy_exporter(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Export should call native ``torch.onnx.export`` with pinned inputs."""
        fake_torch, fake_transformers, tokenizer, model = _install_fake_export_stack(
            monkeypatch,
        )

        result = export_onnx(
            model_name="test/model",
            output_dir=str(tmp_path),
            revision="abc123",
        )

        assert result == str(tmp_path)
        assert fake_transformers.AutoTokenizer.calls == [("test/model", "abc123")]
        assert fake_transformers.AutoModelForSequenceClassification.calls == [
            ("test/model", "abc123"),
        ]
        assert fake_torch.no_grad.calls == 1
        assert fake_torch.onnx.export.calls
        call = fake_torch.onnx.export.calls[0]
        assert call.file_path == str(tmp_path / "model.onnx")
        assert call.input_names == ["input_ids", "attention_mask"]
        assert call.output_names == ["logits"]
        assert call.opset_version == 17
        assert call.do_constant_folding is True
        assert call.dynamo is False
        assert (tmp_path / "model.onnx").read_bytes() == b"onnx"
        assert (tmp_path / "config.json").exists()
        assert (tmp_path / "tokenizer.json").exists()
        assert model.eval_called is True
        assert model.config.saved_to == tmp_path
        assert tokenizer.saved_to == tmp_path

    def test_export_disables_dynamo_when_dynamic_axes_are_used(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Dynamic axes should keep exporter dynamo mode disabled."""
        fake_torch, _, _, _ = _install_fake_export_stack(monkeypatch)

        export_onnx(
            model_name="test/model",
            output_dir=str(tmp_path),
            revision="abc123",
        )

        call = fake_torch.onnx.export.calls[0]
        assert call.dynamo is False
        assert call.dynamic_axes["input_ids"] == {0: "batch", 1: "sequence"}
        assert call.dynamic_axes["attention_mask"] == {0: "batch", 1: "sequence"}
        assert call.dynamic_axes["logits"] == {0: "batch"}


@pytest.mark.skipif(not _has_onnxruntime, reason="onnxruntime not installed")
class TestOnnxRuntimeAvailable:
    """Tests when onnxruntime is installed."""

    def test_providers_list(self) -> None:
        """ONNX Runtime should expose the CPU provider."""
        import onnxruntime as ort

        providers = ort.get_available_providers()
        assert "CPUExecutionProvider" in providers

    def test_ort_version(self) -> None:
        """ONNX Runtime should expose version metadata."""
        import onnxruntime as ort

        assert hasattr(ort, "__version__")


class TestOnnxPipelineIntegration:
    """ONNX backend must integrate with CoherenceScorer."""

    def test_scorer_with_onnx_backend(self) -> None:
        """CoherenceScorer should accept the ONNX backend option."""
        from director_ai.core import CoherenceScorer

        scorer = CoherenceScorer(use_nli=False, scorer_backend="onnx")
        approved, score = scorer.review("test", "test")

        assert isinstance(approved, bool)
        assert 0.0 <= score.score <= 1.0


class TestOnnxPerformanceDoc:
    """Document ONNX backend performance characteristics."""

    def test_heuristic_fallback_fast(self) -> None:
        """Heuristic fallback should remain sub-millisecond per call."""
        scorer = NLIScorer(use_model=True, backend="onnx")
        for _ in range(10):
            scorer.score("warmup", "warmup")

        started_at = time.perf_counter()
        for _ in range(100):
            scorer.score("test", "test")
        per_call_ms = (time.perf_counter() - started_at) / 100 * 1000

        assert per_call_ms < 1.0, (
            f"ONNX heuristic fallback took {per_call_ms:.3f}ms/call (expected <1ms)"
        )


class TestOnnxPathValidation:
    """ONNX path is resolved before use; bad paths fall back safely."""

    def test_nonexistent_onnx_path_falls_back(self) -> None:
        """A nonexistent absolute ONNX path should fall back safely."""
        scorer = NLIScorer(backend="onnx", onnx_path="/nonexistent/onnx/dir")

        assert scorer.model_available is False
        assert 0.0 <= scorer.score("premise", "hypothesis") <= 1.0

    def test_onnx_path_pointing_at_file_falls_back(self, tmp_path: Path) -> None:
        """A file path supplied as an ONNX directory should fall back safely."""
        not_a_dir = tmp_path / "model.onnx"
        not_a_dir.write_text("", encoding="utf-8")

        scorer = NLIScorer(backend="onnx", onnx_path=str(not_a_dir))

        assert scorer.model_available is False

    def test_relative_onnx_path_resolved_not_passed_through(self) -> None:
        """A relative nonexistent path should resolve and fall back safely."""
        scorer = NLIScorer(backend="onnx", onnx_path="./does-not-exist-rel")

        assert scorer.model_available is False
