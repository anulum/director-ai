# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - NLI export real-surface tests
"""Real-surface coverage for public NLI export wiring."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Literal, Protocol, cast

import pytest

from director_ai.core import export_onnx as public_export_onnx
from director_ai.core.nli import export_onnx as compatibility_export_onnx
from director_ai.core.scoring.nli import export_onnx as runtime_export_onnx
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

_PUBLIC_FAKE_LOGITS = object()


class _TorchModule(Protocol):
    """Typed subset of torch needed by the public export path."""

    no_grad: object
    nn: SimpleNamespace
    onnx: SimpleNamespace


class _FakeTorchBaseModule:
    """Minimal ``torch.nn.Module`` base for the runtime export wrapper."""

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Forward calls to ``forward`` like a torch module."""
        return self.forward(*args, **kwargs)

    def forward(self, *args: object, **kwargs: object) -> object:
        """Require subclasses to implement forward."""
        raise NotImplementedError


class _PublicTokenizer:
    """Tokenizer protocol that writes a local artifact during export."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.saved_to: Path | None = None

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str,
        truncation: bool,
        max_length: int,
    ) -> dict[str, object]:
        """Return ONNX-exportable tensor placeholders for the public path."""
        self.calls.append(
            {
                "text": text,
                "return_tensors": return_tensors,
                "truncation": truncation,
                "max_length": max_length,
            }
        )
        return {"input_ids": object(), "attention_mask": object()}

    def save_pretrained(self, output_path: Path) -> None:
        """Write a visible tokenizer artifact like a local export would."""
        self.saved_to = output_path
        (output_path / "tokenizer.json").write_text("{}", encoding="utf-8")


class _PublicModelConfig:
    """Model config protocol that writes a local artifact during export."""

    def __init__(self) -> None:
        self.saved_to: Path | None = None

    def save_pretrained(self, output_path: Path) -> None:
        """Write a visible model config artifact."""
        self.saved_to = output_path
        (output_path / "config.json").write_text("{}", encoding="utf-8")


class _PublicModel:
    """Sequence-classification protocol consumed by export_onnx."""

    def __init__(self) -> None:
        self.config = _PublicModelConfig()
        self.eval_called = False

    def eval(self) -> None:
        """Record that export switched the model to eval mode."""
        self.eval_called = True

    def __call__(self, **inputs: object) -> SimpleNamespace:
        """Return fake logits from the wrapped model."""
        assert set(inputs) == {"input_ids", "attention_mask"}
        return SimpleNamespace(logits=_PUBLIC_FAKE_LOGITS)


def _module(name: str, **attrs: object) -> ModuleType:
    """Return a module object populated with dynamic protocol attributes."""
    module = ModuleType(name)
    for attr_name, value in attrs.items():
        module.__dict__[attr_name] = value
    return module


def _fake_torch() -> _TorchModule:
    """Build a fake torch module that writes the exported ONNX file."""

    class NoGrad:
        """Context manager matching ``torch.no_grad``."""

        def __enter__(self) -> None:
            """Enter the no-grad context."""

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: object,
        ) -> Literal[False]:
            """Exit the no-grad context without suppressing errors."""
            return False

    def no_grad() -> NoGrad:
        """Return the fake no-grad context manager."""
        return NoGrad()

    def export(model: object, args: tuple[object, ...], path: str, **_: object) -> None:
        """Write the ONNX artifact after invoking the runtime wrapper."""
        assert callable(model)
        assert model(*args) is _PUBLIC_FAKE_LOGITS
        Path(path).write_bytes(b"onnx")

    module = _module(
        "torch",
        no_grad=no_grad,
        nn=SimpleNamespace(Module=_FakeTorchBaseModule),
        onnx=SimpleNamespace(export=export),
    )
    return cast(_TorchModule, module)


def test_nli_export_unit_guard_declares_real_surface_companion() -> None:
    """The NLI export unit guard should name its public companion surface."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_nli_export.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_nli_export_real_surface.py" in reason


def test_public_nli_export_paths_share_runtime_exporter() -> None:
    """Public compatibility export paths should resolve to one runtime function."""
    assert public_export_onnx is runtime_export_onnx
    assert compatibility_export_onnx is runtime_export_onnx


def test_public_export_onnx_writes_local_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public export path should produce a complete local ONNX bundle."""
    tokenizer = _PublicTokenizer()
    model = _PublicModel()
    fake_transformers = _module(
        "transformers",
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda model_name, revision: tokenizer,
        ),
        AutoModelForSequenceClassification=SimpleNamespace(
            from_pretrained=lambda model_name, revision: model,
        ),
    )
    fake_torch = _fake_torch()
    monkeypatch.setitem(sys.modules, "torch", cast(ModuleType, fake_torch))
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    output_dir = tmp_path / "public-onnx"

    result = public_export_onnx(
        model_name="test/model",
        output_dir=str(output_dir),
        revision="abc123",
    )

    assert result == str(output_dir)
    assert (output_dir / "model.onnx").read_bytes() == b"onnx"
    assert (output_dir / "tokenizer.json").exists()
    assert (output_dir / "config.json").exists()
    assert model.eval_called is True
    assert model.config.saved_to == output_dir
    assert tokenizer.saved_to == output_dir
    assert tokenizer.calls == [
        {
            "text": (
                "Premise.\n\nChoose your answer: based on the paragraph above "
                'can we conclude that "Hypothesis."?\n\nOPTIONS:\n- Yes\n- No\n'
                "I think the answer is "
            ),
            "return_tensors": "pt",
            "truncation": True,
            "max_length": 512,
        }
    ]
