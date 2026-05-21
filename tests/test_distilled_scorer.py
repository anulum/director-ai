# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for ``director_ai.core.scoring.distilled_scorer``.

Covers construction, lazy loading, ONNX/PyTorch inference paths,
softmax utility, batch scoring, and backend registry integration.
Uses mocks to avoid downloading real models.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

import director_ai.core.scoring.distilled_scorer as distilled_mod
from director_ai.core.scoring.distilled_scorer import (
    DEFAULT_DISTILLED_MODEL,
    DistilledNLIBackend,
    _softmax,
)

# ── _softmax utility ───────────────────────────────────────────────────


class TestSoftmax:
    def test_uniform(self):
        result = _softmax(np.array([0.0, 0.0]))
        np.testing.assert_allclose(result, [0.5, 0.5])

    def test_dominant(self):
        result = _softmax(np.array([10.0, 0.0]))
        assert result[0] > 0.99
        assert result[1] < 0.01

    def test_sums_to_one(self):
        result = _softmax(np.array([1.0, 2.0, 3.0]))
        assert abs(result.sum() - 1.0) < 1e-6

    def test_negative_logits(self):
        result = _softmax(np.array([-10.0, -5.0]))
        assert result.sum() - 1.0 < 1e-6
        assert result[1] > result[0]

    def test_rust_softmax_delegation(self, monkeypatch):
        monkeypatch.setattr(distilled_mod, "_RUST_DISTILLED", True)
        monkeypatch.setattr(
            distilled_mod,
            "rust_softmax",
            lambda flat, cols: [0.8, 0.2],
            raising=False,
        )
        result = _softmax(np.array([1.0, -1.0]))
        np.testing.assert_allclose(result, [0.8, 0.2])

    def test_rust_softmax_non_runtime_fallback(self, monkeypatch):
        monkeypatch.setattr(distilled_mod, "_RUST_DISTILLED", True)
        monkeypatch.setattr(
            distilled_mod,
            "rust_softmax",
            lambda _flat, _cols: (_ for _ in ()).throw(ValueError("ffi unavailable")),
            raising=False,
        )
        result = _softmax(np.array([1.0, -1.0]))
        assert result[0] > result[1]
        assert abs(result.sum() - 1.0) < 1e-6


# ── Construction ────────────────────────────────────────────────────────


class TestConstruction:
    def test_default_model(self):
        b = DistilledNLIBackend()
        assert b._model_path == DEFAULT_DISTILLED_MODEL

    def test_custom_model(self):
        b = DistilledNLIBackend(model_path="/tmp/my-model")
        assert b._model_path == "/tmp/my-model"

    def test_lazy_no_load_at_init(self):
        b = DistilledNLIBackend()
        assert not b._ready
        assert b._session is None
        assert b._model is None

    def test_use_onnx_default(self):
        b = DistilledNLIBackend()
        assert b._use_onnx is True


# ── ONNX inference path (mocked) ───────────────────────────────────────


class TestOnnxPath:
    def _mock_backend(self):
        b = DistilledNLIBackend()
        # Mock ONNX session
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [
            MagicMock(name="input_ids"),
            MagicMock(name="attention_mask"),
        ]
        # Return logits [entailment=2.0, contradiction=-1.0] → P(ent)≈0.95
        mock_session.run.return_value = [np.array([[2.0, -1.0]])]
        b._session = mock_session

        # Mock tokeniser
        mock_tok = MagicMock()
        mock_tok.return_value = {
            "input_ids": np.array([[1, 2, 3]]),
            "attention_mask": np.array([[1, 1, 1]]),
        }
        b._tokeniser = mock_tok
        b._ready = True
        return b

    def test_score_returns_float(self):
        b = self._mock_backend()
        s = b.score("premise", "hypothesis")
        assert isinstance(s, float)

    def test_score_high_entailment(self):
        b = self._mock_backend()
        s = b.score("x", "y")
        assert s > 0.9  # logits [2, -1] → softmax ≈ [0.95, 0.05]

    def test_score_range(self):
        b = self._mock_backend()
        s = b.score("a", "b")
        assert 0.0 <= s <= 1.0

    def test_batch(self):
        b = self._mock_backend()
        scores = b.score_batch([("a", "b"), ("c", "d")])
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)

    def test_empty_batch(self):
        b = self._mock_backend()
        assert b.score_batch([]) == []

    def test_ensure_loaded_uses_local_onnx_without_hub_download(
        self, monkeypatch, tmp_path
    ):
        calls: dict[str, object] = {}
        model_dir = tmp_path / "nli-lite"
        model_dir.mkdir()
        (model_dir / "model.onnx").write_bytes(b"fake-onnx")

        class FakeSession:
            def __init__(self, path, *, providers):
                calls["onnx_path"] = path
                calls["providers"] = providers

            def get_inputs(self):
                return []

            def run(self, outputs, inputs):
                return [np.array([[3.0, 0.0]], dtype=float)]

        class FakeTokenizer:
            @classmethod
            def from_pretrained(cls, model_path, *, revision):
                calls["tokenizer_model_path"] = model_path
                calls["tokenizer_revision"] = revision
                return lambda premise, hypothesis, **kwargs: {}

        fake_ort = types.ModuleType("onnxruntime")
        fake_ort.InferenceSession = FakeSession
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = FakeTokenizer
        monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        backend = DistilledNLIBackend(model_path=str(model_dir))
        score = backend.score("The invoice is paid.", "The invoice is settled.")

        assert backend._ready is True
        assert backend._session is not None
        assert score > 0.95
        assert calls["onnx_path"] == str(model_dir / "model.onnx")
        assert calls["providers"] == ["CPUExecutionProvider"]
        assert calls["tokenizer_model_path"] == str(model_dir)

    def test_local_onnx_rejects_path_outside_allowed_root(self, monkeypatch, tmp_path):
        calls: dict[str, object] = {}
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "model.onnx").write_bytes(b"fake")

        class FakeSession:
            def __init__(self, path, *, providers):
                calls["onnx_path"] = path
                calls["providers"] = providers

        class FakeTokenizer:
            @classmethod
            def from_pretrained(cls, model_path, *, revision):
                calls["tokenizer_model_path"] = model_path
                return MagicMock()

        fake_ort = types.ModuleType("onnxruntime")
        fake_ort.InferenceSession = FakeSession
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = FakeTokenizer
        monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        monkeypatch.setenv("DIRECTOR_ONNX_ALLOWED_DIRS", str(allowed))

        backend = DistilledNLIBackend(model_path=str(outside))
        with pytest.raises(PermissionError, match="DIRECTOR_ONNX_ALLOWED_DIRS"):
            backend._load_onnx()

        assert "onnx_path" not in calls

    def test_local_onnx_rejects_model_file_symlink_escape(self, monkeypatch, tmp_path):
        calls: dict[str, object] = {}
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        external = tmp_path / "external"
        external.mkdir()
        (external / "model.onnx").write_bytes(b"fake")
        model_dir = allowed / "bundle"
        model_dir.mkdir()
        (model_dir / "model.onnx").symlink_to(external / "model.onnx")

        class FakeSession:
            def __init__(self, path, *, providers):
                calls["onnx_path"] = path
                calls["providers"] = providers

        class FakeTokenizer:
            @classmethod
            def from_pretrained(cls, model_path, *, revision):
                calls["tokenizer_model_path"] = model_path
                return MagicMock()

        fake_ort = types.ModuleType("onnxruntime")
        fake_ort.InferenceSession = FakeSession
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = FakeTokenizer
        monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        monkeypatch.setenv("DIRECTOR_ONNX_ALLOWED_DIRS", str(allowed))

        backend = DistilledNLIBackend(model_path=str(model_dir))
        with pytest.raises(PermissionError, match="escapes model directory"):
            backend._load_onnx()

        assert "onnx_path" not in calls

    def test_ensure_loaded_downloads_remote_onnx_with_pinned_revision(
        self, monkeypatch, tmp_path
    ):
        calls: dict[str, object] = {}
        downloaded = tmp_path / "downloaded-model.onnx"
        downloaded.write_bytes(b"fake-onnx")

        class FakeSession:
            def __init__(self, path, *, providers):
                calls["onnx_path"] = path
                calls["providers"] = providers

            def get_inputs(self):
                return []

            def run(self, outputs, inputs):
                return [np.array([[0.0, 3.0]], dtype=float)]

        class FakeTokenizer:
            @classmethod
            def from_pretrained(cls, model_path, *, revision):
                calls["tokenizer_model_path"] = model_path
                calls["tokenizer_revision"] = revision
                return lambda premise, hypothesis, **kwargs: {}

        def hf_hub_download(repo_id, filename, *, revision):
            calls["repo_id"] = repo_id
            calls["filename"] = filename
            calls["download_revision"] = revision
            return str(downloaded)

        fake_ort = types.ModuleType("onnxruntime")
        fake_ort.InferenceSession = FakeSession
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = FakeTokenizer
        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.hf_hub_download = hf_hub_download
        monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

        backend = DistilledNLIBackend(model_path="tenant/nli-lite")
        score = backend.score("Sensor torque is safe.", "Torque is unsafe.")

        assert score < 0.1
        assert calls["repo_id"] == "tenant/nli-lite"
        assert calls["filename"] == "model.onnx"
        assert calls["onnx_path"] == str(downloaded)
        assert calls["download_revision"] == calls["tokenizer_revision"]

    def test_onnx_load_failure_falls_back_to_pytorch(
        self, monkeypatch, tmp_path, caplog
    ):
        calls: dict[str, object] = {}
        model_dir = tmp_path / "broken-onnx"
        model_dir.mkdir()
        (model_dir / "model.onnx").write_bytes(b"broken")

        class FakeTensor:
            def __init__(self, value):
                self.value = value

            def to(self, device):
                calls.setdefault("input_devices", []).append(device)
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

            def __getitem__(self, index):
                assert index == 0
                return FakeTensor(self.value[index])

        class _NoGrad:
            def __enter__(self):
                calls["no_grad_entered"] = True

            def __exit__(self, exc_type, exc, tb):
                calls["no_grad_exited"] = True

        class FakeTokenizer:
            @classmethod
            def from_pretrained(cls, model_path, *, revision):
                calls["tokenizer_model_path"] = model_path
                return cls()

            def __call__(self, premise, hypothesis, **kwargs):
                calls["tokenizer_kwargs"] = kwargs
                return {"input_ids": FakeTensor(np.array([[1, 2, 3]]))}

        class FakeModel:
            @classmethod
            def from_pretrained(cls, model_path, *, revision):
                calls["model_path"] = model_path
                return cls()

            def to(self, device):
                calls["model_device"] = device
                return self

            def eval(self):
                calls["model_eval"] = True
                return self

            def __call__(self, **inputs):
                calls["model_inputs"] = inputs
                return types.SimpleNamespace(logits=FakeTensor(np.array([[1.0, 0.0]])))

        fake_ort = types.ModuleType("onnxruntime")
        fake_ort.InferenceSession = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("corrupt onnx")
        )
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = FakeTokenizer
        fake_transformers.AutoModelForSequenceClassification = FakeModel
        fake_torch = types.ModuleType("torch")
        fake_torch.no_grad = lambda: _NoGrad()

        def softmax(logits, dim):
            calls["softmax_dim"] = dim
            return FakeTensor(np.array([[0.8, 0.2]], dtype=float))

        fake_torch.softmax = softmax
        monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        backend = DistilledNLIBackend(model_path=str(model_dir), device="cpu")
        with caplog.at_level("WARNING", logger="DirectorAI.DistilledNLI"):
            score = backend.score("Fact.", "Claim.")

        assert score == pytest.approx(0.8)
        assert backend._ready is True
        assert calls["model_path"] == str(model_dir)
        assert calls["model_device"] == "cpu"
        assert calls["model_eval"] is True
        assert calls["softmax_dim"] == -1
        assert "ONNX load failed" in caplog.text


# ── PyTorch fallback path (mocked) ─────────────────────────────────────


@__import__("pytest").mark.skipif(
    not __import__("importlib").util.find_spec("torch"),
    reason="torch not installed",
)
class TestPyTorchPath:
    def _mock_backend(self):
        import torch

        b = DistilledNLIBackend(use_onnx=False)
        mock_model = MagicMock()
        mock_model.return_value = MagicMock(logits=torch.tensor([[2.0, -1.0]]))
        b._model = mock_model
        b._torch = torch

        mock_tok = MagicMock()
        mock_tok.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        b._tokeniser = mock_tok
        b._ready = True
        b._device = "cpu"
        return b

    def test_score_pytorch(self):
        b = self._mock_backend()
        s = b.score("premise", "hypothesis")
        assert isinstance(s, float)
        assert s > 0.9


# ── Backend registry ───────────────────────────────────────────────────


class TestRegistry:
    def test_nli_lite_registered(self):
        from director_ai.core.scoring.backends import list_backends

        assert "nli-lite" in list_backends()
