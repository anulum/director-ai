# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — NLI backend contract tests

from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import director_ai.core.scoring._nli_accel as nli_accel
import director_ai.core.scoring._nli_provisioning as nli_provisioning
import director_ai.core.scoring.nli as nli_mod
from director_ai.core.metrics import metrics
from director_ai.core.scoring.backends import ScorerBackend
from director_ai.core.scoring.nli import NLIScorer


class StaticBackend(ScorerBackend):
    def score(self, premise: str, hypothesis: str) -> float:
        return 0.2 if premise in hypothesis else 0.8

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [self.score(premise, hypothesis) for premise, hypothesis in pairs]


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.device = "cpu"

    def to(self, _device: str):
        return self

    def numel(self) -> int:
        return int(self.values.size)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeTokenizer:
    def __init__(self):
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if kwargs.get("return_tensors") == "np":
            batch = len(args[0]) if args and isinstance(args[0], list) else 1
            return {
                "input_ids": np.ones((batch, 3), dtype=np.int32),
                "attention_mask": np.ones((batch, 3), dtype=np.int64),
                "ignored": np.ones((batch, 3), dtype=np.int32),
            }
        batch = len(args[0]) if args and isinstance(args[0], list) else 1
        return {
            "input_ids": FakeTensor(np.ones((batch, 3), dtype=np.int64)),
            "attention_mask": FakeTensor(np.ones((batch, 3), dtype=np.int64)),
        }


class FakeModel:
    def __init__(self, logits):
        self.logits = np.asarray(logits, dtype=np.float64)
        self.config = SimpleNamespace(
            id2label={0: "entailment", 1: "neutral", 2: "contradiction"}
        )
        self.moved_to: str | None = None
        self.eval_called = False

    def parameters(self):
        return iter([SimpleNamespace(device="cpu")])

    def __call__(self, **_inputs):
        return SimpleNamespace(logits=FakeTensor(self.logits))

    def to(self, device: str):
        self.moved_to = device
        return self

    def eval(self):
        self.eval_called = True
        return self


class FakeOnnxSession:
    def __init__(self, logits):
        self.logits = np.asarray(logits, dtype=np.float64)
        self.feed_seen: dict | None = None

    def get_inputs(self):
        return [
            SimpleNamespace(name="input_ids"),
            SimpleNamespace(name="attention_mask"),
        ]

    def run(self, _outputs, feed):
        self.feed_seen = feed
        return [self.logits]


def _install_fake_torch(monkeypatch):
    fake_torch = ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float32 = "float32"

    @contextmanager
    def no_grad():
        yield

    def softmax(tensor, dim: int):
        arr = tensor.values if isinstance(tensor, FakeTensor) else np.asarray(tensor)
        exp = np.exp(arr - arr.max(axis=dim, keepdims=True))
        return FakeTensor(exp / exp.sum(axis=dim, keepdims=True))

    fake_torch.no_grad = no_grad
    fake_torch.softmax = softmax
    fake_torch.nn = SimpleNamespace(
        Softmax=lambda dim: lambda tensor: softmax(tensor, dim)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return fake_torch


def test_nli_model_source_resolves_gcs_via_managed_artifact(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        nli_provisioning,
        "_download_gcs_model_artifact",
        lambda uri: calls.append(uri) or "/cache/model",
    )

    assert nli_provisioning._resolve_model_source("gs://bucket/model") == "/cache/model"
    assert nli_provisioning._resolve_model_source("local/model") == "local/model"
    assert calls == ["gs://bucket/model"]


@pytest.mark.parametrize(
    "uri",
    ["local/model", "gs://", "gs://bucket", "gs://bucket/"],
)
def test_nli_split_gs_uri_rejects_invalid_locations(uri: str) -> None:
    with pytest.raises(ValueError):
        nli_provisioning._split_gs_uri(uri)


def test_nli_split_gs_uri_accepts_bucket_and_prefix() -> None:
    assert nli_provisioning._split_gs_uri("gs://bucket/path/to/model") == (
        "bucket",
        "path/to/model",
    )


def test_nli_safe_cache_name_is_stable_and_filesystem_safe() -> None:
    first = nli_provisioning._safe_cache_name("gs://bucket/path with spaces/model")
    second = nli_provisioning._safe_cache_name("gs://bucket/path with spaces/model")

    assert first == second
    assert " " not in first
    assert first.endswith(second[-16:])


@pytest.mark.parametrize(
    ("rel_path", "expected"),
    [
        ("checkpoint-10/model.safetensors", True),
        ("nested/checkpoint-99/config.json", True),
        ("optimizer.pt", True),
        ("nested/trainer_state.json", True),
        ("model.safetensors", False),
    ],
)
def test_nli_artifact_skip_policy(rel_path: str, expected: bool) -> None:
    assert nli_provisioning._should_skip_artifact(rel_path) is expected


def test_nli_download_gcs_artifact_uses_cache_marker(tmp_path, monkeypatch) -> None:
    cache_root = tmp_path / "hf"
    target = (
        cache_root
        / "director-ai-scorers"
        / nli_provisioning._safe_cache_name("gs://bucket/path/model")
    )
    target.mkdir(parents=True)
    (target / ".director-ai-complete").write_text("cached\n", encoding="utf-8")
    monkeypatch.setenv("DIRECTOR_MODEL_CACHE_DIR", str(cache_root))

    assert nli_provisioning._download_gcs_model_artifact(
        "gs://bucket/path/model"
    ) == str(target)


def test_nli_download_gcs_artifact_fetches_non_training_files(tmp_path, monkeypatch):
    cache_root = tmp_path / "hf"
    monkeypatch.setenv("DIRECTOR_MODEL_CACHE_DIR", str(cache_root))
    downloaded: list[str] = []

    class FakeBlob:
        def __init__(self, name: str):
            self.name = name

        def download_to_filename(self, path: str) -> None:
            downloaded.append(path)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(self.name)

    class FakeClient:
        def bucket(self, name: str) -> str:
            assert name == "bucket"
            return name

        def list_blobs(self, bucket: str, prefix: str):
            assert bucket == "bucket"
            assert prefix == "path/model/"
            return [
                FakeBlob("path/model/config.json"),
                FakeBlob("path/model/checkpoint-1/trainer_state.json"),
                FakeBlob("path/model/model.safetensors"),
            ]

    storage_module = ModuleType("google.cloud.storage")
    storage_module.Client = FakeClient
    cloud_module = ModuleType("google.cloud")
    cloud_module.storage = storage_module
    google_module = ModuleType("google")
    google_module.cloud = cloud_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud_module)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", storage_module)

    model_dir = nli_provisioning._download_gcs_model_artifact("gs://bucket/path/model")

    assert (tmp_path / "hf").as_posix() in model_dir
    assert len(downloaded) == 2
    assert (nli_provisioning.Path(model_dir) / ".director-ai-complete").exists()


def test_nli_download_gcs_artifact_fails_when_no_model_files(tmp_path, monkeypatch):
    monkeypatch.setenv("DIRECTOR_MODEL_CACHE_DIR", str(tmp_path / "hf"))

    class FakeClient:
        def bucket(self, name: str) -> str:
            return name

        def list_blobs(self, bucket: str, prefix: str):
            return [SimpleNamespace(name="path/model/checkpoint-1/trainer_state.json")]

    storage_module = ModuleType("google.cloud.storage")
    storage_module.Client = FakeClient
    cloud_module = ModuleType("google.cloud")
    cloud_module.storage = storage_module
    google_module = ModuleType("google")
    google_module.cloud = cloud_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud_module)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", storage_module)

    with pytest.raises(FileNotFoundError, match="no model artefact files"):
        nli_provisioning._download_gcs_model_artifact("gs://bucket/path/model")


def test_nli_numpy_softmax_and_reducers_use_python_path(monkeypatch) -> None:
    monkeypatch.setattr(nli_accel, "_RUST_NLI", False)
    logits = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64)

    probs = nli_mod._softmax_np(logits)

    assert probs.shape == (2, 2)
    assert probs.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert nli_mod._probs_to_divergence(probs) == pytest.approx(
        [1.0 - probs[0, 1], 1.0 - probs[1, 1]]
    )
    confidences = nli_mod._probs_to_confidence(probs)
    assert all(0.0 <= value <= 1.0 for value in confidences)


def test_nli_rust_reducer_adapters_are_used_for_large_batches(monkeypatch) -> None:
    monkeypatch.setattr(nli_accel, "_RUST_NLI", True)
    monkeypatch.setattr(
        nli_accel,
        "rust_softmax",
        lambda flat, cols: [1.0 / cols for _ in flat],
    )
    monkeypatch.setattr(
        nli_accel,
        "rust_probs_to_divergence",
        lambda flat, ncols, ci, ni: [float(ci + ni + ncols)] * (len(flat) // ncols),
    )
    monkeypatch.setattr(
        nli_accel,
        "rust_probs_to_confidence",
        lambda flat, ncols: [0.42] * (len(flat) // ncols),
    )
    logits = np.ones((50, 3), dtype=np.float64)
    probs = np.ones((10, 3), dtype=np.float64) / 3.0

    assert nli_mod._softmax_np(logits).shape == (50, 3)
    assert nli_mod._probs_to_divergence(probs, (2, 1)) == [6.0] * 10
    assert nli_mod._probs_to_confidence(probs) == [0.42] * 10


def test_nli_three_class_divergence_respects_label_indices(monkeypatch) -> None:
    monkeypatch.setattr(nli_accel, "_RUST_NLI", False)
    probs = np.array([[0.7, 0.2, 0.1]], dtype=np.float64)

    assert nli_mod._probs_to_divergence(probs, (2, 1)) == pytest.approx([0.2])
    assert nli_mod._probs_to_divergence(probs, (0, 1)) == pytest.approx([0.8])


def test_nli_numeric_helpers_handle_empty_and_weighted_values() -> None:
    assert nli_mod._sum_float_list([]) == 0.0
    assert nli_mod._mean_float([]) == 0.0
    assert nli_mod._count_below_threshold([], 0.5) == 0
    assert nli_mod._weighted_sum_float([], []) == 0.0
    assert nli_mod._mean_float([1.0, 3.0]) == pytest.approx(2.0)
    assert nli_mod._count_below_threshold([0.1, 0.7, 0.2], 0.5) == 2
    assert nli_mod._weighted_sum_float([0.2, 0.8], [2.0, 1.0]) == pytest.approx(1.2)


def test_nli_resolve_label_indices_uses_model_config() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            id2label={0: "entailment", 1: "neutral", 2: "contradict"}
        )
    )

    assert nli_mod._resolve_label_indices(model) == (2, 1)
    assert nli_mod._resolve_label_indices(SimpleNamespace(config=None)) == (2, 1)


def test_nli_available_handles_missing_and_mocked_specs(monkeypatch) -> None:
    import importlib.util

    def missing_transformers(name: str):
        if name == "transformers":
            return None
        return object()

    monkeypatch.setattr(importlib.util, "find_spec", missing_transformers)
    assert nli_mod.nli_available() is False

    def mocked_modules(_name: str):
        raise ValueError("mock module has no spec")

    monkeypatch.setattr(importlib.util, "find_spec", mocked_modules)
    assert nli_mod.nli_available() is True


def test_nli_clear_model_cache_releases_cuda(monkeypatch) -> None:
    calls: list[str] = []
    fake_device = ModuleType("director_ai.core._device")
    fake_device.release_torch_cuda = lambda: calls.append("release")
    monkeypatch.setitem(sys.modules, "director_ai.core._device", fake_device)

    nli_mod.clear_model_cache()

    assert calls == ["release"]


def test_nli_load_model_success_uses_revision_dtype_and_device(monkeypatch) -> None:
    _install_fake_torch(monkeypatch)
    fake_model = FakeModel([[3.0, 1.0, 0.0]])
    loaded: dict[str, object] = {}

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_source, **kwargs):
            loaded["tokenizer"] = (model_source, kwargs)
            return FakeTokenizer()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_source, **kwargs):
            loaded["model"] = (model_source, kwargs)
            return fake_model

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    fake_transformers.AutoModelForSequenceClassification = FakeAutoModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(nli_provisioning, "_resolve_revision", lambda *_args: "abc123")
    nli_mod._load_nli_model.cache_clear()

    tokenizer, model = nli_mod._load_nli_model(
        "model/name",
        device="cpu",
        torch_dtype="float16",
        revision=None,
    )

    assert isinstance(tokenizer, FakeTokenizer)
    assert model is fake_model
    assert fake_model.moved_to == "cpu"
    assert fake_model.eval_called is True
    assert loaded["tokenizer"] == (
        "model/name",
        {"use_fast": False, "revision": "abc123"},
    )
    assert loaded["model"][1]["torch_dtype"] == "float16"


def test_nli_load_model_auto_selects_cuda_device(monkeypatch) -> None:
    _install_fake_torch(monkeypatch)
    fake_model = FakeModel([[1.0, 0.0]])
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: FakeTokenizer()
    )
    fake_transformers.AutoModelForSequenceClassification = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: fake_model
    )
    fake_device = ModuleType("director_ai.core._device")
    fake_device.select_torch_device = lambda: "cuda:0"
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "director_ai.core._device", fake_device)
    monkeypatch.setattr(nli_provisioning, "_resolve_revision", lambda *_args: "abc123")
    nli_mod._load_nli_model.cache_clear()

    _, model = nli_mod._load_nli_model("model/name")

    assert model is fake_model
    assert fake_model.moved_to == "cuda:0"


def test_nli_load_model_quantized_path_skips_manual_device_move(monkeypatch) -> None:
    _install_fake_torch(monkeypatch)
    fake_model = FakeModel([[1.0, 2.0]])

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_transformers = ModuleType("transformers")
    fake_transformers.BitsAndBytesConfig = FakeBitsAndBytesConfig
    fake_transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: FakeTokenizer()
    )
    fake_transformers.AutoModelForSequenceClassification = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: fake_model
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(nli_provisioning, "_resolve_revision", lambda *_args: "abc123")
    nli_mod._load_nli_model.cache_clear()

    _, model = nli_mod._load_nli_model(
        "model/name",
        quantize_8bit=True,
        device="cpu",
    )

    assert model is fake_model
    assert fake_model.moved_to is None


def test_nli_load_model_falls_back_when_import_fails(monkeypatch) -> None:
    _install_fake_torch(monkeypatch)
    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    nli_mod._load_nli_model.cache_clear()

    tokenizer, model = nli_mod._load_nli_model("model/name")

    assert tokenizer is None
    assert model is None


def test_nli_custom_backend_routes_score_and_confidence() -> None:
    scorer = NLIScorer(backend=StaticBackend())

    assert scorer.model_available is True
    assert scorer.score("sky", "sky is blue") == pytest.approx(0.2)
    assert scorer.score_batch([("sky", "sky is blue"), ("earth", "sky is blue")]) == [
        0.2,
        0.8,
    ]
    assert scorer.score_batch_with_confidence([("sky", "sky is blue")]) == [(0.2, 1.0)]


def test_nli_builtin_backend_ready_states() -> None:
    assert NLIScorer(backend="lite")._backend_ready is True
    onnx = NLIScorer(backend="onnx")
    assert onnx._backend_ready is False
    onnx._onnx_session = object()
    assert onnx._backend_ready is True


def test_nli_ensure_model_handles_disabled_and_onnx_missing_path() -> None:
    disabled = NLIScorer(use_model=False)
    assert disabled._ensure_model() is False
    assert disabled._model_loaded is True

    onnx = NLIScorer(backend="onnx", onnx_path=None)
    assert onnx._ensure_model() is False
    assert onnx._model_loaded is True


def test_nli_ensure_model_loads_onnx_session(monkeypatch) -> None:
    tokenizer = FakeTokenizer()
    session = FakeOnnxSession([[0.0, 2.0]])
    monkeypatch.setattr(
        nli_mod, "_load_onnx_session", lambda *_args, **_kwargs: (tokenizer, session)
    )

    scorer = NLIScorer(backend="onnx", onnx_path="model.onnx", onnx_batch_size=3)

    assert scorer._ensure_model() is True
    assert scorer._tokenizer is tokenizer
    assert scorer._onnx_session is session
    assert scorer._onnx_batcher is not None


def test_nli_ensure_model_loads_deberta_and_lora(monkeypatch) -> None:
    model = FakeModel([[3.0, 0.0, 1.0]])
    calls: list[str] = []
    monkeypatch.setattr(
        nli_mod, "_load_nli_model", lambda *_args, **_kwargs: (FakeTokenizer(), model)
    )
    scorer = NLIScorer(lora_adapter_path="adapter")
    monkeypatch.setattr(scorer, "_load_lora_adapter", lambda path: calls.append(path))

    assert scorer._ensure_model() is True
    assert scorer._label_indices == (2, 1)
    assert calls == ["adapter"]


def test_nli_ensure_model_loads_deberta_without_lora(monkeypatch) -> None:
    model = FakeModel([[3.0, 0.0, 1.0]])
    monkeypatch.setattr(
        nli_mod, "_load_nli_model", lambda *_args, **_kwargs: (FakeTokenizer(), model)
    )
    scorer = NLIScorer()
    monkeypatch.setattr(
        scorer,
        "_load_lora_adapter",
        lambda _path: pytest.fail("LoRA adapter should not load without a path"),
    )

    assert scorer._ensure_model() is True
    assert scorer._label_indices == (2, 1)


def test_nli_ensure_model_handles_unavailable_deberta(monkeypatch) -> None:
    monkeypatch.setattr(
        nli_mod, "_load_nli_model", lambda *_args, **_kwargs: (None, None)
    )
    scorer = NLIScorer()

    assert scorer._ensure_model() is False
    assert scorer._model_loaded is True


def test_nli_lora_adapter_merges_when_peft_available(monkeypatch) -> None:
    base = FakeModel([[1.0, 0.0]])
    merged = FakeModel([[0.0, 1.0]])

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model, path):
            assert model is base
            assert path == "adapter"
            return SimpleNamespace(merge_and_unload=lambda: merged)

    peft_module = ModuleType("peft")
    peft_module.PeftModel = FakePeftModel
    monkeypatch.setitem(sys.modules, "peft", peft_module)
    scorer = NLIScorer(use_model=False)
    scorer._model = base

    scorer._load_lora_adapter("adapter")

    assert scorer._model is merged
    assert merged.eval_called is True


def test_nli_lora_adapter_logs_when_peft_missing_or_invalid(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "peft", raising=False)
    scorer = NLIScorer(use_model=False)
    scorer._model = FakeModel([[1.0, 0.0]])
    scorer._load_lora_adapter("adapter")

    peft_module = ModuleType("peft")
    peft_module.PeftModel = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("bad")
        )
    )
    monkeypatch.setitem(sys.modules, "peft", peft_module)
    scorer._load_lora_adapter("adapter")


def test_nli_lora_adapter_requires_loaded_base_model(monkeypatch) -> None:
    peft_module = ModuleType("peft")
    peft_module.PeftModel = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: None
    )
    monkeypatch.setitem(sys.modules, "peft", peft_module)
    scorer = NLIScorer(use_model=False)

    with pytest.raises(RuntimeError, match="before base NLI model"):
        scorer._load_lora_adapter("adapter")


def test_nli_lite_backend_routes_single_and_batch(monkeypatch) -> None:
    class FakeLiteScorer:
        def score(self, premise: str, hypothesis: str) -> float:
            return 0.25 if premise in hypothesis else 0.75

        def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [self.score(premise, hypothesis) for premise, hypothesis in pairs]

    fake_lite = ModuleType("director_ai.core.scoring.lite_scorer")
    fake_lite.LiteScorer = FakeLiteScorer
    monkeypatch.setitem(sys.modules, "director_ai.core.scoring.lite_scorer", fake_lite)
    scorer = NLIScorer(backend="lite")

    assert scorer.score("sky", "sky is blue") == pytest.approx(0.25)
    assert scorer.score_batch([("earth", "sky is blue")]) == [0.75]
    assert scorer.score_batch_with_confidence([("sky", "sky is blue")]) == [(0.25, 1.0)]


def test_nli_invalid_backend_contract_is_rejected() -> None:
    with pytest.raises(TypeError, match="backend must be str or ScorerBackend"):
        NLIScorer(backend=object())
    with pytest.raises(ValueError, match="backend must be one of"):
        NLIScorer(backend="missing")


def test_nli_token_cost_counter_resets() -> None:
    scorer = NLIScorer(use_model=False, cost_per_token=0.5)
    scorer._last_token_count = 4

    assert scorer.last_token_count == 4
    assert scorer.last_estimated_cost == pytest.approx(2.0)
    scorer.reset_token_counter()
    assert scorer.last_token_count == 0


@pytest.mark.asyncio
async def test_nli_async_batch_uses_score_batch() -> None:
    scorer = NLIScorer(backend=StaticBackend())

    assert await scorer.ascore("sky", "sky is blue") == pytest.approx(0.2)
    assert await scorer.ascore_batch([("earth", "sky is blue")]) == [0.8]


def test_nli_minicheck_success_tuple_and_list_results() -> None:
    scorer = NLIScorer(backend="minicheck")
    scorer._minicheck_loaded = True
    scorer._minicheck = SimpleNamespace(
        score=lambda docs, claims: (
            ([1], [0.8], [], []) if len(docs) == 1 else [0.7, 0.2]
        )
    )

    assert scorer.score("source", "claim") == pytest.approx(0.2)
    assert scorer.score_batch([("a", "b"), ("c", "d")]) == pytest.approx([0.3, 0.8])
    with_conf = scorer.score_batch_with_confidence([("a", "b")])
    assert with_conf[0][0] == pytest.approx(0.2)
    assert with_conf[0][1] == pytest.approx(1.0)

    single_list = NLIScorer(backend="minicheck")
    single_list._minicheck_loaded = True
    single_list._minicheck = SimpleNamespace(score=lambda docs, claims: [0.65])
    assert single_list.score("source", "claim") == pytest.approx(0.35)


def test_nli_minicheck_disabled_and_failure_fall_back() -> None:
    disabled = NLIScorer(backend="minicheck", use_model=False)
    assert disabled.score("sky", "consistent with reality") == pytest.approx(0.1)
    assert disabled.score_batch([("sky", "opposite is true")]) == pytest.approx([0.9])

    failing = NLIScorer(backend="minicheck")
    failing._minicheck_loaded = True
    failing._minicheck = SimpleNamespace(
        score=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    assert failing.score("sky", "consistent with reality") == pytest.approx(0.1)
    assert failing._minicheck is None

    failing_batch = NLIScorer(backend="minicheck")
    failing_batch._minicheck_loaded = True
    failing_batch._minicheck = SimpleNamespace(
        score=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    assert failing_batch.score_batch([("sky", "consistent with reality")]) == [0.1]
    assert failing_batch._minicheck is None


def test_nli_minicheck_loader_import_failure(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "minicheck", raising=False)
    monkeypatch.delitem(sys.modules, "minicheck.minicheck", raising=False)
    scorer = NLIScorer(backend="minicheck")

    assert scorer._ensure_minicheck() is False


def test_nli_minicheck_loader_runtime_failure(monkeypatch) -> None:
    class FailingMiniCheck:
        def __init__(self, **_kwargs):
            raise AttributeError("bad minicheck")

    minicheck_module = ModuleType("minicheck")
    minicheck_module.MiniCheck = FailingMiniCheck
    monkeypatch.setitem(sys.modules, "minicheck", minicheck_module)
    scorer = NLIScorer(backend="minicheck")

    assert scorer._ensure_minicheck() is False
    assert scorer._minicheck is None


def test_nli_minicheck_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="minicheck_variant"):
        NLIScorer(backend="minicheck", minicheck_variant="unknown")


def test_nli_minicheck_score_falls_back_when_loader_unavailable(monkeypatch) -> None:
    scorer = NLIScorer(backend="minicheck")
    monkeypatch.setattr(scorer, "_ensure_minicheck", lambda: False)

    assert scorer.score("sky", "consistent with reality") == pytest.approx(0.1)
    assert scorer.score_batch([("sky", "consistent with reality")]) == [0.1]


def test_nli_model_score_single_pair_and_factcg(monkeypatch) -> None:
    _install_fake_torch(monkeypatch)
    scorer = NLIScorer(use_model=False, model_name="plain-nli")
    scorer._tokenizer = FakeTokenizer()
    scorer._model = FakeModel([[0.0, 1.0]])

    score = scorer._model_score("premise", "hypothesis")

    assert 0.0 <= score <= 1.0
    assert scorer.last_token_count == 3

    factcg = NLIScorer(use_model=False, model_name="derenlei/FactCG")
    factcg._tokenizer = FakeTokenizer()
    factcg._model = FakeModel([[0.0, 0.2, 0.8]])
    factcg._label_indices = (2, 1)

    factcg_score = factcg._model_score("source text", "claim text")

    probs = np.exp([0.0, 0.2, 0.8])
    probs = probs / probs.sum()
    assert factcg_score == pytest.approx(probs[2] + 0.5 * probs[1], abs=1e-6)
    assert "Choose your answer" in factcg._tokenizer.calls[0][0][0]


def test_nli_model_score_requires_loaded_model() -> None:
    scorer = NLIScorer(use_model=False)

    with pytest.raises(RuntimeError, match="NLI model not loaded"):
        scorer._model_score("premise", "hypothesis")
    with pytest.raises(RuntimeError, match="NLI model not loaded"):
        scorer._model_score_batch([("premise", "hypothesis")])
    with pytest.raises(RuntimeError, match="NLI model not loaded"):
        scorer._model_score_batch_with_confidence([("premise", "hypothesis")])


def test_nli_model_score_batch_and_confidence(monkeypatch) -> None:
    _install_fake_torch(monkeypatch)
    scorer = NLIScorer(use_model=False, model_name="plain-nli")
    scorer._tokenizer = FakeTokenizer()
    scorer._model = FakeModel([[2.0, 0.5, 0.0], [0.0, 1.0, 2.0]])
    scorer._label_indices = (2, 1)

    scores = scorer._model_score_batch([("p1", "h1"), ("p2", "h2")])
    with_conf = scorer._model_score_batch_with_confidence([("p1", "h1"), ("p2", "h2")])

    assert len(scores) == 2
    assert len(with_conf) == 2
    assert scorer.last_token_count == 12
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert all(0.0 <= confidence <= 1.0 for _, confidence in with_conf)


def test_nli_factcg_model_batch_templates(monkeypatch) -> None:
    _install_fake_torch(monkeypatch)
    scorer = NLIScorer(use_model=False, model_name="derenlei/FactCG")
    scorer._tokenizer = FakeTokenizer()
    scorer._model = FakeModel([[0.0, 0.2, 0.8], [0.1, 0.3, 0.6]])
    scorer._label_indices = (2, 1)

    scores = scorer._model_score_batch(
        [("source one", "claim one"), ("source two", "claim two")]
    )
    with_conf = scorer._model_score_batch_with_confidence(
        [("source one", "claim one"), ("source two", "claim two")]
    )

    assert len(scores) == 2
    assert len(with_conf) == 2
    texts = scorer._tokenizer.calls[0][0][0]
    assert all("Choose your answer" in text for text in texts)


def test_nli_public_score_routes_model_backed_paths(monkeypatch) -> None:
    _install_fake_torch(monkeypatch)
    scorer = NLIScorer(use_model=False, model_name="plain-nli")
    scorer._model_loaded = True
    scorer._tokenizer = FakeTokenizer()
    scorer._model = FakeModel([[0.0, 1.0], [2.0, 0.5]])

    assert 0.0 <= scorer.score("p", "h") <= 1.0
    assert len(scorer.score_batch([("p1", "h1"), ("p2", "h2")])) == 2


def test_nli_onnx_score_batch_and_confidence_casts_expected_inputs(monkeypatch) -> None:
    monkeypatch.setattr(nli_accel, "_RUST_NLI", False)
    scorer = NLIScorer(use_model=False, backend="onnx", model_name="plain-nli")
    scorer._tokenizer = FakeTokenizer()
    scorer._onnx_session = FakeOnnxSession([[2.0, 0.0], [0.0, 2.0]])

    scores = scorer._onnx_score_batch([("p1", "h1"), ("p2", "h2")])
    with_conf = scorer._onnx_score_batch_with_confidence([("p1", "h1"), ("p2", "h2")])

    assert len(scores) == 2
    assert len(with_conf) == 2
    assert scorer.last_token_count == 12
    assert set(scorer._onnx_session.feed_seen) == {"input_ids", "attention_mask"}
    assert scorer._onnx_session.feed_seen["input_ids"].dtype == np.int64


def test_nli_factcg_onnx_batch_templates(monkeypatch) -> None:
    monkeypatch.setattr(nli_accel, "_RUST_NLI", False)
    scorer = NLIScorer(use_model=False, backend="onnx", model_name="derenlei/FactCG")
    scorer._tokenizer = FakeTokenizer()
    scorer._onnx_session = FakeOnnxSession([[0.0, 0.2, 0.8], [0.1, 0.3, 0.6]])
    scorer._label_indices = (2, 1)

    scores = scorer._onnx_score_batch(
        [("source one", "claim one"), ("source two", "claim two")]
    )
    with_conf = scorer._onnx_score_batch_with_confidence(
        [("source one", "claim one"), ("source two", "claim two")]
    )

    assert len(scores) == 2
    assert len(with_conf) == 2
    texts = scorer._tokenizer.calls[0][0][0]
    assert all("Choose your answer" in text for text in texts)


def test_nli_onnx_score_requires_loaded_session() -> None:
    scorer = NLIScorer(use_model=False, backend="onnx")

    with pytest.raises(RuntimeError, match="ONNX session not loaded"):
        scorer._onnx_score_batch([("p", "h")])
    with pytest.raises(RuntimeError, match="ONNX session not loaded"):
        scorer._onnx_score_batch_with_confidence([("p", "h")])


def test_nli_score_routes_onnx_and_empty_batches(monkeypatch) -> None:
    scorer = NLIScorer(use_model=False, backend="onnx")
    scorer._model_loaded = True
    scorer._onnx_session = object()
    monkeypatch.setattr(
        scorer, "_onnx_score_batch", lambda pairs: [0.33 for _ in pairs]
    )

    assert scorer.score("p", "h") == pytest.approx(0.33)
    assert scorer.score_batch([("p", "h")]) == [0.33]
    assert scorer.score_batch([]) == []
    assert scorer.score_batch_with_confidence([]) == []


def test_nli_score_batch_with_confidence_routes_onnx(monkeypatch) -> None:
    scorer = NLIScorer(use_model=False, backend="onnx")
    scorer._model_loaded = True
    scorer._onnx_session = object()
    monkeypatch.setattr(
        scorer,
        "_onnx_score_batch_with_confidence",
        lambda pairs: [(0.44, 0.55) for _ in pairs],
    )

    assert scorer.score_batch_with_confidence([("p", "h")]) == [(0.44, 0.55)]


def test_nli_score_batch_with_confidence_routes_model(monkeypatch) -> None:
    scorer = NLIScorer(use_model=False, model_name="plain-nli")
    scorer._model_loaded = True
    scorer._tokenizer = FakeTokenizer()
    scorer._model = FakeModel([[0.0, 1.0]])
    monkeypatch.setattr(
        scorer, "_model_score_batch_with_confidence", lambda pairs: [(0.12, 0.98)]
    )

    assert scorer.score_batch_with_confidence([("p", "h")]) == [(0.12, 0.98)]


def test_nli_score_batch_with_confidence_heuristic_path() -> None:
    scorer = NLIScorer(use_model=False)

    assert scorer.score_batch_with_confidence([("p", "consistent with reality")]) == [
        (0.1, 0.5)
    ]


def test_nli_python_chunking_and_claim_paths(monkeypatch) -> None:
    monkeypatch.setattr(nli_accel, "_RUST_NLI", False)
    scorer = NLIScorer(use_model=False, max_length=40)
    long_source = ". ".join(f"Source sentence {idx} supports sky" for idx in range(8))
    long_summary = ". ".join(f"Claim sentence {idx} supports sky" for idx in range(6))

    chunks = scorer._build_chunks(["verylongsentencewithoutspaces"], budget=1)
    overlap = scorer._build_chunks_overlap(
        ["alpha", "beta", "gamma"], budget=100, overlap_ratio=0.5
    )
    agg, per_hyp, n_prem, n_hyp = scorer._score_chunked_with_counts(
        long_source,
        long_summary,
        inner_agg="mean",
        outer_agg="trimmed_mean",
        overlap_ratio=0.5,
    )
    coverage, divs, claims = scorer.score_claim_coverage(
        long_source,
        long_summary,
        support_threshold=0.6,
    )

    assert chunks == ["verylongsentencewithoutspaces"]
    assert overlap
    assert n_prem >= 1
    assert n_hyp >= 1
    assert len(per_hyp) == n_hyp
    assert 0.0 <= agg <= 1.0
    assert len(divs) == len(claims)
    assert 0.0 <= coverage <= 1.0


def test_nli_python_chunking_empty_and_max_aggregation(monkeypatch) -> None:
    monkeypatch.setattr(nli_accel, "_RUST_NLI", False)
    scorer = NLIScorer(use_model=False, max_length=40)
    assert scorer._build_chunks([], budget=10) == [""]

    long_source = ". ".join(f"Source sentence {idx} supports sky" for idx in range(8))
    long_summary = ". ".join(f"Claim sentence {idx} supports sky" for idx in range(6))
    agg, per_hyp, _, _ = scorer._score_chunked_with_counts(
        long_source,
        long_summary,
        inner_agg="max",
        outer_agg="max",
    )

    assert agg == max(per_hyp)


def test_nli_chunked_short_text_scores_once_and_records_single_chunks(
    monkeypatch,
) -> None:
    monkeypatch.setattr(nli_accel, "_RUST_NLI", False)
    observed: list[tuple[str, int]] = []
    scorer = NLIScorer(use_model=False, max_length=512)
    monkeypatch.setattr(scorer, "score", lambda _premise, _hypothesis: 0.42)
    monkeypatch.setattr(
        metrics,
        "observe",
        lambda name, value: observed.append((name, value)),
    )

    result = scorer._score_chunked_with_counts("short source", "short claim")

    assert result == (0.42, [0.42], 1, 1)
    assert observed == [("nli_premise_chunks", 1), ("nli_hypothesis_chunks", 1)]


def test_nli_python_overlap_handles_single_sentence_over_budget(monkeypatch) -> None:
    monkeypatch.setattr(nli_accel, "_RUST_NLI", False)
    scorer = NLIScorer(use_model=False)

    assert scorer._build_chunks_overlap(["x" * 200], budget=1, overlap_ratio=0.5) == [
        "x" * 200
    ]


def test_nli_rust_overlap_chunk_adapter(monkeypatch) -> None:
    monkeypatch.setattr(nli_accel, "_RUST_NLI", True)
    monkeypatch.setattr(
        nli_accel,
        "rust_build_chunks",
        lambda sentences, budget, overlap_ratio: [
            f"{budget}:{overlap_ratio}:{'|'.join(sentences)}"
        ],
    )
    scorer = NLIScorer(use_model=False)

    assert scorer._build_chunks_overlap(["a", "b"], 12, 0.5) == ["12:0.5:a|b"]


def test_nli_rust_default_chunk_adapter(monkeypatch) -> None:
    monkeypatch.setattr(nli_accel, "_RUST_NLI", True)
    monkeypatch.setattr(
        nli_accel,
        "rust_build_chunks",
        lambda sentences, budget, overlap_ratio: [
            f"{budget}:{overlap_ratio}:{'|'.join(sentences)}"
        ],
    )
    scorer = NLIScorer(use_model=False)

    assert scorer._build_chunks(["a", "b"], 12, 0.0) == ["12:0.0:a|b"]


def test_nli_rust_chunked_score_aggregation_adapter(monkeypatch) -> None:
    monkeypatch.setattr(nli_accel, "_RUST_NLI", True)
    observed: list[tuple[str, int]] = []
    scorer = NLIScorer(use_model=False, max_length=40)
    monkeypatch.setattr(
        scorer,
        "score_batch",
        lambda pairs: [0.1 + 0.01 * (idx % 50) for idx, _ in enumerate(pairs)],
    )
    monkeypatch.setattr(
        nli_accel,
        "rust_aggregate_chunk_scores",
        lambda scores, n_prem, n_hyp, inner_agg, outer_agg: (
            sum(scores) / len(scores),
            [max(scores[index::n_hyp]) for index in range(n_hyp)],
        ),
    )
    monkeypatch.setattr(
        metrics,
        "observe",
        lambda name, value: observed.append((name, value)),
    )
    source = ". ".join(f"Source sentence {idx} with details" for idx in range(10))
    summary = ". ".join(f"Claim sentence {idx} with details" for idx in range(8))

    agg, per_hyp, n_prem, n_hyp = scorer._score_chunked_with_counts(
        source,
        summary,
        inner_agg="mean",
        outer_agg="trimmed_mean",
        overlap_ratio=0.5,
    )

    assert 0.0 <= agg <= 1.0
    assert len(per_hyp) == n_hyp
    assert n_prem >= 1
    assert observed == [
        ("nli_premise_chunks", n_prem),
        ("nli_hypothesis_chunks", n_hyp),
    ]


def test_nli_confidence_weighted_python_aggregation(monkeypatch) -> None:
    monkeypatch.setattr(nli_accel, "_RUST_NLI", False)
    scorer = NLIScorer(use_model=False, max_length=40)
    monkeypatch.setattr(
        scorer,
        "score_batch_with_confidence",
        lambda pairs: [
            (0.2 + 0.1 * idx, 1.0 if idx % 2 == 0 else 0.5)
            for idx, _ in enumerate(pairs)
        ],
    )
    source = ". ".join(f"Source sentence {idx} with details" for idx in range(10))
    summary = ". ".join(f"Claim sentence {idx} with details" for idx in range(8))

    agg, per_hyp = scorer.score_chunked_confidence_weighted(
        source,
        summary,
        inner_agg="min",
        overlap_ratio=0.5,
    )

    assert per_hyp
    assert min(per_hyp) <= agg <= max(per_hyp)


def test_nli_confidence_weighted_short_text_uses_single_pair(monkeypatch) -> None:
    monkeypatch.setattr(nli_accel, "_RUST_NLI", False)
    scorer = NLIScorer(use_model=False, max_length=512)
    monkeypatch.setattr(
        scorer,
        "score_batch_with_confidence",
        lambda pairs: [(0.23, 0.9)],
    )

    assert scorer.score_chunked_confidence_weighted("short source", "short claim") == (
        0.23,
        [0.23],
    )


def test_nli_confidence_weighted_default_inner_max(monkeypatch) -> None:
    monkeypatch.setattr(nli_accel, "_RUST_NLI", False)
    scorer = NLIScorer(use_model=False, max_length=40)
    monkeypatch.setattr(
        scorer,
        "score_batch_with_confidence",
        lambda pairs: [(0.1 + 0.05 * idx, 1.0) for idx, _ in enumerate(pairs)],
    )
    source = ". ".join(f"Source sentence {idx} with details" for idx in range(10))
    summary = ". ".join(f"Claim sentence {idx} with details" for idx in range(8))

    agg, per_hyp = scorer.score_chunked_confidence_weighted(source, summary)

    assert per_hyp
    assert min(per_hyp) <= agg <= max(per_hyp)


def test_nli_confidence_weighted_zero_confidence_falls_back_to_mean(
    monkeypatch,
) -> None:
    monkeypatch.setattr(nli_accel, "_RUST_NLI", False)
    scorer = NLIScorer(use_model=False, max_length=40)
    monkeypatch.setattr(
        scorer,
        "score_batch_with_confidence",
        lambda pairs: [(0.2 + 0.1 * idx, 0.0) for idx, _ in enumerate(pairs)],
    )
    source = ". ".join(f"Source sentence {idx} with details" for idx in range(10))
    summary = ". ".join(f"Claim sentence {idx} with details" for idx in range(8))

    agg, per_hyp = scorer.score_chunked_confidence_weighted(
        source,
        summary,
        inner_agg="mean",
    )

    assert agg == pytest.approx(sum(per_hyp) / len(per_hyp))


def test_nli_rust_confidence_weighted_aggregation_adapter(monkeypatch) -> None:
    monkeypatch.setattr(nli_accel, "_RUST_NLI", True)
    scorer = NLIScorer(use_model=False, max_length=40)
    monkeypatch.setattr(
        scorer,
        "score_batch_with_confidence",
        lambda pairs: [
            (0.1 + 0.01 * (idx % 50), 0.5 + 0.005 * (idx % 50))
            for idx, _ in enumerate(pairs)
        ],
    )
    monkeypatch.setattr(
        nli_accel,
        "rust_aggregate_chunk_scores_confidence_weighted",
        lambda scores, conf, n_prem, n_hyp, inner_agg: (
            sum(score * weight for score, weight in zip(scores, conf, strict=True))
            / sum(conf),
            [max(scores[index::n_hyp]) for index in range(n_hyp)],
        ),
    )
    source = ". ".join(f"Source sentence {idx} with details" for idx in range(10))
    summary = ". ".join(f"Claim sentence {idx} with details" for idx in range(8))

    agg, per_hyp = scorer.score_chunked_confidence_weighted(
        source,
        summary,
        inner_agg="mean",
        overlap_ratio=0.5,
    )

    assert 0.0 <= agg <= 1.0
    assert per_hyp


def test_nli_claim_attribution_python_fallback_and_limits(monkeypatch) -> None:
    monkeypatch.setattr(nli_accel, "_RUST_NLI", False)
    scorer = NLIScorer(use_model=False)
    source = "Sky is blue. Grass is green."
    summary = "Sky is blue. Grass is green."

    coverage, divs, claims, attributions = scorer.score_claim_coverage_with_attribution(
        source,
        summary,
        support_threshold=0.6,
    )

    assert len(attributions) == len(claims)
    assert len(divs) == len(claims)
    assert 0.0 <= coverage <= 1.0
    assert {attr.source_sentence for attr in attributions} <= {
        "Sky is blue.",
        "Grass is green.",
    }

    huge_summary = ". ".join(f"Claim {idx}" for idx in range(101))
    huge_source = ". ".join(f"Source {idx}" for idx in range(100))
    with pytest.raises(ValueError, match="exceeding limit"):
        scorer.score_claim_coverage_with_attribution(huge_source, huge_summary)


def test_nli_claim_attribution_uses_source_when_sentence_split_empty(
    monkeypatch,
) -> None:
    monkeypatch.setattr(nli_accel, "_RUST_NLI", False)
    scorer = NLIScorer(use_model=False)
    monkeypatch.setattr(
        scorer,
        "decompose_claims",
        lambda text: ["claim"] if text == "summary" else [],
    )

    _coverage, _divs, _claims, attributions = (
        scorer.score_claim_coverage_with_attribution(
            "",
            "summary",
        )
    )

    assert attributions[0].source_sentence == ""


def test_nli_claim_empty_paths(monkeypatch) -> None:
    monkeypatch.setattr(nli_accel, "_RUST_NLI", False)
    scorer = NLIScorer(use_model=False)
    monkeypatch.setattr(scorer, "decompose_claims", lambda _text: [])

    coverage, divs, claims = scorer.score_claim_coverage(
        "source",
        "consistent with reality",
    )
    coverage_attr, divs_attr, claims_attr, attributions = (
        scorer.score_claim_coverage_with_attribution("", "consistent with reality")
    )
    decomposed_score, decomposed = scorer.score_decomposed(
        "source",
        "consistent with reality",
    )

    assert coverage == 1.0
    assert divs == [0.1]
    assert claims == ["consistent with reality"]
    assert coverage_attr == 1.0
    assert divs_attr == [0.1]
    assert claims_attr == ["consistent with reality"]
    assert attributions[0].source_sentence == ""
    assert decomposed_score == pytest.approx(0.1)
    assert decomposed == [0.1]


def test_nli_score_decomposed_single_and_multi_claims(monkeypatch) -> None:
    monkeypatch.setattr(nli_accel, "_RUST_NLI", False)
    scorer = NLIScorer(use_model=False)
    monkeypatch.setattr(
        scorer,
        "decompose_claims",
        lambda text: ["claim"] if text == "one" else ["a", "b"],
    )
    monkeypatch.setattr(
        scorer,
        "score",
        lambda _premise, hypothesis: 0.2 if hypothesis == "claim" else 0.4,
    )
    monkeypatch.setattr(scorer, "score_batch", lambda pairs: [0.3, 0.7])

    assert scorer.score_decomposed("source", "one") == (0.2, [0.2])
    assert scorer.score_decomposed("source", "many") == (0.7, [0.3, 0.7])


def test_nli_claim_coverage_uses_rust_reducer(monkeypatch) -> None:
    monkeypatch.setattr(nli_accel, "_RUST_NLI", True)
    monkeypatch.setattr(
        nli_accel, "rust_coverage_from_divergences", lambda divs, threshold: (0.75, 3)
    )
    scorer = NLIScorer(use_model=False)
    monkeypatch.setattr(scorer, "decompose_claims", lambda _summary: ["a", "b"])
    monkeypatch.setattr(scorer, "score_chunked", lambda *_args, **_kwargs: (0.2, [0.2]))

    coverage, divs, claims = scorer.score_claim_coverage("source", "summary")

    assert coverage == pytest.approx(0.75)
    assert divs == [0.2, 0.2]
    assert claims == ["a", "b"]


def test_nli_heuristic_contract_edges() -> None:
    assert NLIScorer._heuristic_score("", "anything") == pytest.approx(0.5)
    assert NLIScorer._heuristic_score("sky is blue", "sky is not blue") >= 0.7
    assert NLIScorer._heuristic_score("premise", "depends on your perspective") == 0.5


def test_tokenize_serialises_concurrent_calls() -> None:
    """``_tokenize`` must hold a lock around the shared tokenizer.

    The fast (Rust) tokenizer raises ``RuntimeError("Already borrowed")`` when
    two threads encode at once — which happens because the logical and factual
    divergence futures run in parallel on the same scorer. The lock must keep
    at most one thread inside the tokenizer at any instant.
    """
    scorer = NLIScorer(use_model=False)
    state = {"active": 0, "max_active": 0}
    probe_guard = threading.Lock()

    def probe(*_args, **_kwargs):
        with probe_guard:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        time.sleep(0.002)
        with probe_guard:
            state["active"] -= 1
        return {"input_ids": FakeTensor(np.ones((1, 3), dtype=np.int64))}

    scorer._tokenizer = probe

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(scorer._tokenize, "premise", "hypothesis") for _ in range(64)
        ]
        for future in futures:
            future.result()

    # Without the lock, up to 8 threads sit inside ``probe`` at once; the lock
    # forces strict serialisation so the observed peak is exactly one.
    assert state["max_active"] == 1


def test_tokenize_raises_when_tokenizer_missing() -> None:
    scorer = NLIScorer(use_model=False)
    scorer._tokenizer = None
    with pytest.raises(RuntimeError, match="NLI model not loaded"):
        scorer._tokenize("premise", "hypothesis")


class _IndexedChunkModel(FakeModel):
    """Size-aware fake: one logit row per input row, forward sizes recorded.

    Row logits derive from a running counter so every pair scores
    differently — concatenation order across chunks becomes observable.
    """

    def __init__(self) -> None:
        super().__init__([[0.0, 0.0]])
        self.forward_sizes: list[int] = []
        self._next = 0

    def __call__(self, **inputs):
        rows = int(inputs["input_ids"].values.shape[0])
        self.forward_sizes.append(rows)
        logits = [[0.0, 0.1 * (self._next + row)] for row in range(rows)]
        self._next += rows
        return SimpleNamespace(logits=FakeTensor(np.asarray(logits)))


class _IndexedChunkOnnxSession(FakeOnnxSession):
    """Size-aware fake ONNX session recording per-run batch sizes."""

    def __init__(self) -> None:
        super().__init__([[0.0, 0.0]])
        self.run_sizes: list[int] = []
        self._next = 0

    def run(self, _outputs, feed):
        self.feed_seen = feed
        rows = int(feed["input_ids"].shape[0])
        self.run_sizes.append(rows)
        logits = [[0.0, 0.1 * (self._next + row)] for row in range(rows)]
        self._next += rows
        return [np.asarray(logits, dtype=np.float64)]


def _chunking_pairs(count: int) -> list[tuple[str, str]]:
    return [(f"premise {i}", f"hypothesis {i}") for i in range(count)]


def test_nli_model_batch_chunks_bound_forward_size_and_preserve_scores(
    monkeypatch,
) -> None:
    """Chunked forwards stay bounded and concatenate to the unchunked scores.

    An unbounded claims×chunks forward OOMed a 24 GB A30 on a long RAGTruth
    sample (2026-07-18); the chunk loop must bound every forward while
    keeping the resulting score list identical.
    """
    import director_ai.core.scoring._nli_model_inference as nli_inference

    _install_fake_torch(monkeypatch)

    def build_scorer() -> tuple[NLIScorer, _IndexedChunkModel]:
        scorer = NLIScorer(use_model=False, model_name="plain-nli")
        scorer._tokenizer = FakeTokenizer()
        model = _IndexedChunkModel()
        scorer._model = model
        return scorer, model

    monkeypatch.setattr(nli_inference, "_MAX_PAIRS_PER_FORWARD", 2)
    chunked_scorer, chunked_model = build_scorer()
    chunked = chunked_scorer._model_score_batch(_chunking_pairs(5))
    chunked_conf = chunked_scorer._model_score_batch_with_confidence(_chunking_pairs(5))

    assert chunked_model.forward_sizes == [2, 2, 1, 2, 2, 1]
    assert len(chunked) == 5
    assert len(chunked_conf) == 5

    monkeypatch.setattr(nli_inference, "_MAX_PAIRS_PER_FORWARD", 1000)
    single_scorer, single_model = build_scorer()
    single = single_scorer._model_score_batch(_chunking_pairs(5))
    single_conf = single_scorer._model_score_batch_with_confidence(_chunking_pairs(5))

    assert single_model.forward_sizes == [5, 5]
    assert chunked == single
    assert chunked_conf == single_conf


def test_nli_onnx_batch_chunks_bound_run_size_and_preserve_scores(
    monkeypatch,
) -> None:
    """The ONNX chunk loop mirrors the PyTorch bound with identical scores."""
    import director_ai.core.scoring._nli_model_inference as nli_inference

    monkeypatch.setattr(nli_accel, "_RUST_NLI", False)

    def build_scorer() -> tuple[NLIScorer, _IndexedChunkOnnxSession]:
        scorer = NLIScorer(use_model=False, backend="onnx", model_name="plain-nli")
        scorer._tokenizer = FakeTokenizer()
        session = _IndexedChunkOnnxSession()
        scorer._onnx_session = session
        return scorer, session

    monkeypatch.setattr(nli_inference, "_MAX_PAIRS_PER_FORWARD", 2)
    chunked_scorer, chunked_session = build_scorer()
    chunked = chunked_scorer._onnx_score_batch(_chunking_pairs(5))
    chunked_conf = chunked_scorer._onnx_score_batch_with_confidence(_chunking_pairs(5))

    assert chunked_session.run_sizes == [2, 2, 1, 2, 2, 1]
    assert len(chunked) == 5
    assert len(chunked_conf) == 5

    monkeypatch.setattr(nli_inference, "_MAX_PAIRS_PER_FORWARD", 1000)
    single_scorer, single_session = build_scorer()
    single = single_scorer._onnx_score_batch(_chunking_pairs(5))
    single_conf = single_scorer._onnx_score_batch_with_confidence(_chunking_pairs(5))

    assert single_session.run_sizes == [5, 5]
    assert chunked == single
    assert chunked_conf == single_conf


def test_nli_batch_empty_pairs_short_circuit(monkeypatch) -> None:
    """An empty batch returns [] without touching the model or session."""
    _install_fake_torch(monkeypatch)
    monkeypatch.setattr(nli_accel, "_RUST_NLI", False)

    scorer = NLIScorer(use_model=False, model_name="plain-nli")
    scorer._tokenizer = FakeTokenizer()
    model = _IndexedChunkModel()
    scorer._model = model

    assert scorer._model_score_batch([]) == []
    assert scorer._model_score_batch_with_confidence([]) == []
    assert model.forward_sizes == []

    onnx_scorer = NLIScorer(use_model=False, backend="onnx", model_name="plain-nli")
    onnx_scorer._tokenizer = FakeTokenizer()
    session = _IndexedChunkOnnxSession()
    onnx_scorer._onnx_session = session

    assert onnx_scorer._onnx_score_batch([]) == []
    assert onnx_scorer._onnx_score_batch_with_confidence([]) == []
    assert session.run_sizes == []
