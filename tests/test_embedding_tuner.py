# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for embedding tuner training-pair construction and compat imports."""

from __future__ import annotations

import contextlib
import importlib
import math
import sys
import types

import pytest

from director_ai.core.retrieval import embedding_tuner
from director_ai.core.retrieval.embedding_tuner import TuneResult, tune_embeddings


def test_embedding_tuner_requires_training_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    with pytest.raises(ImportError, match=r"director-ai\[embeddings\]"):
        tune_embeddings([["alpha", "beta"], ["gamma", "delta"]])


def test_embedding_tuner_requires_training_pairs(monkeypatch):
    _install_fake_training_stack(monkeypatch)

    with pytest.raises(ValueError, match="Need at least 2 documents"):
        tune_embeddings([["single chunk"]])


def test_embedding_tuner_builds_contrastive_pairs_and_saves_model(
    monkeypatch,
    tmp_path,
):
    recorder = _install_fake_training_stack(monkeypatch)
    documents = [
        ["invoice header", "invoice line", "payment terms", "audit trail"],
        ["support greeting", "refund policy"],
    ]
    output_dir = tmp_path / "tenant-embeddings"

    result = tune_embeddings(
        documents,
        base_model="local/base-embedder",
        output_dir=str(output_dir),
        epochs=5,
        batch_size=3,
        seed=7,
    )

    # The fake loss reports the mean label per batch, so the measured mean loss
    # over 4 positive (1.0) + 5 negative (0.0) pairs is 4/9. fit() is a no-op in
    # the fake, so start and end match — but both are now measured, not 0.0.
    assert result.model_path == str(output_dir)
    assert result.train_samples == 9
    assert result.epochs == 5
    assert result.loss_start == pytest.approx(4 / 9)
    assert result.loss_end == pytest.approx(4 / 9)
    assert isinstance(result, TuneResult)
    assert output_dir.is_dir()
    assert recorder["model"].base_model == "local/base-embedder"
    assert recorder["model"].saved_paths == [str(output_dir)]

    examples = recorder["examples"]
    positive_pairs = [example for example in examples if example.label == 1.0]
    negative_pairs = [example for example in examples if example.label == 0.0]
    assert [example.texts for example in positive_pairs] == [
        ["invoice header", "invoice line"],
        ["invoice line", "payment terms"],
        ["payment terms", "audit trail"],
        ["support greeting", "refund policy"],
    ]
    assert len(negative_pairs) == 5
    assert all(
        left != right for left, right in (example.texts for example in negative_pairs)
    )

    loader = recorder["fit"]["train_objectives"][0][0]
    assert [example.texts for example in loader.dataset] == [
        example.texts for example in examples
    ]
    assert loader.shuffle is True
    assert loader.batch_size == 3
    assert len(loader) == math.ceil(len(examples) / 3)
    assert recorder["fit"]["epochs"] == 5
    assert recorder["fit"]["warmup_steps"] == 1
    assert recorder["fit"]["show_progress_bar"] is True
    assert recorder["loss"].model is recorder["model"]


def test_legacy_embedding_tuner_import_exposes_retrieval_module(monkeypatch):
    monkeypatch.delitem(sys.modules, "director_ai.core.embedding_tuner", raising=False)

    legacy = importlib.import_module("director_ai.core.embedding_tuner")

    assert legacy is embedding_tuner
    assert legacy.tune_embeddings is tune_embeddings
    assert legacy.TuneResult is TuneResult


def test_legacy_heuristics_import_exposes_shared_patterns(monkeypatch):
    from director_ai.core.scoring import _heuristics as real_heuristics

    monkeypatch.delitem(sys.modules, "director_ai.core._heuristics", raising=False)

    legacy = importlib.import_module("director_ai.core._heuristics")

    assert legacy is real_heuristics
    assert "not" in legacy.NEGATION_WORDS
    assert legacy.WORD_RE.findall("Alpha beta 42") == ["Alpha", "beta", "42"]
    assert legacy.ENTITY_RE.findall("Ada Lovelace met alan turing") == ["Ada Lovelace"]


def _install_fake_training_stack(monkeypatch):
    recorder: dict[str, object] = {"examples": []}

    class FakeInputExample:
        def __init__(self, *, texts, label):
            self.texts = list(texts)
            self.label = label
            recorder["examples"].append(self)

    class FakeSentenceTransformer:
        def __init__(self, base_model):
            self.base_model = base_model
            self.saved_paths: list[str] = []
            recorder["model"] = self

        def fit(self, **kwargs):
            recorder["fit"] = kwargs

        def save(self, output_dir):
            self.saved_paths.append(output_dir)

        def smart_batching_collate(self, batch):
            # Mirror sentence-transformers' (features, labels) contract.
            return ({}, [example.label for example in batch])

    class _FakeLossValue:
        def __init__(self, value):
            self._value = value

        def item(self):
            return self._value

    class FakeCosineSimilarityLoss:
        def __init__(self, model):
            self.model = model
            recorder["loss"] = self

        def __call__(self, _features, labels):
            mean = sum(labels) / len(labels) if labels else 0.0
            return _FakeLossValue(mean)

    class FakeDataLoader:
        def __init__(self, dataset, *, shuffle=False, batch_size=1, collate_fn=None):
            self.dataset = list(dataset)
            self.shuffle = shuffle
            self.batch_size = batch_size
            self.collate_fn = collate_fn

        def __len__(self):
            return math.ceil(len(self.dataset) / self.batch_size)

        def __iter__(self):
            for start in range(0, len(self.dataset), self.batch_size):
                batch = self.dataset[start : start + self.batch_size]
                yield self.collate_fn(batch) if self.collate_fn else batch

    sentence_transformers = types.ModuleType("sentence_transformers")
    sentence_transformers.InputExample = FakeInputExample
    sentence_transformers.SentenceTransformer = FakeSentenceTransformer
    sentence_transformers.losses = types.SimpleNamespace(
        CosineSimilarityLoss=FakeCosineSimilarityLoss
    )
    sentence_transformers_losses = types.ModuleType("sentence_transformers.losses")
    sentence_transformers_losses.CosineSimilarityLoss = FakeCosineSimilarityLoss
    torch = types.ModuleType("torch")
    torch.no_grad = contextlib.nullcontext
    torch_utils = types.ModuleType("torch.utils")
    torch_utils_data = types.ModuleType("torch.utils.data")
    torch_utils_data.DataLoader = FakeDataLoader
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)
    monkeypatch.setitem(
        sys.modules, "sentence_transformers.losses", sentence_transformers_losses
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", torch_utils_data)
    return recorder
