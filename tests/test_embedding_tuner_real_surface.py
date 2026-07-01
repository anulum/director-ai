# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real-surface coverage for embedding tuner training orchestration."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import cast

import pytest

from director_ai.core.retrieval.embedding_tuner import (
    EmbeddingTrainingStack,
    TuneResult,
    tune_embeddings,
)
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


class _InputExample:
    """Local training example matching sentence-transformers' public shape."""

    def __init__(self, *, texts: Sequence[str], label: float) -> None:
        self.texts = list(texts)
        self.label = float(label)


class _LocalSentenceTransformer:
    """Local trainable model surface used by the production tuner flow."""

    def __init__(self, base_model: str) -> None:
        self.base_model = base_model
        self.fit_calls: list[dict[str, object]] = []
        self.saved_paths: list[str] = []

    def smart_batching_collate(
        self,
        batch: Sequence[_InputExample],
    ) -> tuple[dict[str, object], list[float]]:
        """Return the features/labels shape consumed by the local loss."""
        return {"batch_size": len(batch)}, [example.label for example in batch]

    def fit(self, **kwargs: object) -> None:
        """Record the real fit invocation issued by the tuner."""
        self.fit_calls.append(dict(kwargs))

    def save(self, output_dir: str) -> None:
        """Record the output directory chosen by the tuner."""
        self.saved_paths.append(output_dir)


class _LossValue:
    """Scalar loss value with the PyTorch-style item method."""

    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        """Return the scalar loss value."""
        return self._value


class _CosineSimilarityLoss:
    """Local loss object matching sentence-transformers' loss call shape."""

    def __init__(self, model: _LocalSentenceTransformer) -> None:
        self.model = model

    def __call__(self, features: object, labels: Sequence[float]) -> _LossValue:
        """Return the mean label as a deterministic observable loss."""
        del features
        mean = sum(labels) / len(labels) if labels else 0.0
        return _LossValue(mean)


class _DataLoader:
    """Small iterable with the DataLoader contract used by the tuner."""

    def __init__(
        self,
        dataset: Sequence[_InputExample],
        *,
        shuffle: bool = False,
        batch_size: int = 1,
        collate_fn: object | None = None,
    ) -> None:
        self.dataset = list(dataset)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.collate_fn = collate_fn

    def __len__(self) -> int:
        """Return the number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[object]:
        """Yield batches, applying the optional collate function."""
        for start in range(0, len(self.dataset), self.batch_size):
            batch = self.dataset[start : start + self.batch_size]
            if callable(self.collate_fn):
                yield self.collate_fn(batch)
            else:
                yield batch


def _training_stack(models: list[_LocalSentenceTransformer]) -> EmbeddingTrainingStack:
    """Return a local stack that exercises tuner orchestration."""

    def _sentence_transformer(base_model: str) -> _LocalSentenceTransformer:
        model = _LocalSentenceTransformer(base_model)
        models.append(model)
        return model

    return EmbeddingTrainingStack(
        input_example=_InputExample,
        sentence_transformer=_sentence_transformer,
        data_loader=_DataLoader,
        cosine_similarity_loss=_CosineSimilarityLoss,
        no_grad=nullcontext,
    )


def test_embedding_tuner_unit_guard_declares_this_companion() -> None:
    """The embedding tuner guard must point at this production-surface companion."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_embedding_tuner.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_embedding_tuner_real_surface.py" in reason


def test_tune_embeddings_runs_real_orchestration_with_injected_training_stack(
    tmp_path: Path,
) -> None:
    """Embedding tuning should build pairs, fit, measure loss, and save."""
    models: list[_LocalSentenceTransformer] = []
    documents = [
        ["invoice header", "invoice line", "payment terms"],
        ["support greeting", "refund policy", "refund escalation"],
    ]
    output_dir = tmp_path / "tenant-embedding-model"

    result = tune_embeddings(
        documents,
        base_model="local/base",
        output_dir=str(output_dir),
        epochs=4,
        batch_size=2,
        seed=11,
        training_stack=_training_stack(models),
    )

    model = models[0]
    train_objectives = cast(
        list[tuple[_DataLoader, _CosineSimilarityLoss]],
        model.fit_calls[0]["train_objectives"],
    )
    train_loader = train_objectives[0][0]
    train_examples = train_loader.dataset
    positive_pairs = [
        example.texts for example in train_examples if example.label == 1.0
    ]
    negative_pairs = [
        example.texts for example in train_examples if example.label == 0.0
    ]

    assert isinstance(result, TuneResult)
    assert result.model_path == str(output_dir)
    assert result.train_samples == 10
    assert result.epochs == 4
    assert result.loss_start == pytest.approx(0.4)
    assert result.loss_end == pytest.approx(0.4)
    assert output_dir.is_dir()
    assert model.base_model == "local/base"
    assert model.saved_paths == [str(output_dir)]
    assert model.fit_calls[0]["epochs"] == 4
    assert model.fit_calls[0]["warmup_steps"] == 1
    assert model.fit_calls[0]["show_progress_bar"] is True
    assert positive_pairs == [
        ["invoice header", "invoice line"],
        ["invoice line", "payment terms"],
        ["support greeting", "refund policy"],
        ["refund policy", "refund escalation"],
    ]
    assert len(negative_pairs) == 6
    assert all(left != right for left, right in negative_pairs)
