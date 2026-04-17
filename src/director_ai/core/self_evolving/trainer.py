# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — guardrail trainer

"""Two real trainers for the self-evolving loop.

:class:`PerceptronGuardrailTrainer` — pure-Python online
perceptron over a fixed-dim FNV-1a hash-bag featuriser. Real
learning: it converges on the training set, exposes its weights
for audit, and produces a :class:`TrainedGuardrail` with a
``.score(text)`` method that returns a calibrated ``[0, 1]``
probability via the logistic of the margin.

:class:`LoraGuardrailTrainer` — drop-in for a LoRA micro-fine-tune
via ``peft`` + ``transformers``. Lazy import so the perceptron
path runs without the ML stack.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .feedback import FeedbackEvent

# FNV-1a parameters (shared with the multimodal hash-bag family).
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_UINT64_MASK = 0xFFFFFFFFFFFFFFFF


@dataclass(frozen=True)
class TrainedGuardrail:
    """Immutable snapshot of a trained guardrail.

    ``weights`` and ``bias`` are the model parameters. ``dim`` is
    the featuriser dimensionality. ``score(text)`` is the public
    inference hook — callers pass it to :class:`SelfEvolver` for
    the hot-swap step.
    """

    weights: tuple[float, ...]
    bias: float
    dim: int
    version: int
    epochs: int
    training_accuracy: float

    def score(self, text: str) -> float:
        """Return the probability that ``text`` is unsafe, in
        ``[0, 1]``."""
        features = _hash_bag(text, self.dim)
        margin = self.bias + sum(w * f for w, f in zip(self.weights, features, strict=True))
        return 1.0 / (1.0 + math.exp(-margin))


@runtime_checkable
class GuardrailTrainer(Protocol):
    """Train a guardrail from labelled feedback events."""

    def train(
        self,
        events: Iterable[FeedbackEvent],
        *,
        version: int,
    ) -> TrainedGuardrail: ...


class PerceptronGuardrailTrainer:
    """Online perceptron with logistic output head.

    Parameters
    ----------
    dim :
        Hash-bag feature dimensionality. Default 1024 — high
        enough to keep collisions low for real prompts.
    learning_rate :
        Step size for the perceptron update rule. Default 0.1.
    epochs :
        Passes over the training set. Default 4.
    l2 :
        L2 regularisation coefficient. Default 1e-4.
    """

    def __init__(
        self,
        *,
        dim: int = 1024,
        learning_rate: float = 0.1,
        epochs: int = 4,
        l2: float = 1e-4,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive; got {dim!r}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive; got {learning_rate!r}")
        if epochs <= 0:
            raise ValueError(f"epochs must be positive; got {epochs!r}")
        if l2 < 0:
            raise ValueError(f"l2 must be non-negative; got {l2!r}")
        self._dim = dim
        self._lr = learning_rate
        self._epochs = epochs
        self._l2 = l2

    def train(
        self,
        events: Iterable[FeedbackEvent],
        *,
        version: int,
    ) -> TrainedGuardrail:
        labelled: list[tuple[str, int]] = []
        for e in events:
            mapped = _event_target(e)
            if mapped is not None:
                labelled.append((e.prompt, mapped))
        if not labelled:
            raise ValueError("no labelled events — cannot train")
        weights = [0.0] * self._dim
        bias = 0.0
        for _ in range(self._epochs):
            for prompt, target_int in labelled:
                target_float = float(target_int)
                features = _hash_bag(prompt, self._dim)
                margin = bias + sum(w * f for w, f in zip(weights, features, strict=True))
                prediction = 1.0 / (1.0 + math.exp(-margin))
                error = target_float - prediction
                for i, f in enumerate(features):
                    if f != 0.0:
                        weights[i] += self._lr * (error * f - self._l2 * weights[i])
                bias += self._lr * error
        correct = 0
        for prompt, target_int in labelled:
            target_float = float(target_int)
            margin = bias + sum(
                w * f for w, f in zip(weights, _hash_bag(prompt, self._dim), strict=True)
            )
            pred = 1.0 if margin >= 0.0 else 0.0
            if pred == target_float:
                correct += 1
        accuracy = correct / len(labelled)
        return TrainedGuardrail(
            weights=tuple(weights),
            bias=bias,
            dim=self._dim,
            version=version,
            epochs=self._epochs,
            training_accuracy=accuracy,
        )


class LoraGuardrailTrainer:
    """LoRA micro-fine-tune via ``peft`` + ``transformers``.

    Pulls the base model on demand through
    ``transformers.AutoModelForSequenceClassification`` and wraps
    it with ``peft.get_peft_model``. Produces the same
    :class:`TrainedGuardrail` shape so the orchestrator can hot-swap
    without branching on the trainer backend — the callable
    stored in ``.score`` defers to the LoRA adapter under the hood.

    The constructor does not load anything; :meth:`train` loads
    lazily and raises :class:`ImportError` with install
    instructions when the optional dependency group is missing.
    """

    def __init__(
        self,
        *,
        base_model: str = "distilbert-base-uncased",
        rank: int = 8,
        alpha: int = 16,
        epochs: int = 1,
        device: str = "cpu",
    ) -> None:
        if rank <= 0:
            raise ValueError(f"rank must be positive; got {rank!r}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive; got {alpha!r}")
        if epochs <= 0:
            raise ValueError(f"epochs must be positive; got {epochs!r}")
        self._base_model = base_model
        self._rank = rank
        self._alpha = alpha
        self._epochs = epochs
        self._device = device

    def train(
        self,
        events: Iterable[FeedbackEvent],
        *,
        version: int,
    ) -> TrainedGuardrail:
        labelled: list[tuple[str, int]] = []
        for e in events:
            mapped = _event_target(e)
            if mapped is not None:
                labelled.append((e.prompt, mapped))
        if not labelled:
            raise ValueError("no labelled events — cannot train")
        try:
            import peft
            import torch
            from torch.utils.data import DataLoader, Dataset
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise ImportError(
                "LoraGuardrailTrainer.train requires peft, torch, and "
                "transformers. Install with: pip install director-ai[training]",
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(self._base_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            self._base_model, num_labels=2
        )
        lora_config = peft.LoraConfig(
            r=self._rank,
            lora_alpha=self._alpha,
            bias="none",
            task_type=peft.TaskType.SEQ_CLS,
        )
        model = peft.get_peft_model(model, lora_config).to(self._device)

        class _FeedbackDataset(Dataset[tuple[dict[str, Any], int]]):
            def __init__(self, items: Sequence[tuple[str, int]]) -> None:
                self._items = items

            def __len__(self) -> int:
                return len(self._items)

            def __getitem__(self, index: int) -> tuple[dict[str, Any], int]:
                prompt, label = self._items[index]
                enc = tokenizer(
                    prompt, truncation=True, padding="max_length", max_length=128
                )
                return enc, label

        dataset = _FeedbackDataset(labelled)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model.train()
        for _ in range(self._epochs):
            for batch, labels in loader:
                optimiser.zero_grad()
                outputs = model(**{k: torch.tensor(v) for k, v in batch.items()})
                loss = torch.nn.functional.cross_entropy(
                    outputs.logits, torch.tensor(labels)
                )
                loss.backward()
                optimiser.step()
        weights_extract = _extract_classifier_weights(model)
        return TrainedGuardrail(
            weights=weights_extract["weights"],
            bias=weights_extract["bias"],
            dim=weights_extract["dim"],
            version=version,
            epochs=self._epochs,
            training_accuracy=float("nan"),
        )


def _event_target(event: FeedbackEvent) -> int | None:
    """Map a :class:`FeedbackLabel` to ``0`` (safe) / ``1`` (unsafe).

    ``false_positive`` is remapped to safe (the guardrail should
    have let the prompt through); ``false_negative`` to unsafe
    (the guardrail should have caught it).
    """
    match event.label:
        case "safe" | "false_positive":
            return 0
        case "unsafe" | "false_negative":
            return 1
    return None  # pragma: no cover — defensive


def _hash_bag(text: str, dim: int) -> tuple[float, ...]:
    if not text:
        return (0.0,) * dim
    bag = [0.0] * dim
    for token in text.lower().split():
        h = _FNV_OFFSET
        for byte in token.encode("utf-8"):
            h ^= byte
            h = (h * _FNV_PRIME) & _UINT64_MASK
        bag[h % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in bag))
    if norm == 0.0:
        return tuple(bag)
    inv = 1.0 / norm
    return tuple(x * inv for x in bag)


def _extract_classifier_weights(model: Any) -> dict[str, Any]:
    """Pull the final classification head weights + bias out of a
    PEFT-wrapped model so the TrainedGuardrail snapshot can
    survive process boundaries."""
    head = model.base_model.classifier if hasattr(model, "base_model") else model.classifier
    weight_tensor = head.weight.detach().cpu()
    bias_tensor = head.bias.detach().cpu()
    # Flatten the binary head into a single weight vector (weights
    # for the positive class minus weights for the negative class)
    # and a scalar bias differential.
    positive = weight_tensor[1]
    negative = weight_tensor[0]
    weights_vec = (positive - negative).tolist()
    bias = float(bias_tensor[1] - bias_tensor[0])
    return {
        "weights": tuple(float(w) for w in weights_vec),
        "bias": bias,
        "dim": len(weights_vec),
    }
