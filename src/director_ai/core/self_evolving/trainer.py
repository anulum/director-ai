# SPDX-License-Identifier: BUSL-1.1
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

try:  # pragma: no cover - optional acceleration
    from backfire_kernel import rust_sum_f64

    _RUST_SELF_EVOLVING = True
except ImportError:  # pragma: no cover - mandatory accelerator guard
    _RUST_SELF_EVOLVING = True

    def rust_sum_f64(_values: list[float]) -> float:
        """Raise because the compiled ``rust_sum_f64`` kernel is unavailable."""
        raise RuntimeError("backfire_kernel rust_sum_f64 is unavailable")


from ..mandatory import mandatory_execution
from ..model_revisions import resolve_model_revision
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
        """Return the probability that ``text`` is unsafe, in ``[0, 1]``."""
        features = _hash_bag(text, self.dim)
        margin = self.bias + _dot(self.weights, features)
        return 1.0 / (1.0 + math.exp(-margin))


@runtime_checkable
class GuardrailTrainer(Protocol):
    """Train a guardrail from labelled feedback events."""

    def train(
        self,
        events: Iterable[FeedbackEvent],
        *,
        version: int,
    ) -> TrainedGuardrail:
        """Train a guardrail snapshot from the labelled ``events``."""
        ...


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
        """Train the perceptron head on the labelled events and snapshot it."""
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
                margin = bias + _dot(weights, features)
                prediction = 1.0 / (1.0 + math.exp(-margin))
                error = target_float - prediction
                for i, f in enumerate(features):
                    if f != 0.0:
                        weights[i] += self._lr * (error * f - self._l2 * weights[i])
                bias += self._lr * error
        correct = _sum_float(
            [
                1.0
                if (bias + _dot(weights, _hash_bag(prompt, self._dim)) >= 0.0)
                == (float(target_int) >= 0.5)
                else 0.0
                for prompt, target_int in labelled
            ]
        )
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
    it with ``peft.get_peft_model``. After fine-tuning, the model is
    distilled into the hash-bag head that :class:`TrainedGuardrail`
    evaluates: the fine-tuned model labels each training prompt and a
    :class:`PerceptronGuardrailTrainer` fits those targets, producing the
    same :class:`TrainedGuardrail` shape so the orchestrator can hot-swap
    without branching on the trainer backend. (The raw classifier-head
    weights live in the transformer's hidden space and cannot be scored
    against a hash-bag, so they are not used directly.)

    The constructor does not load anything; :meth:`train` loads
    lazily and raises :class:`ImportError` with install
    instructions when the mandatory dependency group is missing.
    """

    def __init__(
        self,
        *,
        base_model: str = "distilbert-base-uncased",
        base_model_revision: str | None = None,
        rank: int = 8,
        alpha: int = 16,
        epochs: int = 1,
        device: str = "cpu",
        distill_dim: int = 1024,
        distill_epochs: int = 4,
    ) -> None:
        if rank <= 0:
            raise ValueError(f"rank must be positive; got {rank!r}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive; got {alpha!r}")
        if epochs <= 0:
            raise ValueError(f"epochs must be positive; got {epochs!r}")
        if distill_dim <= 0:
            raise ValueError(f"distill_dim must be positive; got {distill_dim!r}")
        if distill_epochs <= 0:
            raise ValueError(f"distill_epochs must be positive; got {distill_epochs!r}")
        self._base_model = base_model
        self._base_model_revision = base_model_revision
        self._rank = rank
        self._alpha = alpha
        self._epochs = epochs
        self._device = device
        self._distill_dim = distill_dim
        self._distill_epochs = distill_epochs

    def train(
        self,
        events: Iterable[FeedbackEvent],
        *,
        version: int,
    ) -> TrainedGuardrail:
        """Run the LoRA micro-fine-tune on the labelled events and snapshot it."""
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
        base_model_revision = resolve_model_revision(
            self._base_model,
            self._base_model_revision,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self._base_model,
            revision=base_model_revision,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self._base_model,
            num_labels=2,
            revision=base_model_revision,
        )
        lora_config = peft.LoraConfig(
            r=self._rank,
            lora_alpha=self._alpha,
            bias="none",
            task_type=peft.TaskType.SEQ_CLS,
        )
        model = peft.get_peft_model(model, lora_config).to(self._device)

        class _FeedbackDataset(
            Dataset[tuple[dict[str, Any], int]]  # type: ignore[misc,unused-ignore] # torch Dataset may be Any without torch stubs in the lean CI type env.
        ):
            def __init__(self, items: Sequence[tuple[str, int]]) -> None:
                self._items = items

            def __len__(self) -> int:
                return len(self._items)

            def __getitem__(self, index: int) -> tuple[dict[str, Any], int]:
                prompt, label = self._items[index]
                enc = tokenizer(
                    prompt, truncation=True, padding="max_length", max_length=128
                )
                # Return tensors so the default collate stacks each field into a
                # [batch, seq] tensor. Returning raw lists makes collate transpose
                # them into a list of per-position tensors that the model cannot
                # consume.
                return {k: torch.tensor(v) for k, v in enc.items()}, label

        dataset = _FeedbackDataset(labelled)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model.train()
        for _ in range(self._epochs):
            for batch, labels in loader:
                optimiser.zero_grad()
                outputs = model(**batch)
                loss: Any = torch.nn.functional.cross_entropy(outputs.logits, labels)
                loss.backward()
                optimiser.step()

        # Distil the fine-tuned model into the hash-bag head that
        # ``TrainedGuardrail.score`` actually evaluates. The raw classifier-head
        # weights live in the transformer's hidden space and are meaningless
        # against a hash-bag featuriser; instead we run the fine-tuned model over
        # the training prompts and use its per-prompt unsafe probability as the
        # distillation target, so the deployed snapshot reproduces the LoRA
        # decision boundary in the feature space its inference hook uses.
        model.eval()
        distilled: list[FeedbackEvent] = []
        for prompt, _label in labelled:
            enc = tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                logits = model(**enc).logits
            prob_unsafe = _unsafe_probability(logits)
            distilled.append(
                FeedbackEvent(
                    prompt=prompt,
                    response="",
                    label="unsafe" if prob_unsafe >= 0.5 else "safe",
                )
            )
        return PerceptronGuardrailTrainer(
            dim=self._distill_dim,
            epochs=self._distill_epochs,
        ).train(distilled, version=version)


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
    return None


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
    norm = math.sqrt(_sum_float([x * x for x in bag]))
    if norm == 0.0:
        return tuple(bag)
    inv = 1.0 / norm
    return tuple(x * inv for x in bag)


def _logits_to_pair(logits: Any) -> tuple[float, float]:
    """Reduce a two-class logits tensor (or nested list) to ``(safe, unsafe)``.

    Accepts a torch tensor or a plain (possibly batched) list, unwrapping a
    leading batch dimension so ``[[a, b]]`` and ``[a, b]`` both yield ``(a, b)``.
    """
    values: Any = (
        logits.detach().cpu().tolist() if hasattr(logits, "detach") else logits
    )
    while isinstance(values, list) and values and isinstance(values[0], list):
        values = values[0]
    return float(values[0]), float(values[1])


def _unsafe_probability(logits: Any) -> float:
    """Return P(unsafe) from a two-class logits tensor via a stable softmax."""
    safe_logit, unsafe_logit = _logits_to_pair(logits)
    ceiling = max(safe_logit, unsafe_logit)
    exp_safe = math.exp(safe_logit - ceiling)
    exp_unsafe = math.exp(unsafe_logit - ceiling)
    return exp_unsafe / (exp_safe + exp_unsafe)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return _sum_float([x * y for x, y in zip(a, b, strict=True)])


def _sum_float(values: list[float]) -> float:
    if _RUST_SELF_EVOLVING:
        with mandatory_execution(__name__, component="mandatory accelerated path"):
            return float(rust_sum_f64(values))
    return sum(values)
