# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — AdversaryScorer

"""Online perceptron distilled from adversarial cases.

The scorer featurises a prompt via an FNV-1a hash bag (same
family the self-evolving guardrail uses) and learns a logistic
head over the mined adversarial / safe corpus. Its ``score(prompt)``
returns the probability the prompt matches the adversarial
distribution; downstream routers compare this against a
caller-configured threshold.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .suite import AdversarialCase

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_UINT64_MASK = 0xFFFFFFFFFFFFFFFF


@dataclass(frozen=True)
class TrainedAdversaryScorer:
    """Immutable snapshot of a trained scorer."""

    weights: tuple[float, ...]
    bias: float
    dim: int
    version: int
    training_accuracy: float

    def score(self, prompt: str) -> float:
        features = _hash_bag(prompt, self.dim)
        margin = self.bias + sum(
            w * f for w, f in zip(self.weights, features, strict=True)
        )
        return 1.0 / (1.0 + math.exp(-margin))


@runtime_checkable
class AdversaryScorer(Protocol):
    """Protocol: anything that scores a prompt's adversarial
    probability in ``[0, 1]``."""

    def score(self, prompt: str) -> float: ...


class PerceptronAdversaryScorer:
    """Train a :class:`TrainedAdversaryScorer` via online
    perceptron + logistic head.

    Parameters
    ----------
    dim :
        Hash-bag feature dimensionality. Default 1024.
    learning_rate :
        Step size. Default 0.1.
    epochs :
        Passes over the training set. Default 4.
    l2 :
        L2 regularisation. Default 1e-4.
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
            raise ValueError("dim must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        self._dim = dim
        self._lr = learning_rate
        self._epochs = epochs
        self._l2 = l2

    def train(
        self,
        *,
        adversarial: Sequence[AdversarialCase],
        safe: Sequence[str],
        version: int,
    ) -> TrainedAdversaryScorer:
        if not adversarial:
            raise ValueError("adversarial set must be non-empty")
        if not safe:
            raise ValueError("safe set must be non-empty")
        labelled: list[tuple[str, int]] = [(case.prompt, 1) for case in adversarial] + [
            (prompt, 0) for prompt in safe
        ]
        weights = [0.0] * self._dim
        bias = 0.0
        for _ in range(self._epochs):
            for prompt, target in labelled:
                features = _hash_bag(prompt, self._dim)
                margin = bias + sum(
                    w * f for w, f in zip(weights, features, strict=True)
                )
                prediction = 1.0 / (1.0 + math.exp(-margin))
                error = float(target) - prediction
                for i, f in enumerate(features):
                    if f != 0.0:
                        weights[i] += self._lr * (error * f - self._l2 * weights[i])
                bias += self._lr * error
        correct = 0
        for prompt, target in labelled:
            margin = bias + sum(
                w * f
                for w, f in zip(weights, _hash_bag(prompt, self._dim), strict=True)
            )
            pred = 1 if margin >= 0 else 0
            if pred == target:
                correct += 1
        accuracy = correct / len(labelled)
        return TrainedAdversaryScorer(
            weights=tuple(weights),
            bias=bias,
            dim=self._dim,
            version=version,
            training_accuracy=accuracy,
        )


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
