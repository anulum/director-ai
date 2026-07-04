# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real public-surface coverage for the contradiction scorer."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Protocol, cast

import pytest
import torch
import transformers

from director_ai.core.scoring.contradiction import (
    ContradictionResult,
    ContradictionScorer,
)
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


class _Tensor(Protocol):
    """Small tensor protocol used by the scorer's torch boundary."""

    def to(self, device: object) -> _Tensor:
        """Return this tensor on ``device``."""

    def tolist(self) -> object:
        """Return this tensor as Python scalar/list data."""


@dataclass(frozen=True, slots=True)
class _ModelOutput:
    """Subset of HuggingFace sequence-classifier output used by the scorer."""

    logits: _Tensor


class _LocalTokenizer:
    """Tokenizer protocol that records public scorer call options."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        premises: list[str],
        hypotheses: list[str],
        *,
        return_tensors: str,
        truncation: bool,
        max_length: int,
        padding: bool,
    ) -> dict[str, _Tensor]:
        """Return deterministic tensor inputs for each premise/hypothesis pair."""
        self.calls.append(
            {
                "premises": tuple(premises),
                "hypotheses": tuple(hypotheses),
                "return_tensors": return_tensors,
                "truncation": truncation,
                "max_length": max_length,
                "padding": padding,
            }
        )
        return {
            "input_ids": _tensor(list(range(len(hypotheses)))),
            "attention_mask": _tensor([1.0] * len(hypotheses)),
        }


class _LocalModel:
    """Torch-backed local classifier preserving the sequence-classifier shape."""

    device = "cpu"

    def __init__(self, logits: list[list[float]]) -> None:
        self._logits = logits
        self.config = SimpleNamespace(
            id2label={0: "entailment", 1: "neutral", 2: "contradiction"}
        )
        self.eval_called = False
        self.moved_to: list[str] = []

    def eval(self) -> _LocalModel:
        """Record that the public loader put the model in evaluation mode."""
        self.eval_called = True
        return self

    def to(self, device: str) -> _LocalModel:
        """Record device movement requested by the public loader."""
        self.device = device
        self.moved_to.append(device)
        return self

    def __call__(self, *, input_ids: _Tensor, attention_mask: _Tensor) -> _ModelOutput:
        """Return logits selected by tokenizer-generated row identifiers."""
        _ = attention_mask
        row_ids = [int(value) for value in cast(list[float], input_ids.tolist())]
        return _ModelOutput(logits=_tensor([self._logits[index] for index in row_ids]))


def _tensor(data: object) -> _Tensor:
    """Build a real torch tensor while keeping the local protocol typed."""
    return cast(_Tensor, torch.tensor(data, dtype=torch.float32))


def _scorer(
    logits: list[list[float]],
    *,
    threshold: float = 0.65,
    max_length: int = 96,
) -> tuple[ContradictionScorer, _LocalTokenizer, _LocalModel]:
    """Build a scorer with deterministic local model/tokenizer protocols."""
    tokenizer = _LocalTokenizer()
    model = _LocalModel(logits)
    scorer = ContradictionScorer(
        model,
        tokenizer,
        contradiction_idx=2,
        threshold=threshold,
        max_length=max_length,
    )
    return scorer, tokenizer, model


def test_contradiction_unit_guard_declares_this_real_surface_companion() -> None:
    """The contradiction scorer unit guard should declare this companion."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_contradiction.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_contradiction_real_surface.py" in category


def test_public_scorer_scores_batch_with_real_torch_softmax() -> None:
    """The public scorer should preserve torch softmax semantics in batch mode."""
    scorer, tokenizer, _model = _scorer(
        [
            [0.0, 0.0, 5.0],
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
        ],
        max_length=17,
    )

    scores = scorer.contradiction_batch(
        [
            ("The sky is blue.", "The sky is green."),
            ("The sky is blue.", "The sky is blue."),
            ("The sky is blue.", "The sky has clouds."),
        ]
    )

    assert scores[0] == pytest.approx(
        float(torch.softmax(torch.tensor([0.0, 0.0, 5.0]), dim=0)[2])
    )
    assert scores[0] > scorer.threshold
    assert scores[1] < 0.01
    assert scores[2] < 0.01
    assert tokenizer.calls == [
        {
            "premises": (
                "The sky is blue.",
                "The sky is blue.",
                "The sky is blue.",
            ),
            "hypotheses": (
                "The sky is green.",
                "The sky is blue.",
                "The sky has clouds.",
            ),
            "return_tensors": "pt",
            "truncation": True,
            "max_length": 17,
            "padding": True,
        }
    ]


def test_public_judge_and_empty_input_short_circuit_are_consistent() -> None:
    """Public judge helpers should agree with threshold and empty-input policy."""
    scorer, tokenizer, _model = _scorer([[0.0, 0.0, 4.0]], threshold=0.8)

    result = scorer.judge("The database is encrypted.", "The database is plaintext.")

    assert isinstance(result, ContradictionResult)
    assert result.contradicts is True
    assert result.contradiction >= scorer.threshold
    assert scorer.contradicts(
        "The database is encrypted.", "The database is plaintext."
    )
    assert scorer.contradiction("", "The database is plaintext.") == 0.0
    assert scorer.contradiction("The database is encrypted.", " ") == 0.0
    assert len(tokenizer.calls) == 2


def test_from_pretrained_uses_transformers_protocol_and_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public loader should wire tokenizer, model, revision, and CUDA device."""
    tokenizer = _LocalTokenizer()
    model = _LocalModel([[0.0, 0.0, 5.0]])
    calls: list[dict[str, object]] = []

    def tokenizer_loader(model_id: str, *, revision: str) -> _LocalTokenizer:
        """Capture tokenizer load arguments without network access."""
        calls.append({"kind": "tokenizer", "model_id": model_id, "revision": revision})
        return tokenizer

    def model_loader(model_id: str, *, revision: str) -> _LocalModel:
        """Capture model load arguments without network access."""
        calls.append({"kind": "model", "model_id": model_id, "revision": revision})
        return model

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification,
        "from_pretrained",
        model_loader,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    scorer = ContradictionScorer.from_pretrained(
        "local/contradiction",
        revision="0" * 40,
        device=1,
        threshold=0.7,
    )

    assert calls == [
        {
            "kind": "tokenizer",
            "model_id": "local/contradiction",
            "revision": "0" * 40,
        },
        {"kind": "model", "model_id": "local/contradiction", "revision": "0" * 40},
    ]
    assert model.eval_called is True
    assert model.moved_to == ["cuda:1"]
    assert scorer.threshold == pytest.approx(0.7)
