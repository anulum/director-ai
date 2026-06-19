# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — RAGTruth hard-negative training tests

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from training.train_ragtruth_token import (  # noqa: E402
    _compute_weighted_token_loss,
    _load_hard_negative_weights,
)


def test_load_hard_negative_weights_rejects_test_split(tmp_path) -> None:
    path = tmp_path / "hard_negatives.jsonl"
    path.write_text(
        json.dumps(
            {
                "row_index": 7,
                "source_split": "test",
                "candidate_weight": 3.0,
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="refusing to train"):
        _load_hard_negative_weights(str(path), max_weight=5.0)


def test_load_hard_negative_weights_caps_and_floors_values(tmp_path) -> None:
    path = tmp_path / "hard_negatives.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "row_index": 10,
                        "source_split": "train",
                        "candidate_weight": 9.0,
                    }
                ),
                json.dumps(
                    {
                        "row_index": 11,
                        "source_split": "train",
                        "candidate_weight": 0.2,
                    }
                ),
            ]
        )
        + "\n"
    )

    weights = _load_hard_negative_weights(str(path), max_weight=4.0)

    assert weights == {10: 4.0, 11: 1.0}


def test_load_hard_negative_weights_requires_row_index(tmp_path) -> None:
    path = tmp_path / "hard_negatives.jsonl"
    path.write_text(json.dumps({"source_split": "train"}) + "\n")

    with pytest.raises(ValueError, match="missing row_index"):
        _load_hard_negative_weights(str(path), max_weight=5.0)


def test_hard_negative_fp_penalty_increases_supported_fp_loss() -> None:
    logits = torch.tensor([[[0.0, 3.0], [3.0, 0.0]]])
    labels = torch.tensor([[0, 0]])
    class_weights = torch.tensor([1.0, 2.0])
    hard_negative_weight = torch.tensor([4.0])

    without_penalty = _compute_weighted_token_loss(
        logits=logits,
        labels=labels,
        class_weights=class_weights,
        focal_gamma=0.0,
        hard_negative_weight=hard_negative_weight,
        hard_negative_fp_penalty=0.0,
    )
    with_penalty = _compute_weighted_token_loss(
        logits=logits,
        labels=labels,
        class_weights=class_weights,
        focal_gamma=0.0,
        hard_negative_weight=hard_negative_weight,
        hard_negative_fp_penalty=0.5,
    )

    assert with_penalty > without_penalty


def test_hard_negative_fp_penalty_ignores_non_hard_negative_rows() -> None:
    logits = torch.tensor([[[0.0, 3.0], [3.0, 0.0]]])
    labels = torch.tensor([[0, 0]])
    class_weights = torch.tensor([1.0, 2.0])
    hard_negative_weight = torch.tensor([1.0])

    without_penalty = _compute_weighted_token_loss(
        logits=logits,
        labels=labels,
        class_weights=class_weights,
        focal_gamma=0.0,
        hard_negative_weight=hard_negative_weight,
        hard_negative_fp_penalty=0.0,
    )
    with_penalty = _compute_weighted_token_loss(
        logits=logits,
        labels=labels,
        class_weights=class_weights,
        focal_gamma=0.0,
        hard_negative_weight=hard_negative_weight,
        hard_negative_fp_penalty=0.5,
    )

    assert torch.equal(with_penalty, without_penalty)
