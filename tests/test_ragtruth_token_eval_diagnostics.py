# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — RAGTruth token-eval diagnostics tests

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from training import eval_ragtruth_token


def test_evaluate_reports_false_positive_diagnostics(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(eval_ragtruth_token, "RESULT", str(tmp_path / "result.json"))
    monkeypatch.setattr(eval_ragtruth_token, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(eval_ragtruth_token, "HARD_NEGATIVES", "0")

    result = eval_ragtruth_token.evaluate(
        [
            {
                "label": 0,
                "resp_probs": [0.91, 0.2],
                "row_index": 10,
                "task_type": "qa",
                "response_tokens": 2,
                "context_tokens": 120,
                "context_chars": 600,
                "response_chars": 12,
                "hallucination_span_count": 0,
            },
            {
                "label": 0,
                "resp_probs": [0.1, 0.2],
                "row_index": 11,
                "task_type": "summarization",
                "response_tokens": 2,
                "context_tokens": 900,
                "context_chars": 9000,
                "response_chars": 12,
                "hallucination_span_count": 0,
            },
            {
                "label": 1,
                "resp_probs": [0.92, 0.1],
                "row_index": 12,
                "task_type": "qa",
                "response_tokens": 2,
                "context_tokens": 130,
                "context_chars": 700,
                "response_chars": 12,
                "hallucination_span_count": 1,
            },
        ]
    )

    diagnostics = result["diagnostics"]
    assert diagnostics["decision_rule"] == {"p": 0.3, "k": 1}
    qa = next(row for row in diagnostics["by_task_type"] if row["group"] == "qa")
    assert qa["fp"] == 1
    assert qa["tp"] == 1
    assert qa["fpr"] == 1.0
    assert diagnostics["top_false_positives"][0]["row_index"] == 10


def test_evaluate_enriches_legacy_cache_metadata(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(eval_ragtruth_token, "RESULT", str(tmp_path / "result.json"))
    monkeypatch.setattr(eval_ragtruth_token, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(eval_ragtruth_token, "HARD_NEGATIVES", "0")
    monkeypatch.setattr(
        eval_ragtruth_token,
        "load_dataset",
        lambda *_args, **_kwargs: [
            {
                "context": "context words",
                "output": "answer",
                "task_type": "qa",
                "hallucination_labels": "[]",
            }
        ],
    )

    result = eval_ragtruth_token.evaluate([{"label": 0, "resp_probs": [0.95]}])

    task_rows = result["diagnostics"]["by_task_type"]
    assert task_rows[0]["group"] == "qa"
    assert result["diagnostics"]["top_false_positives"][0]["context_chars"] == 13


def test_evaluate_writes_hard_negative_jsonl(monkeypatch, tmp_path) -> None:
    hard_negatives = tmp_path / "hard_negatives.jsonl"
    monkeypatch.setattr(eval_ragtruth_token, "RESULT", str(tmp_path / "result.json"))
    monkeypatch.setattr(eval_ragtruth_token, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(eval_ragtruth_token, "HARD_NEGATIVES", str(hard_negatives))
    monkeypatch.setattr(
        eval_ragtruth_token,
        "load_dataset",
        lambda *_args, **_kwargs: [
            {
                "context": "grounded context",
                "query": "question",
                "output": "grounded answer",
                "task_type": "QA",
                "hallucination_labels": "[]",
            },
            {
                "context": "hall context",
                "query": "question",
                "output": "bad answer",
                "task_type": "QA",
                "hallucination_labels": '[{"start": 0, "end": 3}]',
            },
        ],
    )

    result = eval_ragtruth_token.evaluate(
        [
            {"label": 0, "resp_probs": [0.95, 0.91], "row_index": 0},
            {"label": 1, "resp_probs": [0.96, 0.1], "row_index": 1},
        ]
    )

    rows = [json.loads(line) for line in hard_negatives.read_text().splitlines()]
    assert result["hard_negatives"]["count"] == 1
    assert rows[0]["row_index"] == 0
    assert rows[0]["label"] == 0
    assert rows[0]["source_split"] == "test"
    assert rows[0]["context"] == "grounded context"
    assert rows[0]["tokens_at_threshold"] == 2
    assert rows[0]["candidate_weight"] > 1.0


def test_selected_indices_preserve_original_row_numbers(monkeypatch) -> None:
    monkeypatch.setattr(eval_ragtruth_token, "DATASET_ROW_OFFSET", 3)
    monkeypatch.setattr(eval_ragtruth_token, "DATASET_ROW_STRIDE", 4)
    monkeypatch.setattr(eval_ragtruth_token, "DATASET_MAX_ROWS", 3)

    assert eval_ragtruth_token._selected_indices(20) == [3, 7, 11]


def test_selected_indices_reject_invalid_stride(monkeypatch) -> None:
    monkeypatch.setattr(eval_ragtruth_token, "DATASET_ROW_OFFSET", 0)
    monkeypatch.setattr(eval_ragtruth_token, "DATASET_ROW_STRIDE", 0)
    monkeypatch.setattr(eval_ragtruth_token, "DATASET_MAX_ROWS", 0)

    try:
        eval_ragtruth_token._selected_indices(20)
    except ValueError as exc:
        assert "DATASET_ROW_STRIDE" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("expected invalid stride to raise")


def test_evaluate_creates_result_parent_directory(monkeypatch, tmp_path) -> None:
    result = tmp_path / "nested" / "result.json"
    monkeypatch.setattr(eval_ragtruth_token, "RESULT", str(result))
    monkeypatch.setattr(eval_ragtruth_token, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(eval_ragtruth_token, "HARD_NEGATIVES", "0")

    eval_ragtruth_token.evaluate(
        [
            {
                "label": 0,
                "resp_probs": [0.1],
                "task_type": "QA",
                "context_chars": 1,
            },
            {
                "label": 1,
                "resp_probs": [0.9],
                "task_type": "QA",
                "context_chars": 1,
            },
        ]
    )

    assert result.is_file()
