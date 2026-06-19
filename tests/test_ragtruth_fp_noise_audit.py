# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — RAGTruth false-positive noise audit tests

from __future__ import annotations

import json

from training.ragtruth_fp_noise_audit import (
    build_noise_audit,
    categorise_false_positive,
    write_markdown,
)


def _record(label: int, probs: list[float], *, task_type: str = "QA") -> dict:
    return {
        "label": label,
        "resp_probs": probs,
        "row_index": len(probs) + label,
        "task_type": task_type,
        "context_tokens": 64,
        "context_chars": 512,
        "response_tokens": len(probs),
        "response_chars": len(probs) * 8,
    }


def _result() -> dict:
    return {
        "best": {
            "f1": 0.7544,
            "precision": 0.7376,
            "recall": 0.7720,
            "fpr": 0.1474,
            "tp": 728,
            "fp": 259,
            "tn": 1498,
            "fn": 215,
            "p": 0.7,
            "k": 2,
        }
    }


def test_categorise_false_positive_prioritises_structural_factors() -> None:
    item = {
        "task_type": "Data2txt",
        "context_tokens": 1200,
        "context_chars": 9000,
        "response_tokens": 300,
        "response_chars": 1500,
        "tokens_at_threshold": 50,
        "max_token_probability": 0.99,
    }

    category = categorise_false_positive(item)

    assert category["primary_category"] == "data2txt_structural"
    assert category["factors"] == [
        "data2txt_structural",
        "likely_truncation_or_context_loss",
        "long_response_activation",
        "possible_annotation_noise",
    ]


def test_build_noise_audit_blocks_jarvis_for_structural_false_positives(
    tmp_path,
) -> None:
    cache = tmp_path / "cache.json"
    cache.write_text(
        json.dumps(
            [
                _record(1, [0.9, 0.8], task_type="QA"),
                _record(0, [0.95, 0.9], task_type="Data2txt"),
                {
                    **_record(0, [0.92, 0.91, 0.9], task_type="Summary"),
                    "context_tokens": 1100,
                    "context_chars": 8500,
                },
                _record(0, [0.1, 0.2], task_type="QA"),
            ]
        )
    )
    result = tmp_path / "result.json"
    result.write_text(json.dumps(_result()))

    packet = build_noise_audit(
        cache_path=cache,
        result_path=result,
        top_n=5,
    )

    assert packet["false_positive_count"] == 2
    assert packet["decision"]["jarvis_decision"] == "do_not_launch_jarvis"
    assert "task/length/context-aware" in packet["decision"]["recommendation"]
    assert packet["primary_category_counts"] == {
        "data2txt_structural": 1,
        "likely_truncation_or_context_loss": 1,
    }


def test_write_markdown_includes_examples_and_decision(tmp_path) -> None:
    packet = {
        "decision": {
            "jarvis_decision": "do_not_launch_jarvis",
            "recommendation": "manual_label_review_before_training",
            "structural_primary_fraction": 0.5,
            "annotation_primary_fraction": 0.25,
        },
        "decision_rule": {"p": 0.7, "k": 2},
        "baseline_metrics": {
            "f1": 0.75,
            "precision": 0.73,
            "recall": 0.77,
            "fpr": 0.14,
        },
        "false_positive_count": 1,
        "primary_category_counts": {"possible_annotation_noise": 1},
        "factor_counts": {"possible_annotation_noise": 1},
        "task_type_counts": {"QA": 1},
        "top_examples": [
            {
                "row_index": 42,
                "primary_category": "possible_annotation_noise",
                "task_type": "QA",
                "factors": ["possible_annotation_noise"],
                "context_tokens": 20,
                "context_chars": 120,
                "response_tokens": 10,
                "response_chars": 60,
                "tokens_at_threshold": 8,
                "max_token_probability": 0.99,
                "mean_top5_token_probability": 0.95,
                "query_snippet": "question",
                "output_snippet": "answer",
            }
        ],
    }
    output = tmp_path / "audit.md"

    write_markdown(packet, output)

    text = output.read_text()
    assert "do_not_launch_jarvis" in text
    assert "Row 42" in text
    assert "possible_annotation_noise" in text
