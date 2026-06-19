# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — RAGTruth annotation review tests

from __future__ import annotations

import json

import pytest

from training.ragtruth_annotation_review import (
    build_review_packet,
    build_review_template,
    estimate_review_effect,
    load_jsonl,
    write_markdown,
)


def _record(row_index: int, label: int, probs: list[float]) -> dict:
    return {
        "row_index": row_index,
        "label": label,
        "resp_probs": probs,
    }


def _candidate(row_index: int) -> dict:
    return {
        "row_index": row_index,
        "task_type": "QA",
        "current_label": "grounded",
        "primary_category": "possible_annotation_noise",
        "factors": ["possible_annotation_noise"],
        "tokens_at_threshold": 8,
        "max_token_probability": 0.99,
        "query_snippet": "question",
        "output_snippet": "answer",
        "context_snippet": "context",
    }


def _decision(row_index: int, decision: str, rationale: str = "reviewed") -> dict:
    return {
        "row_index": row_index,
        "reviewer_decision": decision,
        "reviewer_rationale": rationale,
    }


def test_build_review_template_preserves_label_and_allowed_decisions() -> None:
    template = build_review_template([_candidate(42)])

    assert template[0]["row_index"] == 42
    assert template[0]["current_label"] == "grounded"
    assert template[0]["primary_category"] == "possible_annotation_noise"
    assert template[0]["tokens_at_threshold"] == 8
    assert template[0]["max_token_probability"] == 0.99
    assert template[0]["reviewer_decision"] == ""
    assert template[0]["allowed_reviewer_decisions"] == [
        "confirmed_grounded",
        "confirmed_hallucinated",
        "exclude_uncertain",
    ]


def test_load_jsonl_rejects_non_object_lines(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("[1, 2]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not a JSON object"):
        load_jsonl(path)


def test_estimate_review_effect_relabels_confirmed_annotation_noise() -> None:
    records = [
        _record(10, 0, [0.9, 0.8]),
        _record(11, 0, [0.1, 0.2]),
        _record(12, 1, [0.9, 0.8]),
    ]

    effect = estimate_review_effect(
        records,
        [_decision(10, "confirmed_hallucinated")],
        threshold=0.7,
        min_tokens=2,
    )

    assert effect["baseline_metrics"]["fp"] == 1
    assert effect["review_adjusted_metrics"]["fp"] == 0
    assert effect["review_adjusted_metrics"]["tp"] == 2
    assert effect["review_adjusted_metrics"]["fpr"] == 0.0


def test_estimate_review_effect_excludes_uncertain_rows() -> None:
    records = [
        _record(10, 0, [0.9, 0.8]),
        _record(11, 0, [0.1, 0.2]),
        _record(12, 1, [0.9, 0.8]),
    ]

    effect = estimate_review_effect(
        records,
        [_decision(10, "exclude_uncertain")],
        threshold=0.7,
        min_tokens=2,
    )

    assert effect["excluded_count"] == 1
    assert effect["review_adjusted_metrics"]["fp"] == 0
    assert effect["review_adjusted_metrics"]["tn"] == 1


def test_estimate_review_effect_rejects_invalid_decisions() -> None:
    with pytest.raises(ValueError, match="invalid reviewer_decision"):
        estimate_review_effect(
            [_record(10, 0, [0.9])],
            [_decision(10, "maybe")],
            threshold=0.7,
            min_tokens=1,
        )


def test_estimate_review_effect_rejects_unknown_rows() -> None:
    with pytest.raises(ValueError, match="is not in cache"):
        estimate_review_effect(
            [_record(10, 0, [0.9])],
            [_decision(99, "confirmed_grounded", "")],
            threshold=0.7,
            min_tokens=1,
        )


def test_build_review_packet_without_decisions_stays_pending(tmp_path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps(_candidate(10)) + "\n", encoding="utf-8")
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps([_record(10, 0, [0.9, 0.8])]), encoding="utf-8")
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"best": {"p": 0.7, "k": 2}}), encoding="utf-8")

    packet = build_review_packet(
        candidates_path=candidates,
        cache_path=cache,
        result_path=result,
    )

    assert packet["status"] == "manual_review_pending"
    assert packet["candidate_count"] == 1


def test_build_review_packet_with_decisions_adds_sensitivity(tmp_path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps(_candidate(10)) + "\n", encoding="utf-8")
    cache = tmp_path / "cache.json"
    cache.write_text(
        json.dumps(
            [
                _record(10, 0, [0.9, 0.8]),
                _record(11, 0, [0.1, 0.2]),
                _record(12, 1, [0.9, 0.8]),
            ]
        ),
        encoding="utf-8",
    )
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"best": {"p": 0.7, "k": 2}}), encoding="utf-8")
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(
        json.dumps(_decision(10, "confirmed_hallucinated")) + "\n",
        encoding="utf-8",
    )

    packet = build_review_packet(
        candidates_path=candidates,
        cache_path=cache,
        result_path=result,
        decisions_path=decisions,
    )

    assert packet["review_effect"]["review_adjusted_metrics"]["fp"] == 0
    assert packet["status"] == "review_adjustment_meets_internal_gate"


def test_write_markdown_includes_pending_and_adjusted_metrics(tmp_path) -> None:
    output = tmp_path / "packet.md"
    packet = {
        "status": "review_adjustment_does_not_meet_internal_gate",
        "candidate_count": 1,
        "decision_rule": {"p": 0.7, "k": 2},
        "next_action": "review only",
        "review_effect": {
            "reviewed_count": 1,
            "decision_counts": {"confirmed_grounded": 1},
            "excluded_count": 0,
            "baseline_metrics": {
                "f1": 0.5,
                "precision": 0.5,
                "recall": 0.5,
                "fpr": 0.5,
            },
            "review_adjusted_metrics": {
                "f1": 0.6,
                "precision": 0.6,
                "recall": 0.6,
                "fpr": 0.4,
            },
            "note": "internal only",
        },
    }

    write_markdown(packet, output)

    text = output.read_text(encoding="utf-8")
    assert "review_adjustment_does_not_meet_internal_gate" in text
    assert "Adjusted: F1 `0.6000`" in text
