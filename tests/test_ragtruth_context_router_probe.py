# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — RAGTruth context router probe tests

from __future__ import annotations

import json

from training.ragtruth_context_router_probe import (
    Rule,
    build_context_router_probe,
    build_router,
    evaluate_router,
    segment_key,
    select_rule,
    write_review_candidates,
)


def _record(
    label: int,
    probs: list[float],
    *,
    task_type: str = "QA",
    context_tokens: int = 64,
    response_tokens: int | None = None,
) -> dict:
    return {
        "label": label,
        "resp_probs": probs,
        "row_index": len(probs) + label,
        "task_type": task_type,
        "context_tokens": context_tokens,
        "context_chars": context_tokens * 5,
        "response_tokens": response_tokens
        if response_tokens is not None
        else len(probs),
        "response_chars": (
            response_tokens if response_tokens is not None else len(probs)
        )
        * 6,
    }


def test_segment_key_uses_task_context_and_response_buckets() -> None:
    record = _record(
        0,
        [0.1],
        task_type="Data2txt",
        context_tokens=1200,
        response_tokens=300,
    )

    assert segment_key(record, "task_context_response") == (
        "Data2txt|ctx=>1024|resp=>256"
    )


def test_select_rule_can_prefer_fpr_gate_candidate() -> None:
    records = [
        _record(1, [0.99, 0.98]),
        _record(1, [0.97, 0.96]),
        _record(0, [0.2, 0.1]),
        _record(0, [0.3, 0.2]),
    ]

    selected = select_rule(records, max_fpr=0.0, min_recall=0.5)

    assert selected["selected_from_gate_pool"] is True
    assert selected["metrics"]["fpr"] == 0.0
    assert selected["metrics"]["recall"] >= 0.5


def test_build_and_evaluate_router_routes_supported_segments() -> None:
    records = [
        _record(1, [0.99, 0.98], task_type="QA"),
        _record(1, [0.97, 0.96], task_type="QA"),
        _record(0, [0.2, 0.1], task_type="QA"),
        _record(0, [0.3, 0.2], task_type="QA"),
    ] * 5

    router = build_router(
        records,
        mode="task",
        min_segment_size=4,
        max_fpr=0.0,
        min_recall=0.5,
    )
    metrics = evaluate_router(records, router)

    assert "QA" in router["segments"]
    assert metrics["fpr"] == 0.0
    assert metrics["routed_segments"]["QA"] == len(records)


def test_build_context_router_probe_records_negative_decision(tmp_path) -> None:
    calibration = [
        _record(1, [0.99, 0.98], task_type="QA"),
        _record(1, [0.95, 0.94], task_type="Summary"),
        _record(0, [0.2, 0.1], task_type="QA"),
        _record(0, [0.3, 0.2], task_type="Summary"),
    ] * 5
    test = [
        _record(1, [0.99, 0.98], task_type="QA"),
        _record(0, [0.99, 0.98], task_type="QA"),
    ]
    audit = {
        "top_examples": [
            {
                "row_index": 7,
                "task_type": "QA",
                "factors": ["possible_annotation_noise"],
                "query_snippet": "q",
                "output_snippet": "o",
                "context_snippet": "c",
            }
        ]
    }
    calibration_cache = tmp_path / "calibration.json"
    test_cache = tmp_path / "test.json"
    noise_audit = tmp_path / "noise.json"
    calibration_cache.write_text(json.dumps(calibration))
    test_cache.write_text(json.dumps(test))
    noise_audit.write_text(json.dumps(audit))

    packet = build_context_router_probe(
        calibration_cache_path=calibration_cache,
        test_cache_path=test_cache,
        noise_audit_path=noise_audit,
        max_fpr=0.0,
        min_recall=1.0,
        min_f1=1.0,
        min_segment_size=4,
    )

    assert packet["decision"] == "do_not_launch_jarvis_from_context_router_probe"
    assert packet["best_gate_candidate"] is None
    assert packet["annotation_review_candidates"][0]["row_index"] == 7


def test_write_review_candidates_does_not_relabel(tmp_path) -> None:
    jsonl = tmp_path / "review.jsonl"
    md = tmp_path / "review.md"

    write_review_candidates(
        [
            {
                "row_index": 3,
                "task_type": "QA",
                "factors": ["possible_annotation_noise"],
                "query_snippet": "q",
                "output_snippet": "o",
                "context_snippet": "c",
            }
        ],
        jsonl,
        md,
    )

    payload = json.loads(jsonl.read_text().strip())
    assert payload["review_status"] == "needs_manual_review"
    assert payload["current_label"] == "grounded"
    assert "Row 3" in md.read_text()


def test_evaluate_router_accepts_fixed_rule_shape() -> None:
    router = {
        "mode": "global",
        "default": {
            "rule": {"p": Rule(0.8, 1).p, "k": 1, "max_density": 1.0},
            "metrics": {},
        },
        "segments": {},
    }

    metrics = evaluate_router(
        [_record(1, [0.9]), _record(0, [0.1])],
        router,
    )

    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
