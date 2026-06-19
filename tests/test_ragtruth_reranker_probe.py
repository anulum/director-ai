# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — RAGTruth reranker probe tests

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

from training.ragtruth_reranker_probe import (
    aggregate_confusions,
    build_probe,
    evaluate_calibration_to_test,
    feature_names,
    record_features,
    select_threshold,
)


def _record(label: int, probs: list[float], task_type: str = "QA") -> dict:
    return {
        "label": label,
        "resp_probs": probs,
        "task_type": task_type,
        "context_chars": 512,
        "response_chars": 128,
        "context_tokens": 80,
        "response_tokens": len(probs),
        "row_index": len(probs) + label,
    }


def test_record_features_match_feature_names() -> None:
    record = _record(0, [0.1, 0.9, 0.95], "Summary")

    token_only = record_features(record, include_metadata=False)
    with_metadata = record_features(record, include_metadata=True)

    assert len(token_only) == len(feature_names(include_metadata=False))
    assert len(with_metadata) == len(feature_names(include_metadata=True))
    assert with_metadata[-3:] == [0.0, 1.0, 0.0]
    assert with_metadata[0] == 0.95


def test_select_threshold_prefers_gate_when_available() -> None:
    labels = np.array([1, 1, 1, 0, 0, 0])
    probabilities = np.array([0.95, 0.9, 0.2, 0.15, 0.1, 0.05])

    selected = select_threshold(
        labels,
        probabilities,
        max_fpr=0.0,
        min_recall=0.6,
        min_f1=0.7,
    )

    assert selected["fpr"] == 0.0
    assert selected["recall"] >= 0.6
    assert selected["f1"] >= 0.7


def test_aggregate_confusions_recomputes_metrics() -> None:
    metrics = aggregate_confusions(
        [
            {"tp": 2, "fp": 1, "tn": 3, "fn": 1},
            {"tp": 1, "fp": 0, "tn": 4, "fn": 2},
        ]
    )

    assert metrics["tp"] == 3
    assert metrics["fp"] == 1
    assert metrics["tn"] == 7
    assert metrics["fn"] == 3
    assert metrics["fpr"] == 0.125


def test_build_probe_does_not_launch_when_no_gate_candidate(tmp_path) -> None:
    records = [
        _record(1, [0.9, 0.8], "QA"),
        _record(1, [0.85, 0.2], "Summary"),
        _record(1, [0.3, 0.2], "Data2txt"),
        _record(0, [0.8, 0.7], "QA"),
        _record(0, [0.75, 0.4], "Summary"),
        _record(0, [0.2, 0.1], "Data2txt"),
        _record(1, [0.88, 0.81], "QA"),
        _record(0, [0.82, 0.79], "Summary"),
    ]
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps(records))

    packet = build_probe(
        cache_path=cache,
        n_splits=2,
        max_fpr=0.0,
        min_recall=1.0,
        min_f1=1.0,
    )

    assert packet["method"] == (
        "exploratory_stratified_cv_on_test_cache_not_promotion_evidence"
    )
    assert packet["decision"] == "do_not_launch_jarvis_from_reranker_probe"
    assert packet["best_gate_candidate"] is None


def test_evaluate_calibration_to_test_uses_separate_threshold_source() -> None:
    calibration = [
        _record(1, [0.95, 0.9], "QA"),
        _record(1, [0.88, 0.86], "Summary"),
        _record(0, [0.2, 0.1], "QA"),
        _record(0, [0.3, 0.1], "Data2txt"),
    ]
    test = [
        _record(1, [0.93, 0.9], "QA"),
        _record(0, [0.25, 0.1], "Data2txt"),
    ]

    rows = evaluate_calibration_to_test(
        calibration,
        test,
        include_metadata=False,
        max_fpr=0.0,
        min_recall=0.5,
        min_f1=0.5,
    )

    assert rows
    assert all("selected_threshold" in row for row in rows)
    assert all(row["folds"] == [] for row in rows)
    assert rows[0]["aggregate"]["tp"] + rows[0]["aggregate"]["fn"] == 1


def test_build_probe_records_split_cache_paths(tmp_path) -> None:
    calibration = [
        _record(1, [0.95, 0.9], "QA"),
        _record(1, [0.88, 0.86], "Summary"),
        _record(0, [0.2, 0.1], "QA"),
        _record(0, [0.3, 0.1], "Data2txt"),
    ]
    test = [
        _record(1, [0.93, 0.9], "QA"),
        _record(0, [0.25, 0.1], "Data2txt"),
    ]
    calibration_cache = tmp_path / "calibration.json"
    test_cache = tmp_path / "test.json"
    calibration_cache.write_text(json.dumps(calibration))
    test_cache.write_text(json.dumps(test))

    packet = build_probe(
        cache_path=test_cache,
        calibration_cache_path=calibration_cache,
        test_cache_path=test_cache,
        max_fpr=0.0,
        min_recall=0.5,
        min_f1=0.5,
    )

    assert (
        packet["method"]
        == "calibration_cache_threshold_selection_then_test_cache_evaluation"
    )
    assert packet["calibration_cache_path"] == str(calibration_cache)
    assert packet["test_cache_path"] == str(test_cache)
