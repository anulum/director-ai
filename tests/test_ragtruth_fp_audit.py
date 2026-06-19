# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — RAGTruth false-positive audit tests

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("torch")

from training.ragtruth_fp_audit import (
    apply_rule,
    build_audit,
    token_features,
    write_markdown,
)


def _record(label: int, probs: list[float]) -> dict:
    return {
        "label": label,
        "resp_probs": probs,
        "task_type": "QA",
        "context_chars": 128,
        "response_chars": 64,
        "context_tokens": 16,
        "response_tokens": len(probs),
        "row_index": label + len(probs),
    }


def _result(best: dict) -> dict:
    return {
        "model_dir": "/tmp/model",
        "model_sha256": "abc123",
        "best": {
            "f1": best["f1"],
            "precision": best["precision"],
            "recall": best["recall"],
            "balanced_accuracy": best["balanced_accuracy"],
            "fpr": best["fpr"],
            "tp": best["tp"],
            "fp": best["fp"],
            "tn": best["tn"],
            "fn": best["fn"],
            "p": best["p"],
            "k": best["k"],
        },
        "diagnostics": {
            "by_task_type": [
                {"group": "QA", "n": 3, "fp": 2, "fpr": 0.5, "f1": 0.4},
                {"group": "Data2txt", "n": 2, "fp": 1, "fpr": 1.0, "f1": 0.0},
            ],
            "by_response_token_bucket": [
                {"group": "0-128", "n": 4, "fp": 1, "fpr": 0.25, "f1": 0.5}
            ],
            "by_context_token_bucket": [
                {"group": ">1024", "n": 4, "fp": 2, "fpr": 0.67, "f1": 0.5}
            ],
            "by_context_char_bucket": [
                {"group": "8001-16000", "n": 4, "fp": 2, "fpr": 0.67, "f1": 0.5}
            ],
        },
    }


def test_token_features_and_apply_rule_density_gate() -> None:
    record = _record(0, [0.9, 0.8, 0.1, 0.05])

    features = token_features(record, p=0.7)
    flags = apply_rule(
        [record],
        p=0.7,
        k=2,
        min_max_probability=0.85,
        max_threshold_density=0.4,
    )

    assert features["tokens_at_threshold"] == 2.0
    assert features["threshold_density"] == 0.5
    assert features["max_token_probability"] == 0.9
    assert flags.tolist() == [False]


def test_build_audit_marks_low_fpr_calibration_not_smoke_ready(tmp_path) -> None:
    records = [
        _record(1, [0.99, 0.95]),
        _record(1, [0.99, 0.94]),
        _record(1, [0.4, 0.35]),
        _record(0, [0.6, 0.5]),
        _record(0, [0.55, 0.2]),
        _record(0, [0.1, 0.05]),
    ]
    cache = tmp_path / "token_eval_probs.json"
    cache.write_text(json.dumps(records))
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            _result(
                {
                    "f1": 0.75,
                    "precision": 0.75,
                    "recall": 0.75,
                    "balanced_accuracy": 0.8,
                    "fpr": 0.1,
                    "tp": 3,
                    "fp": 1,
                    "tn": 2,
                    "fn": 0,
                    "p": 0.7,
                    "k": 1,
                }
            )
        )
    )

    packet = build_audit(
        cache_path=cache,
        result_paths=[result],
        max_fpr=0.0,
        min_f1=0.81,
    )

    selected = packet["calibration"]["selected_low_fpr"]
    assert selected is not None
    assert selected["fpr"] == 0.0
    assert selected["f1"] < 0.81
    assert packet["calibration"]["decision"] == (
        "do_not_launch_jarvis_from_posthoc_calibration"
    )
    assert packet["worst_false_positive_groups"]["task_type"][0]["group"] == "QA"


def test_write_markdown_includes_decision(tmp_path) -> None:
    packet = {
        "result_summaries": [
            {
                "path": "/tmp/result.json",
                "f1": 0.75,
                "precision": 0.8,
                "recall": 0.7,
                "fpr": 0.08,
                "p": 0.85,
                "k": 1,
            }
        ],
        "worst_false_positive_groups": {"task_type": []},
        "calibration": {
            "selected_overall": {
                "f1": 0.75,
                "precision": 0.8,
                "recall": 0.7,
                "fpr": 0.08,
                "p": 0.85,
                "k": 1,
            },
            "selected_low_fpr": None,
            "decision": "do_not_launch_jarvis_from_posthoc_calibration",
        },
    }
    output = tmp_path / "audit.md"

    write_markdown(packet, output)

    text = output.read_text()
    assert "RAGTruth False-Positive Audit and Calibration" in text
    assert "do_not_launch_jarvis_from_posthoc_calibration" in text


def test_apply_rule_returns_bool_array() -> None:
    flags = apply_rule(
        [_record(1, [0.9]), _record(0, [0.1])],
        p=0.8,
        k=1,
        min_max_probability=0.0,
        max_threshold_density=1.0,
    )

    assert flags.dtype == np.bool_
    assert flags.tolist() == [True, False]
