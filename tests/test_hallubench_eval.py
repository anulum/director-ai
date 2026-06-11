# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — HalluBench benchmark harness tests

from __future__ import annotations

import json

import pytest

from benchmarks import hallubench_eval as hb


def _sample_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "question_id": "Em00001",
        "two_images": "yes",
        "is_temporal": "yes",
        "img1_path": "emergency_images/pre.png",
        "img1_type": "RGB",
        "img2_path": "emergency_images/post.png",
        "img2_type": "RGB",
        "application": "emergency",
        "sub_application": "turkey_earthquake",
        "task_type": "counting",
        "output_form": "short",
        "question": "How many damaged buildings are visible?",
        "ground_truth": "12",
        "source_dataset": "disasterM3",
        "original_id": 42,
        "original_q": "Count damaged buildings.",
        "original_qtype": "Basic Counting",
    }
    row.update(overrides)
    return row


def test_normalise_sample_keeps_paths_not_image_payloads() -> None:
    sample = hb.normalise_sample(_sample_row())

    assert sample.question_id == "Em00001"
    assert sample.application == "emergency"
    assert sample.is_temporal is True
    assert sample.image_refs == (
        hb.ImageRef(path="emergency_images/pre.png", modality="RGB", role="img1"),
        hb.ImageRef(path="emergency_images/post.png", modality="RGB", role="img2"),
    )
    assert "image" not in sample.to_result_dict()
    assert "bytes" not in repr(sample.to_result_dict()).lower()


def test_prediction_metrics_handle_numeric_short_answer() -> None:
    sample = hb.normalise_sample(_sample_row(ground_truth="12 buildings"))

    correct = hb.evaluate_prediction(sample, "There are 12 damaged buildings.")
    wrong = hb.evaluate_prediction(sample, "There are 14 damaged buildings.")

    assert correct.exact_match is False
    assert correct.numeric_match is True
    assert correct.passed is True
    assert wrong.numeric_match is False
    assert wrong.passed is False


def test_prediction_metrics_handle_long_report_token_overlap() -> None:
    sample = hb.normalise_sample(
        _sample_row(
            question_id="Ur10001",
            two_images="no",
            is_temporal="no",
            img2_path="",
            img2_type="",
            application="urban",
            sub_application="",
            task_type="report",
            output_form="long",
            ground_truth="Residential buildings are dense and roads are visible.",
        )
    )

    metric = hb.evaluate_prediction(
        sample,
        "Dense residential buildings are visible near roads.",
        long_answer_f1_threshold=0.5,
    )

    assert metric.token_f1 >= 0.5
    assert metric.passed is True


def test_run_benchmark_from_predictions_filters_and_aggregates(tmp_path) -> None:
    rows = [
        _sample_row(question_id="Em00001", ground_truth="12"),
        _sample_row(
            question_id="Ur00001",
            two_images="no",
            is_temporal="no",
            img2_path="",
            img2_type="",
            application="urban",
            task_type="recognition",
            output_form="short",
            ground_truth="urban",
        ),
    ]
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "\n".join(
            [
                json.dumps({"question_id": "Em00001", "prediction": "12"}),
                json.dumps({"question_id": "Ur00001", "prediction": "rural"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = hb.run_hallubench_benchmark(
        rows=rows,
        predictions_jsonl=predictions,
        model_id="unit-test-vlm",
    )

    assert result["benchmark"] == "HalluBench"
    assert result["benchmark_evidence"] is False
    assert result["dataset"]["source"] == "AuwAuwAuw/HalluBench"
    assert result["model_id"] == "unit-test-vlm"
    assert result["overall"]["total"] == 2
    assert result["overall"]["passed"] == 1
    assert result["overall"]["accuracy"] == pytest.approx(0.5)
    assert result["by_application"]["emergency"]["accuracy"] == pytest.approx(1.0)
    assert result["by_application"]["urban"]["accuracy"] == pytest.approx(0.0)
    assert all("ground_truth" not in row for row in result["per_sample"])
    assert all("prediction" not in row for row in result["per_sample"])


def test_missing_prediction_is_counted_separately(tmp_path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("", encoding="utf-8")

    result = hb.run_hallubench_benchmark(
        rows=[_sample_row()],
        predictions_jsonl=predictions,
    )

    assert result["overall"]["total"] == 1
    assert result["overall"]["missing_predictions"] == 1
    assert result["overall"]["accuracy"] == pytest.approx(0.0)


def test_loader_reports_gated_access_error(monkeypatch) -> None:
    def _raise(*args: object, **kwargs: object) -> object:
        raise PermissionError("gated dataset")

    monkeypatch.setattr(hb, "_hf_load_dataset", _raise)

    with pytest.raises(hb.HalluBenchAccessError, match="gated"):
        hb.load_hallubench_rows(split="train")
