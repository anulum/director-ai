# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — DatasetTypeClassifier Tests
"""Multi-angle tests for the dataset-type meta-classifier.

Covers feature extraction, the Rust word-overlap delegation, and the pickle-free
JSON model loader: NumPy reproduction of sklearn predict_proba (binary sigmoid
and multinomial softmax), per-dataset threshold selection, the confidence gate,
artefact validation errors, and the bundled artefact.
"""

from __future__ import annotations

import json
import math

import pytest

from director_ai.core.meta_classifier import (
    DatasetTypeClassifier,
    MetaClassifier,
    _word_overlap,
    extract_features,
    extract_text_features,
)


class TestExtractFeatures:
    def test_all_keys_present(self):
        feat = extract_features("The sky is blue.", "Blue sky.", 0.8, 0.9)
        expected = {
            "nli_score",
            "confidence",
            "premise_len",
            "hypothesis_len",
            "premise_word_count",
            "hypothesis_word_count",
            "word_overlap",
            "has_negation_premise",
            "has_negation_hypothesis",
            "negation_asymmetry",
            "chunk_count",
            "score_distance_from_half",
            "has_question_mark",
            "num_entities_premise",
            "num_entities_hypothesis",
            "len_ratio",
            "premise_sent_count",
            "hypothesis_sent_count",
            "avg_word_len_premise",
            "avg_word_len_hypothesis",
        }
        assert set(feat.keys()) == expected

    def test_empty_strings(self):
        feat = extract_features("", "", 0.5, 0.5)
        assert feat["premise_len"] == 0
        assert feat["hypothesis_len"] == 0
        assert feat["word_overlap"] == 0.0
        assert feat["avg_word_len_premise"] == 0.0

    def test_negation_detection(self):
        feat = extract_features("This is not correct.", "This is correct.", 0.3, 0.7)
        assert feat["has_negation_premise"] == 1
        assert feat["has_negation_hypothesis"] == 0
        assert feat["negation_asymmetry"] == 1

    def test_question_mark(self):
        feat = extract_features("Context.", "Is this correct?", 0.5, 0.5)
        assert feat["has_question_mark"] == 1

    def test_entity_count(self):
        feat = extract_features(
            "Paris and London are cities.", "Berlin is too.", 0.5, 0.5
        )
        assert feat["num_entities_premise"] >= 2
        assert feat["num_entities_hypothesis"] >= 1

    def test_len_ratio(self):
        feat = extract_features("A" * 100, "B" * 10, 0.5, 0.5)
        assert feat["len_ratio"] == pytest.approx(10.0)

    def test_text_features_zeroes_nli(self):
        feat = extract_text_features("premise", "hypothesis")
        assert feat["nli_score"] == 0.0
        assert feat["confidence"] == 0.0


class TestWordOverlapRustDelegation:
    def test_python_fallback_overlap(self, monkeypatch):
        import director_ai.core.meta_classifier as meta_mod

        monkeypatch.setattr(meta_mod, "_RUST_META", False)
        assert _word_overlap("alpha beta", "alpha gamma") == pytest.approx(1.0 / 3.0)
        assert _word_overlap("", "alpha gamma") == 0.0

    def test_rust_overlap_delegation(self, monkeypatch):
        import director_ai.core.meta_classifier as meta_mod

        monkeypatch.setattr(meta_mod, "_RUST_META", True)
        monkeypatch.setattr(
            meta_mod,
            "rust_word_overlap",
            lambda text_a, text_b: 0.75 if text_a and text_b else 0.0,
            raising=False,
        )
        assert _word_overlap("alpha", "beta") == 0.75

    def test_rust_overlap_exception_falls_back_to_python(self, monkeypatch):
        import director_ai.core.meta_classifier as meta_mod

        monkeypatch.setattr(meta_mod, "_RUST_META", True)

        def _boom(_a, _b):
            raise RuntimeError("ffi fail")

        monkeypatch.setattr(meta_mod, "rust_word_overlap", _boom, raising=False)
        assert _word_overlap("alpha beta", "alpha gamma") == pytest.approx(1.0 / 3.0)

    def test_rust_overlap_non_runtime_exception_falls_back_to_python(self, monkeypatch):
        import director_ai.core.meta_classifier as meta_mod

        monkeypatch.setattr(meta_mod, "_RUST_META", True)

        def _boom(_a, _b):
            raise ValueError("ffi fail")

        monkeypatch.setattr(meta_mod, "rust_word_overlap", _boom, raising=False)
        assert _word_overlap("alpha beta", "alpha gamma") == pytest.approx(1.0 / 3.0)

    def test_rust_overlap_type_error_falls_back_to_python(self, monkeypatch):
        import director_ai.core.meta_classifier as meta_mod

        monkeypatch.setattr(meta_mod, "_RUST_META", True)

        def _boom(_a, _b):
            raise TypeError("ffi fail")

        monkeypatch.setattr(meta_mod, "rust_word_overlap", _boom, raising=False)
        assert _word_overlap("alpha beta", "alpha gamma") == pytest.approx(1.0 / 3.0)

    def test_extract_features_uses_overlap_helper(self, monkeypatch):
        import director_ai.core.meta_classifier as meta_mod

        monkeypatch.setattr(
            meta_mod,
            "_word_overlap",
            lambda premise, hypothesis: 0.42,
            raising=True,
        )
        feat = extract_features("Premise text", "Hypothesis text", 0.5, 0.6)
        assert feat["word_overlap"] == 0.42


def _write_json(tmp_path, payload, name="model.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _binary_payload():
    # 2-feature binary logistic regression with an identity scaler.
    return {
        "format": "director.dataset_type_classifier.v1",
        "model": "multinomial_logistic_regression",
        "mode": "binary",
        "feature_cols": ["nli_score", "confidence"],
        "classes": [0, 1],
        "coef": [[2.0, 1.0]],
        "intercept": [0.5],
        "scaler_mean": [0.0, 0.0],
        "scaler_scale": [1.0, 1.0],
        "confidence_gate": 0.5,
    }


def _dataset_type_payload():
    # 2-feature, 3-class multinomial with an identity scaler.
    return {
        "format": "director.dataset_type_classifier.v1",
        "model": "multinomial_logistic_regression",
        "mode": "dataset_type",
        "feature_cols": ["premise_len", "hypothesis_len"],
        "classes": [0, 1, 2],
        "coef": [[0.01, 0.0], [0.0, 0.01], [0.0, 0.0]],
        "intercept": [0.0, 0.0, 0.0],
        "scaler_mean": [0.0, 0.0],
        "scaler_scale": [1.0, 1.0],
        "label_names": ["fact", "legal", "medical"],
        "dataset_thresholds": {"fact": 0.30, "legal": 0.62, "medical": 0.58},
        "confidence_gate": 0.3,
    }


class TestArtefactValidation:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DatasetTypeClassifier(str(tmp_path / "absent.json"))

    def test_invalid_json_raises(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid classifier JSON"):
            DatasetTypeClassifier(str(path))

    def test_wrong_format_raises(self, tmp_path):
        path = _write_json(tmp_path, {"format": "something.else"})
        with pytest.raises(ValueError, match="Unsupported classifier artefact"):
            DatasetTypeClassifier(path)

    def test_incomplete_artefact_raises(self, tmp_path):
        payload = _binary_payload()
        del payload["coef"]
        path = _write_json(tmp_path, payload)
        with pytest.raises(ValueError, match="Incomplete classifier artefact"):
            DatasetTypeClassifier(path)


class TestBinaryPrediction:
    def test_predict_matches_sigmoid(self, tmp_path):
        clf = DatasetTypeClassifier(_write_json(tmp_path, _binary_payload()))
        # nli_score=0.91, confidence=0.84 -> score = 2*0.91 + 1*0.84 + 0.5 = 3.16
        supported, prob = clf.predict(
            "Alice verified the claim.",
            "Alice verified the claim.",
            nli_score=0.91,
            confidence=0.84,
        )
        expected = 1.0 / (1.0 + math.exp(-3.16))
        assert prob == pytest.approx(expected, abs=1e-9)
        assert supported is True

    def test_predict_false_for_negative_score(self, tmp_path):
        clf = DatasetTypeClassifier(_write_json(tmp_path, _binary_payload()))
        supported, prob = clf.predict("x", "y", nli_score=-1.0, confidence=0.0)
        # score = 2*(-1.0) + 0.5 = -1.5 -> sigmoid < 0.5 -> argmax = class 0
        assert supported is False
        assert prob < 0.5

    def test_binary_predict_threshold_returns_none(self, tmp_path):
        clf = DatasetTypeClassifier(_write_json(tmp_path, _binary_payload()))
        threshold, conf = clf.predict_threshold("premise", "hypothesis")
        assert threshold is None
        assert conf == 0.0

    def test_proba_sums_to_one(self, tmp_path):
        clf = DatasetTypeClassifier(_write_json(tmp_path, _binary_payload()))
        feat = extract_features("p", "h", 0.3, 0.7)
        proba = clf._predict_proba(feat)
        assert proba.sum() == pytest.approx(1.0)
        assert len(proba) == 2


class TestDatasetTypePrediction:
    def test_named_threshold_and_confidence(self, tmp_path):
        clf = DatasetTypeClassifier(_write_json(tmp_path, _dataset_type_payload()))
        # Long premise drives class 0 (fact); confidence above the 0.3 gate.
        threshold, conf = clf.predict_threshold("A" * 400, "B" * 5)
        assert threshold == pytest.approx(0.30)
        assert conf >= 0.3

    def test_softmax_sums_to_one(self, tmp_path):
        clf = DatasetTypeClassifier(_write_json(tmp_path, _dataset_type_payload()))
        proba = clf._predict_proba(extract_text_features("A" * 50, "B" * 50))
        assert proba.sum() == pytest.approx(1.0)
        assert len(proba) == 3

    def test_below_gate_returns_none(self, tmp_path):
        clf = DatasetTypeClassifier(_write_json(tmp_path, _dataset_type_payload()))
        clf._confidence_gate = 0.99
        threshold, conf = clf.predict_threshold("A" * 50, "B" * 50)
        assert threshold is None
        assert 0.0 <= conf <= 1.0

    def test_no_label_names_falls_back(self, tmp_path):
        payload = _dataset_type_payload()
        payload["label_names"] = None
        clf = DatasetTypeClassifier(_write_json(tmp_path, payload))
        threshold, conf = clf.predict_threshold("A" * 400, "B" * 5)
        assert threshold is None
        assert conf >= 0.0

    def test_missing_dataset_name_returns_none(self, tmp_path):
        payload = _dataset_type_payload()
        payload["dataset_thresholds"] = {"legal": 0.62}  # 'fact' (class 0) absent
        clf = DatasetTypeClassifier(_write_json(tmp_path, payload))
        threshold, conf = clf.predict_threshold("A" * 400, "B" * 5)
        assert threshold is None
        assert conf >= 0.3

    def test_no_thresholds_returns_none(self, tmp_path):
        payload = _dataset_type_payload()
        payload["dataset_thresholds"] = None
        clf = DatasetTypeClassifier(_write_json(tmp_path, payload))
        threshold, conf = clf.predict_threshold("A" * 50, "B" * 50)
        assert threshold is None
        assert conf == 0.0


class TestBundledArtefact:
    def test_bundled_json_loads_and_predicts(self):
        from pathlib import Path

        import director_ai.core.scoring.meta_classifier as meta_mod

        bundled = (
            Path(meta_mod.__file__).parent.parent
            / "models"
            / "dataset_type_classifier.json"
        )
        clf = DatasetTypeClassifier(str(bundled))
        proba = clf._predict_proba(
            extract_text_features("The capital of France is Paris.", "Paris.")
        )
        assert proba.sum() == pytest.approx(1.0)
        assert len(proba) == 11
        threshold, conf = clf.predict_threshold(
            "The capital of France is Paris.", "Paris is the capital."
        )
        assert 0.0 <= conf <= 1.0
        if threshold is not None:
            assert 0.0 < threshold < 1.0

    def test_bundled_artefact_is_v1_format(self):
        from pathlib import Path

        import director_ai.core.scoring.meta_classifier as meta_mod

        bundled = (
            Path(meta_mod.__file__).parent.parent
            / "models"
            / "dataset_type_classifier.json"
        )
        payload = json.loads(bundled.read_text(encoding="utf-8"))
        assert payload["format"] == "director.dataset_type_classifier.v1"
        assert payload["mode"] == "dataset_type"


def test_backward_compat_alias():
    assert MetaClassifier is DatasetTypeClassifier


def test_sum_float_rust_and_python_paths(monkeypatch):
    import director_ai.core.scoring.meta_classifier as meta_mod

    # Accelerated path: force the Rust flag on and delegate to a stub so the
    # branch is exercised without depending on the compiled kernel.
    monkeypatch.setattr(meta_mod, "_RUST_META", True)
    calls = {"n": 0}

    def _sum(values: list[float]) -> float:
        calls["n"] += 1
        return sum(values)

    monkeypatch.setattr(meta_mod, "rust_sum_f64", _sum, raising=True)
    assert meta_mod._sum_float([1.0, 2.0, 3.5]) == pytest.approx(6.5)
    assert calls["n"] == 1
    # Pure-Python fallback when the kernel is unavailable.
    monkeypatch.setattr(meta_mod, "_RUST_META", False)
    assert meta_mod._sum_float([1.0, 2.0, 3.5]) == pytest.approx(6.5)
    assert meta_mod._sum_float([]) == 0.0
