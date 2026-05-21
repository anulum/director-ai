# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — DatasetTypeClassifier Tests
"""Multi-angle tests for dataset type meta-classifier.

Covers: feature extraction, model training, prediction, save/load,
AdaBoost classifier, pipeline integration with adaptive thresholds,
and performance documentation.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest

from director_ai.core.meta_classifier import (
    DatasetTypeClassifier,
    MetaClassifier,
    _word_overlap,
    extract_features,
    extract_text_features,
)


class _IdentityScaler:
    def transform(self, x):
        self.last_x = x
        return x


class _DeterministicClassifier:
    def __init__(self, *, probabilities, prediction=1):
        self.probabilities = np.array(probabilities, dtype=float)
        self.prediction = prediction
        self.last_proba_x = None
        self.last_predict_x = None

    def predict_proba(self, x):
        self.last_proba_x = x
        return np.array([self.probabilities], dtype=float)

    def predict(self, x):
        self.last_predict_x = x
        return np.array([self.prediction])


def _write_bundle(tmp_path, bundle):
    path = tmp_path / "meta_bundle.pkl"
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    return str(path)


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

    def test_rust_overlap_non_runtime_exception_falls_back_to_python(
        self, monkeypatch
    ):
        import director_ai.core.meta_classifier as meta_mod

        monkeypatch.setattr(meta_mod, "_RUST_META", True)

        def _boom(_a, _b):
            raise ValueError("ffi fail")

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


def _make_binary_bundle(tmp_path):
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    cols = ["nli_score", "confidence", "premise_len", "hypothesis_len"]
    x_train = np.array(
        [
            [0.8, 0.9, 50, 20],
            [0.2, 0.7, 60, 15],
            [0.9, 0.95, 40, 25],
            [0.1, 0.6, 70, 10],
        ]
    )
    y = np.array([1, 0, 1, 0])
    scaler = StandardScaler().fit(x_train)
    clf = RandomForestClassifier(n_estimators=5, random_state=42).fit(
        scaler.transform(x_train), y
    )
    path = str(tmp_path / "test_binary.pkl")
    with open(path, "wb") as f:
        pickle.dump(
            {
                "classifier": clf,
                "scaler": scaler,
                "feature_cols": cols,
                "mode": "binary",
            },
            f,
        )
    return path


def _make_dataset_type_bundle(tmp_path):
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    cols = ["premise_len", "hypothesis_len"]
    x_train = np.array([[100, 20], [200, 30], [50, 50], [300, 10], [150, 25], [60, 40]])
    y = np.array([0, 0, 1, 1, 2, 2])
    scaler = StandardScaler().fit(x_train)
    clf = RandomForestClassifier(n_estimators=5, random_state=42).fit(
        scaler.transform(x_train), y
    )
    path = str(tmp_path / "test_ds.pkl")
    thresholds = {"DatasetA": 0.3, "DatasetB": 0.6, "DatasetC": 0.5}
    with open(path, "wb") as f:
        pickle.dump(
            {
                "classifier": clf,
                "scaler": scaler,
                "feature_cols": cols,
                "mode": "dataset_type",
                "label_names": ["DatasetA", "DatasetB", "DatasetC"],
                "dataset_thresholds": thresholds,
                "confidence_gate": 0.3,
            },
            f,
        )
    return path


class TestDatasetTypeClassifier:
    def test_inconsistent_sklearn_version_warning_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "director_ai.core.meta_classifier.InconsistentVersionWarning",
            UserWarning,
        )
        path = _write_bundle(
            tmp_path,
            {
                "classifier": _DeterministicClassifier(
                    probabilities=[0.5, 0.5],
                    prediction=1,
                ),
                "scaler": _IdentityScaler(),
                "feature_cols": ["nli_score", "confidence"],
            },
        )

        def _warn_then_load(_raw):
            warnings.warn(
                "sklearn version mismatch",
                category=UserWarning,
                stacklevel=1,
            )
            return {
                "classifier": _DeterministicClassifier(
                    probabilities=[0.5, 0.5],
                    prediction=1,
                ),
                "scaler": _IdentityScaler(),
                "feature_cols": ["nli_score", "confidence"],
            }

        monkeypatch.setattr("director_ai.core.meta_classifier.pickle.loads", _warn_then_load)

        with pytest.raises(ValueError, match="Incompatible sklearn artefact"):
            DatasetTypeClassifier(path)

    def test_invalid_bundle_rejected_with_path_context(self, tmp_path):
        path = _write_bundle(tmp_path, {"scaler": _IdentityScaler()})

        with pytest.raises(ValueError, match="missing 'classifier' key") as exc_info:
            DatasetTypeClassifier(path)

        assert path in str(exc_info.value)

    def test_binary_predict_uses_configured_feature_columns_and_probability(
        self,
        tmp_path,
    ):
        scaler = _IdentityScaler()
        classifier = _DeterministicClassifier(probabilities=[0.25, 0.75], prediction=1)
        path = _write_bundle(
            tmp_path,
            {
                "classifier": classifier,
                "scaler": scaler,
                "feature_cols": ["nli_score", "confidence", "chunk_count"],
            },
        )

        runtime = DatasetTypeClassifier(path)
        supported, probability = runtime.predict(
            "Alice verified the claim.",
            "Alice verified the claim?",
            nli_score=0.91,
            confidence=0.84,
            chunk_count=3,
        )

        assert supported is True
        assert probability == pytest.approx(0.75)
        assert runtime._scaler.last_x.tolist() == [[0.91, 0.84, 3]]
        assert runtime._clf.last_proba_x.tolist() == [[0.91, 0.84, 3]]
        assert runtime._clf.last_predict_x.tolist() == [[0.91, 0.84, 3]]

    def test_binary_predict_returns_false_for_negative_prediction(self, tmp_path):
        path = _write_bundle(
            tmp_path,
            {
                "classifier": _DeterministicClassifier(
                    probabilities=[0.8, 0.2],
                    prediction=0,
                ),
                "scaler": _IdentityScaler(),
                "feature_cols": ["nli_score", "confidence"],
                "mode": "binary",
            },
        )

        supported, probability = DatasetTypeClassifier(path).predict(
            "The answer is unsupported.",
            "The answer is supported.",
            nli_score=0.2,
            confidence=0.6,
        )

        assert supported is False
        assert probability == pytest.approx(0.2)

    def test_dataset_threshold_returns_named_threshold_and_confidence(self, tmp_path):
        path = _write_bundle(
            tmp_path,
            {
                "classifier": _DeterministicClassifier(
                    probabilities=[0.1, 0.7, 0.2],
                    prediction=1,
                ),
                "scaler": _IdentityScaler(),
                "feature_cols": ["premise_len", "hypothesis_len"],
                "mode": "dataset_type",
                "label_names": ["fact", "legal", "medical"],
                "dataset_thresholds": {"legal": 0.62, "medical": 0.58},
                "confidence_gate": 0.5,
            },
        )

        threshold, confidence = DatasetTypeClassifier(path).predict_threshold(
            "A" * 40,
            "B" * 10,
        )

        assert threshold == pytest.approx(0.62)
        assert confidence == pytest.approx(0.7)

    def test_dataset_threshold_without_label_names_falls_back_with_confidence(
        self,
        tmp_path,
    ):
        path = _write_bundle(
            tmp_path,
            {
                "classifier": _DeterministicClassifier(
                    probabilities=[0.6, 0.4],
                    prediction=0,
                ),
                "scaler": _IdentityScaler(),
                "feature_cols": ["premise_len", "hypothesis_len"],
                "mode": "dataset_type",
                "dataset_thresholds": {"fact": 0.7},
                "confidence_gate": 0.5,
            },
        )

        threshold, confidence = DatasetTypeClassifier(path).predict_threshold(
            "premise",
            "hypothesis",
        )

        assert threshold is None
        assert confidence == pytest.approx(0.6)

    def test_dataset_threshold_missing_dataset_name_returns_none_with_confidence(
        self,
        tmp_path,
    ):
        path = _write_bundle(
            tmp_path,
            {
                "classifier": _DeterministicClassifier(
                    probabilities=[0.2, 0.8],
                    prediction=1,
                ),
                "scaler": _IdentityScaler(),
                "feature_cols": ["premise_len", "hypothesis_len"],
                "mode": "dataset_type",
                "label_names": ["known", "unknown"],
                "dataset_thresholds": {"known": 0.7},
                "confidence_gate": 0.5,
            },
        )

        threshold, confidence = DatasetTypeClassifier(path).predict_threshold(
            "premise",
            "hypothesis",
        )

        assert threshold is None
        assert confidence == pytest.approx(0.8)

    def test_binary_predict(self, tmp_path):
        path = _make_binary_bundle(tmp_path)
        clf = DatasetTypeClassifier(path)
        supported, prob = clf.predict("The sky is blue.", "The sky is blue.", 0.9, 0.95)
        assert isinstance(supported, bool)
        assert 0.0 <= prob <= 1.0

    def test_binary_predict_threshold_returns_none(self, tmp_path):
        path = _make_binary_bundle(tmp_path)
        clf = DatasetTypeClassifier(path)
        threshold, conf = clf.predict_threshold("premise", "hypothesis")
        assert threshold is None
        assert conf == 0.0

    def test_dataset_type_predict_threshold(self, tmp_path):
        path = _make_dataset_type_bundle(tmp_path)
        clf = DatasetTypeClassifier(path)
        threshold, conf = clf.predict_threshold("A" * 100, "B" * 20)
        # Should return a threshold from the dict or None if below gate
        if threshold is not None:
            assert threshold in (0.3, 0.5, 0.6)
        assert 0.0 <= conf <= 1.0

    def test_low_confidence_returns_none(self, tmp_path):
        path = _make_dataset_type_bundle(tmp_path)
        clf = DatasetTypeClassifier(path)
        clf._confidence_gate = 0.99  # force gate too high
        threshold, conf = clf.predict_threshold("A" * 100, "B" * 20)
        assert threshold is None

    def test_backward_compat_alias(self):
        assert MetaClassifier is DatasetTypeClassifier
