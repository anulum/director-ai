# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Â© Concepts 1996â€“2026 Miroslav Ĺ otek. All rights reserved.
# Â© Code 2020â€“2026 Miroslav Ĺ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI â€” DatasetTypeClassifier Tests

from __future__ import annotations

import pickle

import numpy as np
import pytest

from director_ai.core.meta_classifier import (
    DatasetTypeClassifier,
    MetaClassifier,
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


def _make_binary_bundle(tmp_path):
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
