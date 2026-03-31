# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Meta-Classifier Runtime (Packaged)

"""Lightweight meta-classifier for production NLI threshold adaptation.

Loads a pre-trained sklearn logistic regression model from a pickle
bundle and predicts per-input thresholds based on text features.
"""

from __future__ import annotations

import logging
import pickle  # nosec B403 — intentional; runtime warns on untrusted paths
import re

import numpy as np

logger = logging.getLogger("DirectorAI.MetaClassifier")

NEGATION_WORDS = frozenset(
    {
        "not",
        "no",
        "never",
        "neither",
        "nobody",
        "nothing",
        "nowhere",
        "nor",
        "cannot",
        "can't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "won't",
        "wouldn't",
        "shouldn't",
        "couldn't",
        "doesn't",
        "don't",
        "didn't",
        "hasn't",
        "haven't",
        "hadn't",
        "without",
        "false",
    }
)

FEATURE_COLS = [
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
]

TEXT_FEATURE_COLS = [
    c
    for c in FEATURE_COLS
    if c not in ("nli_score", "confidence", "score_distance_from_half")
]


def extract_features(
    premise: str,
    hypothesis: str,
    nli_score: float,
    confidence: float,
    chunk_count: int = 1,
) -> dict:
    """Extract meta-classifier features from a scored pair."""
    h_words = hypothesis.lower().split()
    p_words = premise.lower().split()
    h_set = set(h_words)
    p_set = set(p_words)

    return {
        "nli_score": nli_score,
        "confidence": confidence,
        "premise_len": len(premise),
        "hypothesis_len": len(hypothesis),
        "premise_word_count": len(p_set),
        "hypothesis_word_count": len(h_set),
        "word_overlap": len(p_set & h_set) / max(len(p_set | h_set), 1),
        "has_negation_premise": int(bool(p_set & NEGATION_WORDS)),
        "has_negation_hypothesis": int(bool(h_set & NEGATION_WORDS)),
        "negation_asymmetry": int(
            bool(p_set & NEGATION_WORDS) != bool(h_set & NEGATION_WORDS),
        ),
        "chunk_count": chunk_count,
        "score_distance_from_half": abs(nli_score - 0.5),
        "has_question_mark": int("?" in hypothesis),
        "num_entities_premise": len(re.findall(r"[A-Z][a-z]+", premise)),
        "num_entities_hypothesis": len(re.findall(r"[A-Z][a-z]+", hypothesis)),
        "len_ratio": len(premise) / max(len(hypothesis), 1),
        "premise_sent_count": (
            premise.count(".") + premise.count("!") + premise.count("?")
        ),
        "hypothesis_sent_count": (
            hypothesis.count(".") + hypothesis.count("!") + hypothesis.count("?")
        ),
        "avg_word_len_premise": (sum(len(w) for w in p_words) / max(len(p_words), 1)),
        "avg_word_len_hypothesis": (
            sum(len(w) for w in h_words) / max(len(h_words), 1)
        ),
    }


def extract_text_features(premise: str, hypothesis: str) -> dict:
    """Extract text-only features (no NLI score needed)."""
    return extract_features(premise, hypothesis, nli_score=0.0, confidence=0.0)


class DatasetTypeClassifier:
    """Logistic regression that predicts dataset type for threshold selection.

    Loads a trained sklearn model bundle and either:
    - (binary mode) predicts support/hallucination directly, or
    - (dataset_type mode) predicts which dataset distribution the input
      resembles, then selects a per-dataset NLI threshold.
    """

    def __init__(self, model_path: str):
        import hashlib

        logger.warning(
            "Loading pickle model from %s — ensure this file is trusted",
            model_path,
        )
        with open(model_path, "rb") as f:
            raw = f.read()
        sha = hashlib.sha256(raw).hexdigest()[:16]
        logger.info("Model SHA256 prefix: %s (%d bytes)", sha, len(raw))
        bundle = pickle.loads(raw)  # nosec B301 — warned above; hash logged for auditability
        if not isinstance(bundle, dict) or "classifier" not in bundle:
            raise ValueError(
                f"Invalid model bundle at {model_path}: missing 'classifier' key"
            )
        self._clf = bundle["classifier"]
        self._scaler = bundle["scaler"]
        self._feature_cols = bundle["feature_cols"]
        self._mode = bundle.get("mode", "binary")
        self._label_names = bundle.get("label_names")
        self._dataset_thresholds = bundle.get("dataset_thresholds")
        self._confidence_gate = bundle.get("confidence_gate", 0.5)

    def predict(
        self,
        premise: str,
        hypothesis: str,
        nli_score: float,
        confidence: float,
        chunk_count: int = 1,
    ) -> tuple[bool, float]:
        """Predict support/hallucination with probability."""
        feat = extract_features(premise, hypothesis, nli_score, confidence, chunk_count)
        x = np.array([[feat[c] for c in self._feature_cols]])
        x_scaled = self._scaler.transform(x)
        prob = self._clf.predict_proba(x_scaled)[0]
        pred = int(self._clf.predict(x_scaled)[0])
        return bool(pred == 1), float(prob[1])

    def predict_threshold(
        self,
        premise: str,
        hypothesis: str,
    ) -> tuple[float | None, float]:
        """Predict per-dataset NLI threshold via dataset-type classification.

        Returns (threshold_or_None, confidence). If confidence is below
        the gate, threshold is None and the caller should fall back to
        per-task-type thresholds.
        """
        if self._mode != "dataset_type" or not self._dataset_thresholds:
            return None, 0.0

        feat = extract_text_features(premise, hypothesis)
        x = np.array([[feat[c] for c in self._feature_cols]])
        x_scaled = self._scaler.transform(x)
        probs = self._clf.predict_proba(x_scaled)[0]
        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])

        if conf < self._confidence_gate:
            return None, conf

        if not self._label_names:
            return None, conf
        ds_name = self._label_names[pred_idx]
        threshold = self._dataset_thresholds.get(ds_name)
        return threshold, conf


MetaClassifier = DatasetTypeClassifier
