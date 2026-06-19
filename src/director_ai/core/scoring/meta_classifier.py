# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Meta-Classifier Runtime (Packaged)

"""Lightweight meta-classifier for production NLI threshold adaptation.

Loads a pre-trained multinomial logistic-regression model from a pickle-free
JSON artefact and predicts per-input thresholds from text features. The model
reduces to a few float arrays (scaler mean/scale, coefficients, intercepts), so
prediction is reproduced exactly in NumPy with no scikit-learn or pickle
dependency at runtime; the JSON is produced by
``scripts/convert_classifier_to_json.py`` with a parity gate against sklearn.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np

from ..mandatory import mandatory_execution
from ..text_overlap import word_overlap

logger = logging.getLogger("DirectorAI.MetaClassifier")

_CLASSIFIER_FORMAT = "director.dataset_type_classifier.v1"

try:
    from backfire_kernel import rust_sum_f64

    _RUST_META = True
except ImportError:

    def rust_sum_f64(_values: list[float]) -> float:
        """Raise to signal the mandatory Rust sum accelerator is missing."""
        raise RuntimeError("backfire_kernel rust_sum_f64 is unavailable")

    _RUST_META = True


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


def _word_overlap(text_a: str, text_b: str) -> float:
    """Return lexical Jaccard overlap in ``[0, 1]``.

    Delegates to the shared measured-fast-path helper: pure Python below a large
    -input threshold (faster for these claim-sized inputs), the Rust kernel only
    above it. See :mod:`director_ai.core.text_overlap`.
    """
    return word_overlap(text_a, text_b, logger_name=__name__)


def extract_features(
    premise: str,
    hypothesis: str,
    nli_score: float,
    confidence: float,
    chunk_count: int = 1,
) -> dict[str, float]:
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
        "word_overlap": _word_overlap(premise, hypothesis),
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
        "avg_word_len_premise": (
            _sum_float([float(len(w)) for w in p_words]) / max(len(p_words), 1)
        ),
        "avg_word_len_hypothesis": (
            _sum_float([float(len(w)) for w in h_words]) / max(len(h_words), 1)
        ),
    }


def extract_text_features(premise: str, hypothesis: str) -> dict[str, float]:
    """Extract text-only features (no NLI score needed)."""
    return extract_features(premise, hypothesis, nli_score=0.0, confidence=0.0)


def _sum_float(values: list[float]) -> float:
    if _RUST_META:
        with mandatory_execution(__name__, component="mandatory accelerated path"):
            return float(rust_sum_f64(values))
    return sum(values)


class DatasetTypeClassifier:
    """Logistic regression that predicts dataset type for threshold selection.

    Loads a pickle-free JSON model artefact and either:
    - (binary mode) predicts support/hallucination directly, or
    - (dataset_type mode) predicts which dataset distribution the input
      resembles, then selects a per-dataset NLI threshold.
    """

    def __init__(self, model_path: str):
        import hashlib

        raw = Path(model_path).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()[:16]
        logger.info(
            "Loading classifier %s SHA256 prefix: %s (%d bytes)",
            model_path,
            sha,
            len(raw),
        )
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid classifier JSON at {model_path}: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("format") != _CLASSIFIER_FORMAT:
            raise ValueError(
                f"Unsupported classifier artefact at {model_path}: "
                f"expected format {_CLASSIFIER_FORMAT!r}"
            )
        try:
            self._coef = np.asarray(payload["coef"], dtype=float)
            self._intercept = np.asarray(payload["intercept"], dtype=float)
            self._scaler_mean = np.asarray(payload["scaler_mean"], dtype=float)
            self._scaler_scale = np.asarray(payload["scaler_scale"], dtype=float)
            self._feature_cols = list(payload["feature_cols"])
        except KeyError as exc:
            raise ValueError(
                f"Incomplete classifier artefact at {model_path}: missing {exc}"
            ) from exc
        self._mode = payload.get("mode", "binary")
        self._label_names = payload.get("label_names")
        self._dataset_thresholds = payload.get("dataset_thresholds")
        self._confidence_gate = float(payload.get("confidence_gate", 0.5))

    def _predict_proba(self, feat: dict[str, float]) -> np.ndarray:
        """Reproduce sklearn ``predict_proba`` for one feature row, in NumPy.

        Applies the stored standard scaler and the (binary or multinomial)
        logistic regression: ``softmax`` for the multiclass model, ``sigmoid``
        for a binary one. Matches the trained sklearn model bit-for-bit (see the
        converter's parity gate).
        """
        x = np.array([feat[c] for c in self._feature_cols], dtype=float)
        x_scaled = (x - self._scaler_mean) / self._scaler_scale
        scores = x_scaled @ self._coef.T + self._intercept
        if self._coef.shape[0] == 1:  # binary
            p1 = 1.0 / (1.0 + np.exp(-scores))
            return np.array([1.0 - p1[0], p1[0]])
        shifted = scores - np.max(scores)
        exp = np.exp(shifted)
        return np.asarray(exp / np.sum(exp), dtype=float)

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
        prob = self._predict_proba(feat)
        pred = int(np.argmax(prob))
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
        probs = self._predict_proba(feat)
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
