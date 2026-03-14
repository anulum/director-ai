# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Meta-Classifier Training
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""Train a lightweight logistic regression meta-classifier (Phase 6A).

Takes (NLI_score, confidence_entropy, text_length, has_negation,
chunk_count) as features and predicts the optimal decision. Better
than a fixed threshold — learns when the NLI model is reliable vs
when to defer.

Usage::

    # Step 1: Generate features from AggreFact
    python tools/train_meta_classifier.py --generate-features \\
        --output features/aggrefact_meta_features.json

    # Step 2: Train meta-classifier
    python tools/train_meta_classifier.py --train \\
        --features features/aggrefact_meta_features.json \\
        --output models/meta_classifier.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path

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


def extract_features(
    premise: str,
    hypothesis: str,
    nli_score: float,
    confidence: float,
    chunk_count: int = 1,
) -> dict:
    """Extract meta-classifier features from a scored pair."""
    h_words = set(hypothesis.lower().split())
    p_words = set(premise.lower().split())

    return {
        "nli_score": nli_score,
        "confidence": confidence,
        "premise_len": len(premise),
        "hypothesis_len": len(hypothesis),
        "premise_word_count": len(p_words),
        "hypothesis_word_count": len(h_words),
        "word_overlap": (len(p_words & h_words) / max(len(p_words | h_words), 1)),
        "has_negation_premise": int(bool(p_words & NEGATION_WORDS)),
        "has_negation_hypothesis": int(bool(h_words & NEGATION_WORDS)),
        "negation_asymmetry": int(
            bool(p_words & NEGATION_WORDS) != bool(h_words & NEGATION_WORDS),
        ),
        "chunk_count": chunk_count,
        "score_distance_from_half": abs(nli_score - 0.5),
        "has_question_mark": int("?" in hypothesis),
        "num_entities_premise": len(re.findall(r"[A-Z][a-z]+", premise)),
        "num_entities_hypothesis": len(re.findall(r"[A-Z][a-z]+", hypothesis)),
    }


def generate_features_aggrefact(
    model_name: str | None = None,
    max_samples: int | None = None,
    output_path: str = "features/aggrefact_meta_features.json",
) -> list[dict]:
    """Score all AggreFact samples and extract meta-features."""
    from benchmarks.aggrefact_eval import _BinaryNLIPredictor, _load_aggrefact

    predictor = _BinaryNLIPredictor(model_name=model_name)
    rows = _load_aggrefact(max_samples)

    features_list: list[dict] = []
    for i, row in enumerate(rows):
        doc = row.get("doc", "")
        claim = row.get("claim", "")
        label = row.get("label")
        ds_name = row.get("dataset", "unknown")
        if label is None or not doc or not claim:
            continue

        ent_prob = predictor.score(doc, claim)

        # Approximate confidence from entailment probability distance from 0.5
        confidence = abs(ent_prob - 0.5) * 2.0

        feat = extract_features(
            premise=doc,
            hypothesis=claim,
            nli_score=ent_prob,
            confidence=confidence,
        )
        feat["label"] = int(label)
        feat["dataset"] = ds_name
        features_list.append(feat)

        if (i + 1) % 1000 == 0:
            logger.info("Processed %d/%d samples", i + 1, len(rows))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(features_list, f)
    logger.info("Saved %d feature vectors to %s", len(features_list), output_path)
    return features_list


def train_meta_classifier(
    features_path: str,
    output_path: str = "models/meta_classifier.pkl",
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Train logistic regression on extracted features."""
    import pickle

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    with open(features_path) as f:
        data = json.load(f)

    feature_cols = [
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
    ]

    x_data = np.array([[d[c] for c in feature_cols] for d in data])
    y = np.array([d["label"] for d in data])

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
        C=1.0,
    )
    clf.fit(x_train_s, y_train)

    y_pred_train = clf.predict(x_train_s)
    y_pred_test = clf.predict(x_test_s)

    results = {
        "train_balanced_acc": float(balanced_accuracy_score(y_train, y_pred_train)),
        "test_balanced_acc": float(balanced_accuracy_score(y_test, y_pred_test)),
        "train_f1": float(f1_score(y_train, y_pred_train, average="macro")),
        "test_f1": float(f1_score(y_test, y_pred_test, average="macro")),
        "train_size": len(y_train),
        "test_size": len(y_test),
        "feature_importances": {
            col: float(coef)
            for col, coef in zip(feature_cols, clf.coef_[0], strict=True)
        },
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model_bundle = {
        "classifier": clf,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }
    with open(output_path, "wb") as f:
        pickle.dump(model_bundle, f)

    metrics_path = output_path.replace(".pkl", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Meta-classifier saved to %s", output_path)
    logger.info(
        "Test BA: %.1f%%, Train BA: %.1f%%",
        results["test_balanced_acc"] * 100,
        results["train_balanced_acc"] * 100,
    )

    # Feature importance ranking
    importance = sorted(
        results["feature_importances"].items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    logger.info("Feature importance (top 5):")
    for name, coef in importance[:5]:
        logger.info("  %s: %.4f", name, coef)

    return results


class MetaClassifier:
    """Lightweight meta-classifier for production use.

    Loads a trained logistic regression model and predicts whether
    a given NLI score + features should be classified as supported.
    """

    def __init__(self, model_path: str):
        import pickle

        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        self._clf = bundle["classifier"]
        self._scaler = bundle["scaler"]
        self._feature_cols = bundle["feature_cols"]

    def predict(
        self,
        premise: str,
        hypothesis: str,
        nli_score: float,
        confidence: float,
        chunk_count: int = 1,
    ) -> tuple[bool, float]:
        """Predict support/hallucination with probability.

        Returns (is_supported, probability).
        """
        feat = extract_features(
            premise,
            hypothesis,
            nli_score,
            confidence,
            chunk_count,
        )
        x = np.array([[feat[c] for c in self._feature_cols]])
        x_scaled = self._scaler.transform(x)
        prob = self._clf.predict_proba(x_scaled)[0]
        pred = int(self._clf.predict(x_scaled)[0])
        return bool(pred == 1), float(prob[1])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Meta-classifier training")
    parser.add_argument("--generate-features", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--features", type=str, default="features/aggrefact_meta_features.json"
    )
    parser.add_argument("--output", type=str, default="models/meta_classifier.pkl")
    args = parser.parse_args()

    if args.generate_features:
        generate_features_aggrefact(
            model_name=args.model,
            max_samples=args.max_samples,
            output_path=args.features,
        )

    if args.train:
        if not os.path.exists(args.features):
            logger.error("Features file not found: %s", args.features)
            logger.error("Run with --generate-features first")
            raise SystemExit(1)
        train_meta_classifier(
            features_path=args.features,
            output_path=args.output,
        )
