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
        "premise_sent_count": premise.count(".")
        + premise.count("!")
        + premise.count("?"),
        "hypothesis_sent_count": hypothesis.count(".")
        + hypothesis.count("!")
        + hypothesis.count("?"),
        "avg_word_len_premise": np.mean([len(w) for w in p_words]) if p_words else 0.0,
        "avg_word_len_hypothesis": np.mean([len(w) for w in h_words])
        if h_words
        else 0.0,
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
    """Train logistic regression on extracted features (legacy path)."""
    import pickle

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    with open(features_path) as f:
        data = json.load(f)

    # Handle both old (15-feature) and new (20-feature) formats
    available = set(data[0].keys()) if data else set()
    feature_cols = [c for c in FEATURE_COLS if c in available]

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

    test_ba = float(balanced_accuracy_score(y_test, clf.predict(x_test_s)))
    train_ba = float(balanced_accuracy_score(y_train, clf.predict(x_train_s)))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(
            {"classifier": clf, "scaler": scaler, "feature_cols": feature_cols}, f
        )

    logger.info("Meta-classifier saved to %s", output_path)
    logger.info("Test BA: %.1f%%, Train BA: %.1f%%", test_ba * 100, train_ba * 100)
    return {"train_balanced_acc": train_ba, "test_balanced_acc": test_ba}


def generate_features_from_cache(
    cache_path: str,
    output_path: str = "features/aggrefact_meta_features.json",
) -> list[dict]:
    """Extract features using cached NLI scores (no GPU needed).

    Pairs cached scores with original HF dataset text to compute
    text features without re-running inference.
    """
    from benchmarks.aggrefact_eval import _load_aggrefact

    data = json.loads(Path(cache_path).read_text())
    cached = data["scores"]

    rows = _load_aggrefact()
    valid_rows = [
        r
        for r in rows
        if r.get("label") is not None and r.get("doc") and r.get("claim")
    ]

    if len(valid_rows) != len(cached):
        raise ValueError(
            f"Cache/dataset mismatch: {len(cached)} scores vs {len(valid_rows)} valid rows"
        )

    features_list: list[dict] = []
    for row, entry in zip(valid_rows, cached, strict=True):
        nli_score = entry["score"]
        feat = extract_features(
            premise=row["doc"],
            hypothesis=row["claim"],
            nli_score=nli_score,
            confidence=abs(nli_score - 0.5) * 2.0,
        )
        feat["label"] = entry["label"]
        feat["dataset"] = entry["dataset"]
        features_list.append(feat)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(features_list, f)
    logger.info("Saved %d feature vectors to %s", len(features_list), output_path)
    return features_list


def _macro_ba_from_datasets(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    datasets: np.ndarray,
) -> float:
    """Macro-averaged balanced accuracy across datasets (leaderboard metric)."""
    from sklearn.metrics import balanced_accuracy_score

    accs = []
    for ds in sorted(set(datasets)):
        mask = datasets == ds
        if mask.sum() == 0:
            continue
        accs.append(balanced_accuracy_score(y_true[mask], y_pred[mask]))
    return float(np.mean(accs))


def _global_threshold_ba(
    scores: np.ndarray,
    y_true: np.ndarray,
    datasets: np.ndarray,
) -> tuple[float, float]:
    """Sweep global threshold, return (best_threshold, macro_BA)."""
    best_t, best_ba = 0.5, 0.0
    for t_int in range(10, 91):
        t = t_int / 100.0
        y_pred = (scores >= t).astype(int)
        ba = _macro_ba_from_datasets(y_true, y_pred, datasets)
        if ba > best_ba:
            best_ba, best_t = ba, t
    return best_t, best_ba


def _per_dataset_oracle_ba(
    scores: np.ndarray,
    y_true: np.ndarray,
    datasets: np.ndarray,
) -> float:
    """Per-dataset threshold oracle (upper bound)."""
    from sklearn.metrics import balanced_accuracy_score

    accs = []
    for ds in sorted(set(datasets)):
        mask = datasets == ds
        s, yt = scores[mask], y_true[mask]
        best_ba = 0.0
        for t_int in range(10, 91):
            t = t_int / 100.0
            ba = balanced_accuracy_score(yt, (s >= t).astype(int))
            if ba > best_ba:
                best_ba = ba
        accs.append(best_ba)
    return float(np.mean(accs))


def evaluate_classifiers(
    features_path: str,
    seed: int = 42,
) -> dict:
    """Compare global threshold, meta-classifiers, and per-dataset oracle.

    Evaluation modes:
    1. Leave-one-dataset-out CV (generalization to unseen dataset types)
    2. Stratified 5-fold CV (overall performance)
    3. Full-data training (production model metrics)
    """
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    with open(features_path) as f:
        data = json.load(f)

    x_mat = np.array([[d[c] for c in FEATURE_COLS] for d in data])
    y = np.array([d["label"] for d in data])
    datasets = np.array([d["dataset"] for d in data])
    scores = np.array([d["nli_score"] for d in data])

    # ── Baselines ──
    global_t, global_ba = _global_threshold_ba(scores, y, datasets)
    oracle_ba = _per_dataset_oracle_ba(scores, y, datasets)

    print(f"\n{'=' * 70}")
    print("  Meta-Classifier Evaluation — LLM-AggreFact 29,320 samples")
    print(f"{'=' * 70}")
    print("\n  Baselines:")
    print(f"    Global threshold (t={global_t:.2f}):     {global_ba:.2%}")
    print(f"    Per-dataset oracle:              {oracle_ba:.2%}")
    print(f"    Gap to close:                    {oracle_ba - global_ba:+.2%}")

    classifiers = {
        "LogReg": lambda: LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
            C=1.0,
        ),
        "RF-100": lambda: RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "GBT-100": lambda: GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=seed,
        ),
    }

    # ── Leave-one-dataset-out CV ──
    unique_ds = sorted(set(datasets))
    print(f"\n  Leave-One-Dataset-Out CV ({len(unique_ds)} folds):")
    print(f"  {'':20s} {'LOO BA':>10s}  {'per-dataset detail'}")
    print(f"  {'-' * 60}")

    loo_results: dict[str, dict] = {}
    for clf_name, clf_factory in classifiers.items():
        per_ds_ba: dict[str, float] = {}
        for held_out in unique_ds:
            train_mask = datasets != held_out
            test_mask = datasets == held_out
            x_train, x_test = x_mat[train_mask], x_mat[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            scaler = StandardScaler()
            x_train_s = scaler.fit_transform(x_train)
            x_test_s = scaler.transform(x_test)

            clf = clf_factory()
            clf.fit(x_train_s, y_train)
            y_pred = clf.predict(x_test_s)
            per_ds_ba[held_out] = float(balanced_accuracy_score(y_test, y_pred))

        macro_ba = float(np.mean(list(per_ds_ba.values())))
        loo_results[clf_name] = {"macro_ba": macro_ba, "per_dataset": per_ds_ba}

        worst = min(per_ds_ba, key=per_ds_ba.get)
        best = max(per_ds_ba, key=per_ds_ba.get)
        print(
            f"  {clf_name:20s} {macro_ba:10.2%}  "
            f"worst={worst}({per_ds_ba[worst]:.1%}) "
            f"best={best}({per_ds_ba[best]:.1%})"
        )

    # Global threshold LOO comparison
    loo_global_per_ds = {}
    for held_out in unique_ds:
        mask = datasets == held_out
        y_pred = (scores[mask] >= global_t).astype(int)
        loo_global_per_ds[held_out] = float(balanced_accuracy_score(y[mask], y_pred))
    loo_global_ba = float(np.mean(list(loo_global_per_ds.values())))
    print(f"  {'Global(t=' + f'{global_t:.2f})':20s} {loo_global_ba:10.2%}")

    # ── Stratified 5-fold CV ──
    print("\n  Stratified 5-Fold CV:")
    print(f"  {'':20s} {'5-Fold BA':>10s}")
    print(f"  {'-' * 40}")

    kfold_results: dict[str, float] = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for clf_name, clf_factory in classifiers.items():
        fold_bas: list[float] = []
        for train_idx, test_idx in skf.split(x_mat, y):
            scaler = StandardScaler()
            x_train_s = scaler.fit_transform(x_mat[train_idx])
            x_test_s = scaler.transform(x_mat[test_idx])
            clf = clf_factory()
            clf.fit(x_train_s, y[train_idx])
            y_pred = clf.predict(x_test_s)
            ba = _macro_ba_from_datasets(y[test_idx], y_pred, datasets[test_idx])
            fold_bas.append(ba)
        mean_ba = float(np.mean(fold_bas))
        kfold_results[clf_name] = mean_ba
        print(f"  {clf_name:20s} {mean_ba:10.2%} (+/- {np.std(fold_bas):.2%})")

    # ── Full-data training (feature importance) ──
    print("\n  Feature Importance (RF-100, full data):")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_mat)
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(x_scaled, y)
    importances = sorted(
        zip(FEATURE_COLS, rf.feature_importances_, strict=True),
        key=lambda x: x[1],
        reverse=True,
    )
    for name, imp in importances[:10]:
        print(f"    {name:30s} {imp:.4f}")

    # ── Per-dataset detail for best classifier ──
    best_clf_name = max(loo_results, key=lambda k: loo_results[k]["macro_ba"])
    best_loo = loo_results[best_clf_name]
    print(f"\n  Per-Dataset Detail (LOO, {best_clf_name}):")
    print(
        f"  {'Dataset':20s} {'Global':>8s} {'Clf':>8s} {'Oracle':>8s} {'Clf-Global':>10s}"
    )
    print(f"  {'-' * 60}")
    for ds in unique_ds:
        g = loo_global_per_ds[ds]
        c = best_loo["per_dataset"][ds]
        # Oracle for this dataset
        mask = datasets == ds
        o = _per_dataset_oracle_ba(scores[mask], y[mask], datasets[mask])
        delta = c - g
        print(f"  {ds:20s} {g:8.1%} {c:8.1%} {o:8.1%} {delta:+10.1%}")

    # ── Summary ──
    best_loo_ba = best_loo["macro_ba"]
    print(f"\n  {'=' * 70}")
    print("  Summary:")
    print(f"    Global threshold:    {global_ba:.2%}")
    print(f"    Best classifier:     {best_loo_ba:.2%} ({best_clf_name}, LOO)")
    print(f"    Per-dataset oracle:  {oracle_ba:.2%}")
    delta_global = best_loo_ba - global_ba
    delta_oracle = oracle_ba - best_loo_ba
    print(f"    Gain over global:    {delta_global:+.2%}")
    print(f"    Remaining to oracle: {delta_oracle:+.2%}")
    print(f"  {'=' * 70}\n")

    return {
        "global_threshold": global_t,
        "global_ba": global_ba,
        "oracle_ba": oracle_ba,
        "loo_results": {k: v["macro_ba"] for k, v in loo_results.items()},
        "kfold_results": kfold_results,
        "best_classifier": best_clf_name,
        "best_loo_ba": best_loo_ba,
    }


def train_production_model(
    features_path: str,
    output_path: str = "models/meta_classifier.pkl",
    classifier: str = "rf",
    seed: int = 42,
) -> dict:
    """Train final production model on all data."""
    import pickle

    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    with open(features_path) as f:
        data = json.load(f)

    x_mat = np.array([[d[c] for c in FEATURE_COLS] for d in data])
    y = np.array([d["label"] for d in data])
    datasets = np.array([d["dataset"] for d in data])

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_mat)

    clf_map = {
        "lr": lambda: LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
            C=1.0,
        ),
        "rf": lambda: RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "gbt": lambda: GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=seed,
        ),
    }

    clf = clf_map[classifier]()
    clf.fit(x_scaled, y)
    y_pred = clf.predict(x_scaled)
    train_ba = _macro_ba_from_datasets(y, y_pred, datasets)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "classifier": clf,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
    }
    with open(output_path, "wb") as f:
        pickle.dump(bundle, f)

    logger.info(
        "Production model saved to %s (train BA: %.2f%%)", output_path, train_ba * 100
    )
    return {"train_ba": train_ba, "classifier": classifier, "path": output_path}


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

        ds_name = self._label_names[pred_idx]
        threshold = self._dataset_thresholds.get(ds_name)
        return threshold, conf


# Text-only feature extraction (no NLI score dependency)
TEXT_FEATURE_COLS = [
    c
    for c in FEATURE_COLS
    if c not in ("nli_score", "confidence", "score_distance_from_half")
]


def extract_text_features(premise: str, hypothesis: str) -> dict:
    """Extract text-only features (no NLI score needed)."""
    return extract_features(premise, hypothesis, nli_score=0.0, confidence=0.0)


# Per-dataset optimal NLI thresholds from L40S cached score sweep (29,320 samples)
DATASET_NLI_THRESHOLDS = {
    "AggreFact-CNN": 0.70,
    "AggreFact-XSum": 0.30,
    "ClaimVerify": 0.72,
    "ExpertQA": 0.17,
    "FactCheck-GPT": 0.22,
    "Lfqa": 0.58,
    "RAGTruth": 0.63,
    "Reveal": 0.34,
    "TofuEval-MediaS": 0.66,
    "TofuEval-MeetB": 0.54,
    "Wice": 0.21,
}


def train_dataset_type_classifier(
    features_path: str,
    output_path: str = "models/dataset_type_classifier.pkl",
    n_estimators: int = 20,
    max_depth: int = 6,
    confidence_gate: float = 0.5,
    seed: int = 42,
) -> dict:
    """Train RF classifier that predicts dataset type from text features.

    The model predicts which AggreFact sub-dataset a (premise, hypothesis)
    pair belongs to, enabling per-dataset optimal NLI thresholds.
    """
    import pickle

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    with open(features_path) as f:
        data = json.load(f)

    x_mat = np.array([[d[c] for c in TEXT_FEATURE_COLS] for d in data])
    ds_labels = np.array([d["dataset"] for d in data])
    nli_scores = np.array([d["nli_score"] for d in data])
    y_true = np.array([d["label"] for d in data])

    le = LabelEncoder()
    y_ds = le.fit_transform(ds_labels)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_mat)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )

    # 5-fold CV to estimate confidence-gated BA
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_probs = cross_val_predict(clf, x_scaled, y_ds, cv=skf, method="predict_proba")

    # Simulate confidence-gated threshold selection

    gated_preds = []
    gated_true = []
    gated_ds = []
    global_t = 0.46  # global NLI threshold fallback

    for i in range(len(data)):
        pred_idx = int(np.argmax(cv_probs[i]))
        conf = cv_probs[i][pred_idx]
        score = nli_scores[i]

        if conf >= confidence_gate:
            ds_name = le.inverse_transform([pred_idx])[0]
            t = DATASET_NLI_THRESHOLDS.get(ds_name, global_t)
        else:
            t = global_t

        gated_preds.append(int(score >= t))
        gated_true.append(y_true[i])
        gated_ds.append(ds_labels[i])

    gated_ba = _macro_ba_from_datasets(
        np.array(gated_true),
        np.array(gated_preds),
        np.array(gated_ds),
    )

    # Train final model on all data
    clf.fit(x_scaled, y_ds)

    bundle = {
        "classifier": clf,
        "scaler": scaler,
        "feature_cols": TEXT_FEATURE_COLS,
        "label_names": list(le.classes_),
        "dataset_thresholds": DATASET_NLI_THRESHOLDS,
        "confidence_gate": confidence_gate,
        "mode": "dataset_type",
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(bundle, f)

    size_kb = os.path.getsize(output_path) / 1024
    logger.info(
        "Dataset-type classifier: %.2f%% BA (5-fold gated), %.0f KB, saved to %s",
        gated_ba * 100,
        size_kb,
        output_path,
    )
    return {"gated_ba": gated_ba, "size_kb": size_kb, "path": output_path}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Meta-classifier: train from cached NLI scores, evaluate, deploy"
    )
    parser.add_argument(
        "--generate-features",
        action="store_true",
        help="Generate features via GPU inference (slow)",
    )
    parser.add_argument(
        "--from-cache",
        type=str,
        default=None,
        metavar="PATH",
        help="Generate features from cached scores JSON (no GPU)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run full evaluation: LOO-CV + 5-fold + comparison",
    )
    parser.add_argument(
        "--train", action="store_true", help="Train production model (binary)"
    )
    parser.add_argument(
        "--train-dataset-type",
        action="store_true",
        help="Train dataset-type classifier (RF-20-d6)",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="rf",
        choices=["lr", "rf", "gbt"],
        help="Classifier type for --train (default: rf)",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--features", type=str, default="features/aggrefact_meta_features.json"
    )
    parser.add_argument("--output", type=str, default="models/meta_classifier.pkl")
    args = parser.parse_args()

    if args.from_cache:
        generate_features_from_cache(
            cache_path=args.from_cache,
            output_path=args.features,
        )
    elif args.generate_features:
        generate_features_aggrefact(
            model_name=args.model,
            max_samples=args.max_samples,
            output_path=args.features,
        )

    if args.evaluate:
        if not os.path.exists(args.features):
            logger.error("Features file not found: %s", args.features)
            logger.error("Run with --from-cache or --generate-features first")
            raise SystemExit(1)
        evaluate_classifiers(features_path=args.features)

    if args.train:
        if not os.path.exists(args.features):
            logger.error("Features file not found: %s", args.features)
            raise SystemExit(1)
        train_production_model(
            features_path=args.features,
            output_path=args.output,
            classifier=args.classifier,
        )

    if args.train_dataset_type:
        if not os.path.exists(args.features):
            logger.error("Features file not found: %s", args.features)
            raise SystemExit(1)
        train_dataset_type_classifier(
            features_path=args.features,
            output_path=args.output,
        )
