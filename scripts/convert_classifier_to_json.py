# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — dataset-type classifier pickle→JSON converter
"""Convert the bundled dataset-type classifier from pickle to safe JSON.

The shipped classifier is a multinomial ``LogisticRegression`` over a
``StandardScaler``; both reduce to a handful of float arrays. This script reads
the legacy ``.pkl``, extracts those arrays, writes a pickle-free JSON artefact,
and — before writing — verifies that the pure-NumPy reproduction used at runtime
matches scikit-learn's ``predict_proba`` to within a tight tolerance on random
inputs. Requires scikit-learn only at conversion time; the runtime loader never
imports it.

Usage::

    python -m scripts.convert_classifier_to_json \
        src/director_ai/core/models/dataset_type_classifier.pkl \
        src/director_ai/core/models/dataset_type_classifier.json
"""

from __future__ import annotations

import json
import pickle  # nosec B403 - converting a first-party, hash-logged artefact
import sys
from pathlib import Path

import numpy as np


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _numpy_proba(payload: dict, x: np.ndarray) -> np.ndarray:
    """Reproduce sklearn predict_proba from the JSON payload, in NumPy only."""
    mean = np.asarray(payload["scaler_mean"])
    scale = np.asarray(payload["scaler_scale"])
    coef = np.asarray(payload["coef"])
    intercept = np.asarray(payload["intercept"])
    x_scaled = (x - mean) / scale
    scores = x_scaled @ coef.T + intercept
    if coef.shape[0] == 1:  # binary
        p1 = 1.0 / (1.0 + np.exp(-scores))
        return np.hstack([1.0 - p1, p1])
    return _softmax(scores)


def convert(pkl_path: Path, json_path: Path) -> dict:
    raw = pkl_path.read_bytes()
    bundle = pickle.loads(raw)  # nosec B301 - first-party artefact being migrated
    clf = bundle["classifier"]
    scaler = bundle["scaler"]
    payload = {
        "format": "director.dataset_type_classifier.v1",
        "model": "multinomial_logistic_regression",
        "mode": bundle.get("mode", "binary"),
        "feature_cols": list(bundle["feature_cols"]),
        "classes": [int(c) for c in clf.classes_],
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "label_names": bundle.get("label_names"),
        "dataset_thresholds": bundle.get("dataset_thresholds"),
        "confidence_gate": float(bundle.get("confidence_gate", 0.5)),
    }

    # Parity gate: the NumPy reproduction must match sklearn before we ship JSON.
    rng = np.random.default_rng(0)
    n_features = len(payload["feature_cols"])
    samples = rng.normal(size=(256, n_features))
    sk_proba = clf.predict_proba(scaler.transform(samples))
    np_proba = _numpy_proba(payload, samples)
    max_err = float(np.max(np.abs(sk_proba - np_proba)))
    if max_err > 1e-9:
        raise SystemExit(f"parity check failed: max |Δproba| = {max_err:.2e}")
    print(f"parity OK: max |Δproba| = {max_err:.2e} over {len(samples)} samples")

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {json_path} ({json_path.stat().st_size} bytes)")
    return payload


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        raise SystemExit(
            "usage: convert_classifier_to_json.py <input.pkl> <output.json>"
        )
    convert(Path(argv[0]), Path(argv[1]))


if __name__ == "__main__":
    main(sys.argv[1:])
