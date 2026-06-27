# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - fine-tune metrics real-surface tests
"""Real fine-tuning callback-shape coverage for metric computation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from director_ai.core.training.finetune import _compute_metrics


@dataclass
class _EvalPrediction:
    """Small structural stand-in for Transformers ``EvalPrediction``."""

    predictions: NDArray[np.float64]
    label_ids: NDArray[np.int64]


def test_compute_metrics_accepts_transformers_eval_prediction_shape() -> None:
    """Metric callback should accept the object shape passed by Transformers."""
    eval_pred = _EvalPrediction(
        predictions=np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64),
        label_ids=np.asarray([0, 1], dtype=np.int64),
    )

    assert _compute_metrics(eval_pred) == {"balanced_accuracy": 1.0, "f1": 1.0}
