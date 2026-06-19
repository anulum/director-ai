# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — RAGTruth checkpoint selector tests

from __future__ import annotations

import json
from pathlib import Path

from training.select_ragtruth_checkpoint import select_checkpoint


def _write_result(path: Path, *, f1: float, fpr: float, precision: float = 0.8) -> None:
    path.write_text(
        json.dumps(
            {
                "model_dir": str(path.parent / path.stem),
                "model_sha256": None,
                "best": {
                    "f1": f1,
                    "precision": precision,
                    "recall": 0.74,
                    "balanced_accuracy": 0.82,
                    "fpr": fpr,
                    "p": 0.8,
                    "k": 2,
                    "tp": 700,
                    "fp": 100,
                    "tn": 1600,
                    "fn": 240,
                },
            }
        )
    )


def test_select_checkpoint_uses_fpr_tiebreak(tmp_path) -> None:
    higher_fpr = tmp_path / "higher_fpr.json"
    lower_fpr = tmp_path / "lower_fpr.json"
    _write_result(higher_fpr, f1=0.7640, fpr=0.090)
    _write_result(lower_fpr, f1=0.7632, fpr=0.070)

    result = select_checkpoint([higher_fpr, lower_fpr], f1_tie_delta=0.002)

    assert result["selected"]["path"] == str(lower_fpr)
    assert result["selected"]["fpr"] == 0.070
    assert result["selected"]["passes_gate"] is True


def test_select_checkpoint_prefers_clear_f1_gain(tmp_path) -> None:
    lower_f1 = tmp_path / "lower_f1.json"
    higher_f1 = tmp_path / "higher_f1.json"
    _write_result(lower_f1, f1=0.763, fpr=0.050)
    _write_result(higher_f1, f1=0.780, fpr=0.075)

    result = select_checkpoint([lower_f1, higher_f1], f1_tie_delta=0.002)

    assert result["selected"]["path"] == str(higher_f1)
