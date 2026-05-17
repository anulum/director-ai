# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Distillation reproducibility tests

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "training" / "train_distillation.py"
SPEC = importlib.util.spec_from_file_location("train_distillation", TRAINER)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

TrainingRunConfig = MODULE.TrainingRunConfig
build_parser = MODULE.build_parser
build_subset = MODULE.build_subset
validate_training_run_config = MODULE.validate_training_run_config
write_training_run_manifest = MODULE.write_training_run_manifest


class TinyDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __getitem__(self, key: str) -> list[Any]:
        return [row[key] for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def select(self, indices: list[int]) -> TinyDataset:
        return TinyDataset([self.rows[index] for index in indices])


def _dataset() -> TinyDataset:
    rows: list[dict[str, Any]] = []
    for index in range(10):
        rows.append(
            {
                "premise": f"summ premise {index}",
                "hypothesis": f"summ hypothesis {index}",
                "label": index % 3,
                "source": "halueval_summarization",
            }
        )
        rows.append(
            {
                "premise": f"general premise {index}",
                "hypothesis": f"general hypothesis {index}",
                "label": index % 3,
                "source": "fever",
            }
        )
    return TinyDataset(rows)


def test_build_subset_uses_explicit_seed_for_reproducible_sampling() -> None:
    first = build_subset(_dataset(), summ_target=5, general_target=5, seed=11)
    second = build_subset(_dataset(), summ_target=5, general_target=5, seed=11)
    third = build_subset(_dataset(), summ_target=5, general_target=5, seed=12)

    assert first.rows == second.rows
    assert first.rows != third.rows


def test_training_parser_exposes_reproducibility_controls() -> None:
    args = build_parser().parse_args(
        [
            "--teacher",
            "training/output/deberta-v3-base-hallucination",
            "--output-dir",
            "MODELS/lite-scorer-v2/student",
        ]
    )

    assert args.seed == 20260518
    assert args.eval_limit == 5000
    assert args.num_workers == 2


def test_validate_training_run_config_rejects_invalid_bounds() -> None:
    config = TrainingRunConfig(
        teacher="teacher",
        student="student",
        output_dir=Path("MODELS/lite-scorer-v2/student"),
        lr=5e-5,
        epochs=0,
        batch_size=32,
        max_length=256,
        temperature=3.0,
        alpha=1.5,
        summ_target=15000,
        general_target=15000,
        seed=-1,
        eval_limit=0,
        num_workers=-1,
    )

    assert validate_training_run_config(config) == [
        "epochs must be positive",
        "alpha must be in (0, 1]",
        "seed must be non-negative",
        "eval_limit must be positive",
        "num_workers must be non-negative",
    ]


def test_write_training_run_manifest_records_inputs_without_score_claim(
    tmp_path: Path,
) -> None:
    config = TrainingRunConfig(
        teacher="training/output/deberta-v3-base-hallucination",
        student="microsoft/MiniLM-L6-H384-uncased",
        output_dir=tmp_path,
        lr=5e-5,
        epochs=5,
        batch_size=32,
        max_length=256,
        temperature=3.0,
        alpha=0.5,
        summ_target=15000,
        general_target=15000,
        seed=20260518,
        eval_limit=5000,
        num_workers=2,
    )

    write_training_run_manifest(
        config,
        train_rows=30000,
        eval_rows=5000,
        device="cpu",
        teacher_params=435000000,
        student_params=22000000,
    )

    payload = json.loads((tmp_path / "training_run_manifest.json").read_text())
    assert payload["schema_version"] == "1.0.0"
    assert payload["public_score_claim"] is False
    assert payload["claim_boundary"].startswith("Training run metadata only")
    assert payload["seed"] == 20260518
    assert payload["eval_limit"] == 5000
    assert payload["train_rows"] == 30000
    assert payload["eval_rows"] == 5000
    assert "balanced_accuracy" not in payload
