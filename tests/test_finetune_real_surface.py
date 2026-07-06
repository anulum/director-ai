# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real-surface coverage for fine-tuning data validation wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from director_ai.core.training.finetune import finetune_nli
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    """Write rows as JSONL and return the file path."""
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_finetune_unit_guard_declares_this_companion() -> None:
    """The mocked fine-tune unit guard should point at this companion."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_finetune.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_finetune_real_surface.py" in reason


def test_finetune_gpu_unit_guard_declares_real_surface_companions() -> None:
    """The GPU fine-tune guard should name its public companion surfaces."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_finetune_gpu.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_finetune_real_surface.py" in reason
    assert "tests/test_finetune_api_real_surface.py" in reason
    assert "tests/test_finetune_benchmark_real_surface.py" in reason


def test_finetune_nli_rejects_non_binary_labels_before_optional_training_imports(
    tmp_path: Path,
) -> None:
    """Direct Python training should fail closed before loading Transformers."""
    train_path = _write_jsonl(
        tmp_path / "train.jsonl",
        [
            {
                "premise": "A signed approval exists.",
                "hypothesis": "Approved.",
                "label": 1,
            },
            {
                "premise": "No revocation was filed.",
                "hypothesis": "Revoked.",
                "label": 2,
            },
        ],
    )

    with pytest.raises(ValueError, match="label must be 0 or 1"):
        finetune_nli(train_path)


def test_finetune_nli_truncates_large_label_error_reports(tmp_path: Path) -> None:
    """Repeated invalid labels should fail with bounded diagnostics."""
    train_path = _write_jsonl(
        tmp_path / "train.jsonl",
        [
            {
                "premise": f"Premise {index}.",
                "hypothesis": f"Claim {index}.",
                "label": 9,
            }
            for index in range(11)
        ],
    )

    with pytest.raises(ValueError, match="truncated, too many label errors"):
        finetune_nli(train_path)
