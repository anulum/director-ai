# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Fine-tune API real-surface tests
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


@pytest.mark.parametrize(
    "warmup_ratio,expected_steps",
    [(0.0, 0), (0.25, 1), (1.0, 4), (-0.1, None), (1.1, None), (float("nan"), None)],
)
def test_finetune_trains_local_cpu_checkpoint(
    tmp_path: Path, warmup_ratio: float, expected_steps: int | None
) -> None:
    """Train and reload a real local checkpoint with the locked Transformers API."""
    import math
    from concurrent.futures import ThreadPoolExecutor

    import torch
    from transformers import (
        BertConfig,
        BertForSequenceClassification,
        BertTokenizer,
        TrainingArguments,
    )

    from director_ai.core.training.finetune import FinetuneConfig

    previous_threads = torch.get_num_threads()
    previous_rng = torch.get_rng_state()
    torch.set_num_threads(1)
    try:
        model_dir = tmp_path / "base-model"
        model_dir.mkdir()
        words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "evidence", "approval"]
        tokenizer = BertTokenizer(
            vocab={word: index for index, word in enumerate(words)}
        )
        tokenizer.save_pretrained(model_dir)
        configuration = BertConfig(
            vocab_size=len(words),
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
        configuration.num_labels = 2
        model = BertForSequenceClassification.from_pretrained(
            None, config=configuration, state_dict={}
        )
        model.save_pretrained(model_dir)
        train_path = _write_jsonl(
            tmp_path / "train.jsonl",
            [
                {"premise": "evidence", "hypothesis": "approval", "label": label}
                for label in (0, 1, 0, 1)
            ],
        )
        output = tmp_path / "trained-model"
        config = FinetuneConfig(
            base_model=str(model_dir),
            output_dir=str(output),
            epochs=1,
            batch_size=1,
            max_length=8,
            fp16=False,
            warmup_ratio=warmup_ratio,
        )
        if expected_steps is None:
            with pytest.raises(ValueError, match="warmup_ratio"):
                finetune_nli(train_path, config=config)
            return
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(finetune_nli, train_path, config=config).result(
                timeout=30
            )
        assert result.train_samples == 4
        assert result.epochs_completed == 1
        assert math.isfinite(result.final_loss)
        # This file was written by the real Trainer in this isolated fixture.
        saved_args = torch.load(output / "training_args.bin", weights_only=False)
        assert isinstance(saved_args, TrainingArguments)
        assert saved_args.get_warmup_steps(4) == expected_steps
        reloaded = BertForSequenceClassification.from_pretrained(output)
        assert not torch.equal(model.classifier.weight, reloaded.classifier.weight)
        encoded = tokenizer("evidence", "approval", return_tensors="pt")
        with torch.no_grad():
            logits = reloaded(**encoded).logits
        assert logits.shape == (1, 2)
        assert torch.isfinite(logits).all()
    finally:
        torch.set_rng_state(previous_rng)
        torch.set_num_threads(previous_threads)
