# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — standalone training data and evaluation tests
"""Exercise the standalone runner with its real human-readable training format."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from transformers import BertTokenizer, EvalPrediction

from tools import run_training_standalone as runner

ROOT = Path(__file__).resolve().parents[2]


def test_standalone_training_examples_roundtrip_existing_jsonl(tmp_path: Path) -> None:
    """Retain text, labels and extra fields in previously prepared JSON Lines."""
    existing = {
        "premise": "Sète appears in the document.",
        "hypothesis": "The document mentions Sète.",
        "label": 1,
        "dataset": "local",
    }
    path = tmp_path / "examples.jsonl"
    path.write_text(
        "\n" + json.dumps(existing, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    assert runner.load_jsonl(str(path)) == [existing]
    output = tmp_path / "prepared" / "examples.jsonl"
    runner.write_jsonl(runner.load_jsonl(path), output)
    assert runner.load_jsonl(output) == [existing]
    assert "Sète" in output.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "record",
    [
        "[]",
        '{"premise":3,"hypothesis":"known","label":0}',
        '{"premise":"known","hypothesis":null,"label":0}',
        '{"premise":"known","hypothesis":"known","label":"entailment"}',
    ],
)
def test_standalone_rejects_invalid_training_examples(
    tmp_path: Path, record: str
) -> None:
    """Fail at the actual JSONL boundary for each malformed example field."""
    path = tmp_path / "examples.jsonl"
    path.write_text(record + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid training example"):
        runner.load_jsonl(path)


def test_standalone_binary_labels_and_reproducible_split(tmp_path: Path) -> None:
    """Split actual loaded records deterministically without losing accepted labels."""
    assert [
        runner.nli_to_binary(label)
        for label in ("Supports", "Refutes", "neutral", 0, 2, 1)
    ] == [1, 0, None, 1, 0, None]
    path = tmp_path / "examples.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "premise": f"example {index}",
                    "hypothesis": "claim",
                    "label": index % 2,
                }
            )
            + "\n"
            for index in range(10)
        ),
        encoding="utf-8",
    )
    first = runner.split_data(runner.load_jsonl(path), eval_ratio=0.2, seed=7)
    second = runner.split_data(runner.load_jsonl(path), eval_ratio=0.2, seed=7)
    assert first == second
    assert tuple(map(len, first)) == (8, 2)
    assert {row["premise"] for group in first for row in group} == {
        f"example {index}" for index in range(10)
    }


def test_standalone_tokenises_and_keeps_binary_labels(tmp_path: Path) -> None:
    """Map genuine text through the Dataset and its local Hugging Face tokenizer."""
    path = tmp_path / "examples.jsonl"
    path.write_text(
        '{"premise":"first","hypothesis":"claim","label":1}\n'
        '{"premise":"","hypothesis":"ignored","label":0}\n'
        '{"premise":"second","hypothesis":"claim","label":0}\n',
        encoding="utf-8",
    )
    vocab = {
        word: index
        for index, word in enumerate(
            ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "first", "second", "claim")
        )
    }
    tokenizer = BertTokenizer(vocab=vocab)
    dataset = runner.tokenize_dataset(runner.load_jsonl(path), tokenizer, max_length=64)
    assert len(dataset) == 2
    assert dataset[0]["input_ids"].shape == (64,)
    assert [dataset[index]["labels"].item() for index in range(2)] == [1, 0]


def test_standalone_evaluates_actual_trainer_predictions() -> None:
    """Keep the meaning of the scikit-learn metrics on a genuine EvalPrediction."""
    logits = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    reference = np.array([1, 0, 1, 0])
    prediction = EvalPrediction(predictions=logits, label_ids=reference)
    assert runner.compute_metrics(prediction) == {"balanced_accuracy": 0.5, "f1": 0.5}


def test_standalone_empty_training_skips_model_loading(tmp_path: Path) -> None:
    """Run the public training entry point through its empty-input boundary."""
    train, evaluate = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    for path in (train, evaluate):
        path.write_text("", encoding="utf-8")
    model = tmp_path / "model"
    assert runner.train_model("empty", str(train), evaluate, model) is None
    assert not model.exists()


def test_standalone_help_preserves_the_public_script_entry() -> None:
    """Use the standalone script's argument parser without fetching a model."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/run_training_standalone.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert "--max-samples" in result.stdout
