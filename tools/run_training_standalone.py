# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Standalone GPU Fine-tuning Script

"""Self-contained fine-tuning script for JarvisLabs (requires Python 3.11+).

Requires: transformers, datasets (with PyArrow), torch, scikit-learn, numpy

Datasets:
  - VitaminC (370K fact-verification pairs) → factcg-vitaminc
  - SNLI (550K general NLI) → factcg-snli
  - ContractNLI (6.8K legal NLI) → factcg-legal

The command exits with an error if no model is trained successfully.

Usage::

    python run_training_standalone.py
    python run_training_standalone.py --max-samples 50000  # faster
"""

import json
import logging
import os
import random
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Literal, Protocol, SupportsInt, TypedDict, cast

import numpy as np
import torch
from datasets import ClassLabel, Dataset, load_dataset
from numpy.typing import NDArray
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


class _TrainingExample(TypedDict):
    """One binary fact verification training example."""

    premise: str
    hypothesis: str
    label: int


class _TrainingResult(TypedDict):
    """Checkpoint metrics stored in the standalone training result."""

    name: str
    train_samples: int
    eval_samples: int
    balanced_accuracy: float
    f1: float
    training_time_min: float
    eval_metrics: dict[str, float]


class _Tokenizer(Protocol):
    """Tokenizer operations used by the batched Dataset mapping."""

    def __call__(
        self,
        text: Sequence[str],
        *,
        truncation: bool,
        padding: Literal["max_length"],
        max_length: int,
    ) -> Mapping[str, object]:
        """Encode a batch to the requested fixed sequence length."""
        ...


class _ClassificationMetrics(Protocol):
    """The two untyped scikit-learn entry points used for evaluation."""

    def balanced_accuracy_score(
        self, labels: NDArray[np.generic], predictions: NDArray[np.generic]
    ) -> float:
        """Average recall across the classes present in the reference labels."""
        ...

    def f1_score(
        self,
        labels: NDArray[np.generic],
        predictions: NDArray[np.generic],
        *,
        average: Literal["binary"],
        zero_division: Literal[0],
    ) -> float:
        """Compute positive-class F1, with zero for undefined scores."""
        ...


_metrics = cast(_ClassificationMetrics, import_module("sklearn.metrics"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("train")

BASE_MODEL = "yaxili96/FactCG-DeBERTa-v3-Large"
FACTCG_TEMPLATE = (
    "{premise}\n\nChoose your answer: based on the paragraph above "
    'can we conclude that "{hypothesis}"?\n\nOPTIONS:\n- Yes\n- No\n'
    "I think the answer is "
)
DATA_DIR = Path("/home/director-ai/data")
MODELS_DIR = Path("/home/director-ai/models")

MAX_SAMPLES = 100_000  # per dataset, overridable via --max-samples


def write_jsonl(rows: Sequence[_TrainingExample], path: Path) -> None:
    """Write fact verification examples as UTF-8 JSON Lines.

    Parameters
    ----------
    rows : Sequence of _TrainingExample
        Binary examples to write in input order.
    path : Path
        Destination file. Parent directories are created if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    logger.info("Wrote %d samples to %s", len(rows), path)


def split_data(
    rows: list[_TrainingExample], eval_ratio: float = 0.1, seed: int = 42
) -> tuple[list[_TrainingExample], list[_TrainingExample]]:
    """Shuffle examples in place and reserve the leading evaluation fraction.

    Parameters
    ----------
    rows : list of _TrainingExample
        Examples to shuffle and split.
    eval_ratio : float
        Fraction of examples to reserve, with one reserved when available.
    seed : int
        Random seed for a reproducible partition.

    Returns
    -------
    tuple of list
        Training examples followed by evaluation examples.
    """
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_eval = max(1, int(len(rows) * eval_ratio))
    return rows[n_eval:], rows[:n_eval]


# ── Data Preparation ───────────────────────────────────────────────


def nli_to_binary(label_raw: str | SupportsInt) -> int | None:
    """Map NLI entailment and contradiction labels to binary outcomes.

    Parameters
    ----------
    label_raw : str or SupportsInt
        NLI label text or a numeric dataset label.

    Returns
    -------
    int or None
        One for entailment, zero for contradiction and None for excluded labels.
    """
    if isinstance(label_raw, str):
        low = label_raw.lower()
        if low in ("entailment", "supports"):
            return 1
        if low in ("contradiction", "refutes"):
            return 0
        return None
    label_int = int(label_raw)
    if label_int == 0:
        return 1  # entailment
    if label_int == 2:
        return 0  # contradiction
    return None  # neutral (-1, 1)


def prepare_vitaminc(max_samples: int) -> tuple[int, int]:
    """Prepare labelled VitaminC claims and Wikipedia evidence.

    Parameters
    ----------
    max_samples : int
        Maximum number of accepted binary examples across the dataset splits.

    Returns
    -------
    tuple of int
        Numbers written to the training and evaluation files.
    """
    logger.info("Loading VitaminC...")
    rows: list[_TrainingExample] = []
    for split in ["train", "validation", "test"]:
        try:
            ds = load_dataset("tals/vitaminc", split=split)
            for item in ds:
                evidence = item.get("evidence", "")
                claim = item.get("claim", "")
                label_raw = item.get("label")
                if not evidence or not claim or label_raw is None:
                    continue
                label = nli_to_binary(label_raw)
                if label is not None:
                    rows.append(
                        {"premise": evidence, "hypothesis": claim, "label": label},
                    )
        except Exception as e:
            logger.warning("VitaminC split %s failed: %s", split, e)

    if len(rows) > max_samples:
        rng = random.Random(42)
        rng.shuffle(rows)
        rows = rows[:max_samples]

    logger.info("VitaminC: %d binary samples", len(rows))
    train, eval_ = split_data(rows)
    write_jsonl(train, DATA_DIR / "vitaminc_train.jsonl")
    write_jsonl(eval_, DATA_DIR / "vitaminc_eval.jsonl")
    return len(train), len(eval_)


def prepare_snli(max_samples: int) -> tuple[int, int]:
    """Prepare labelled Stanford Natural Language Inference examples.

    Parameters
    ----------
    max_samples : int
        Maximum number of accepted binary examples across the dataset splits.

    Returns
    -------
    tuple of int
        Numbers written to the training and evaluation files.
    """
    logger.info("Loading SNLI...")
    rows: list[_TrainingExample] = []
    for split in ["train", "validation", "test"]:
        try:
            ds = load_dataset("stanfordnlp/snli", split=split)
            for item in ds:
                premise = item.get("premise", "")
                hypothesis = item.get("hypothesis", "")
                label_raw = item.get("label")
                if not premise or not hypothesis or label_raw is None:
                    continue
                label = nli_to_binary(label_raw)
                if label is not None:
                    rows.append(
                        {"premise": premise, "hypothesis": hypothesis, "label": label},
                    )
        except Exception as e:
            logger.warning("SNLI split %s failed: %s", split, e)

    if len(rows) > max_samples:
        rng = random.Random(42)
        rng.shuffle(rows)
        rows = rows[:max_samples]

    logger.info("SNLI: %d binary samples", len(rows))
    train, eval_ = split_data(rows)
    write_jsonl(train, DATA_DIR / "snli_train.jsonl")
    write_jsonl(eval_, DATA_DIR / "snli_eval.jsonl")
    return len(train), len(eval_)


def prepare_legal(max_samples: int) -> tuple[int, int]:
    """Prepare labelled ContractNLI examples.

    Parameters
    ----------
    max_samples : int
        Maximum number of accepted binary examples across the dataset splits.

    Returns
    -------
    tuple of int
        Numbers written to the training and evaluation files.
    """
    logger.info("Loading ContractNLI...")
    parquet_root = (
        "hf://datasets/kiddothe2b/contract-nli@refs%2Fconvert%2Fparquet/contractnli_a"
    )
    rows: list[_TrainingExample] = []
    for split in ["train", "test"]:
        try:
            ds = load_dataset(
                "parquet",
                data_files={split: f"{parquet_root}/{split}/*.parquet"},
                split=split,
            )
            labels = ds.features["label"]
            if not isinstance(labels, ClassLabel):
                raise ValueError("ContractNLI requires named class labels")
            for item in ds:
                premise = item.get("premise", "")
                hypothesis = item.get("hypothesis", "")
                label_raw = item.get("label")
                if not premise or not hypothesis or not isinstance(label_raw, int):
                    continue
                if not 0 <= label_raw < labels.num_classes:
                    continue
                label = nli_to_binary(labels.int2str(label_raw))
                if label is not None:
                    rows.append(
                        {"premise": premise, "hypothesis": hypothesis, "label": label},
                    )
        except Exception as e:
            logger.warning("ContractNLI split %s failed: %s", split, e)

    if len(rows) > max_samples:
        rng = random.Random(42)
        rng.shuffle(rows)
        rows = rows[:max_samples]

    logger.info("ContractNLI: %d binary samples", len(rows))
    train, eval_ = split_data(rows)
    write_jsonl(train, DATA_DIR / "legal_train.jsonl")
    write_jsonl(eval_, DATA_DIR / "legal_eval.jsonl")
    return len(train), len(eval_)


# ── Training ───────────────────────────────────────────────────────


def load_jsonl(path: str | Path) -> list[_TrainingExample]:
    """Read binary examples produced by the standalone data preparation.

    Parameters
    ----------
    path : str or Path
        UTF-8 JSON Lines file with premise, hypothesis and integer label fields.

    Returns
    -------
    list of _TrainingExample
        Validated examples in file order. Empty lines are skipped.

    Raises
    ------
    ValueError
        If a nonempty line cannot be decoded as a training example.
    """
    rows: list[_TrainingExample] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row: object = json.loads(line)
                if not (
                    isinstance(row, dict)
                    and isinstance(row.get("premise"), str)
                    and isinstance(row.get("hypothesis"), str)
                    and isinstance(row.get("label"), int)
                ):
                    raise ValueError(f"Invalid training example in {path}")
                rows.append(cast(_TrainingExample, row))
    return rows


def tokenize_dataset(
    rows: Sequence[_TrainingExample], tokenizer: _Tokenizer, max_length: int = 512
) -> Dataset:
    """Encode nonempty examples with the FactCG prompt and retain their labels.

    Bind the in-memory Arrow table to a content fingerprint so tokenization
    can proceed independently of Arrow's Python pickling hooks.

    Parameters
    ----------
    rows : Sequence of _TrainingExample
        Labelled premise and hypothesis pairs.
    tokenizer : _Tokenizer
        Batch tokenizer for the selected training model.
    max_length : int
        Fixed token sequence length after truncation and padding.

    Returns
    -------
    datasets.Dataset
        Encoded examples in PyTorch format.
    """
    texts: list[str] = []
    labels: list[int] = []
    fingerprint = sha256()
    for row in rows:
        if row["premise"] and row["hypothesis"]:
            text = FACTCG_TEMPLATE.format(
                premise=row["premise"], hypothesis=row["hypothesis"]
            )
            label = row["label"]
            texts.append(text)
            labels.append(label)
            fingerprint.update(json.dumps([text, label]).encode("utf-8"))

    table = import_module("pyarrow").Table.from_pydict(
        {"text": texts, "labels": labels}
    )
    ds = Dataset(table, fingerprint=fingerprint.hexdigest())

    def _tokenize(batch: Mapping[str, Sequence[str]]) -> Mapping[str, object]:
        """Encode the text column of one batch."""
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    ds = ds.map(_tokenize, batched=True, batch_size=256, remove_columns=["text"])
    ds.set_format("torch")
    return ds


def compute_metrics(eval_pred: Iterable[NDArray[np.generic]]) -> dict[str, float]:
    """Score model predictions with balanced accuracy and binary F1.

    Parameters
    ----------
    eval_pred : Iterable of numpy.ndarray
        Batch logits followed by their reference labels.

    Returns
    -------
    dict of str to float
        Balanced accuracy and positive-class F1.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "balanced_accuracy": _metrics.balanced_accuracy_score(labels, preds),
        "f1": _metrics.f1_score(labels, preds, average="binary", zero_division=0),
    }


def train_model(
    name: str, train_path: str | Path, eval_path: str | Path, output_dir: Path
) -> _TrainingResult | None:
    """Fine-tune the standalone classifier and persist checkpoint metrics.

    Parameters
    ----------
    name : str
        Dataset name included in logs and result metadata.
    train_path : str or Path
        Prepared training examples.
    eval_path : str or Path
        Prepared evaluation examples.
    output_dir : Path
        Destination for the model, tokenizer and training result.

    Returns
    -------
    _TrainingResult or None
        Evaluation metrics and sample counts, or None for empty training data.
    """
    logger.info("=== Training: %s ===", name)
    train_rows = load_jsonl(train_path)
    eval_rows = load_jsonl(eval_path)
    if not train_rows:
        logger.warning("Skipping %s: 0 train samples", name)
        return None
    logger.info("Samples: %d train, %d eval", len(train_rows), len(eval_rows))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    train_ds = tokenize_dataset(train_rows, tokenizer)
    eval_ds = tokenize_dataset(eval_rows, tokenizer)

    n_steps = len(train_rows) // 24 * 3  # rough total steps
    eval_save_steps = max(200, n_steps // 15)  # ~15 evals per run

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=24,
        per_device_eval_batch_size=48,
        learning_rate=2e-5,
        warmup_steps=0.1,
        weight_decay=0.01,
        fp16=True,
        seed=42,
        eval_strategy="steps",
        eval_steps=eval_save_steps,
        save_strategy="steps",
        save_steps=eval_save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="balanced_accuracy",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    eval_metrics: dict[str, float] = trainer.evaluate()
    bal_acc = eval_metrics.get("eval_balanced_accuracy", 0.0)
    logger.info("Best balanced accuracy: %.1f%%", bal_acc * 100)
    logger.info("Training time: %.1f minutes", elapsed / 60)

    result: _TrainingResult = {
        "name": name,
        "train_samples": len(train_rows),
        "eval_samples": len(eval_rows),
        "balanced_accuracy": round(bal_acc, 4),
        "f1": round(eval_metrics.get("eval_f1", 0.0), 4),
        "training_time_min": round(elapsed / 60, 1),
        "eval_metrics": {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in eval_metrics.items()
        },
    }
    with open(output_dir / "training_result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ── Main ───────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    cli_args = parser.parse_args()
    max_samples = cli_args.max_samples

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram = (
        torch.cuda.get_device_properties(0).total_memory / 1e9
        if torch.cuda.is_available()
        else 0
    )
    print(f"GPU: {gpu_name}, VRAM: {vram:.1f} GB")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Prepare data
    logger.info("=== Step 1: Preparing datasets (max %d samples each) ===", max_samples)
    stats = {}
    prep_fns = [
        ("vitaminc", prepare_vitaminc),
        ("snli", prepare_snli),
        ("legal", prepare_legal),
    ]
    for name, prep_fn in prep_fns:
        try:
            n_train, n_eval = prep_fn(max_samples)
            stats[name] = {"train": n_train, "eval": n_eval}
        except Exception as e:
            logger.error("%s prep failed: %s", name, e)
            import traceback

            traceback.print_exc()
    logger.info("Data stats: %s", stats)

    # Step 2: Train
    results = {}
    total_start = time.time()

    runs = [
        (
            "vitaminc",
            DATA_DIR / "vitaminc_train.jsonl",
            DATA_DIR / "vitaminc_eval.jsonl",
            MODELS_DIR / "factcg-vitaminc",
        ),
        (
            "snli",
            DATA_DIR / "snli_train.jsonl",
            DATA_DIR / "snli_eval.jsonl",
            MODELS_DIR / "factcg-snli",
        ),
        (
            "legal",
            DATA_DIR / "legal_train.jsonl",
            DATA_DIR / "legal_eval.jsonl",
            MODELS_DIR / "factcg-legal",
        ),
    ]

    for name, train_path, eval_path, output_dir in runs:
        if not train_path.exists():
            logger.warning("Skipping %s: data file missing", name)
            continue
        try:
            result = train_model(name, train_path, eval_path, output_dir)
            if result:
                results[name] = result
        except Exception as e:
            logger.error("%s training failed: %s", name, e)
            import traceback

            traceback.print_exc()

    total_time = (time.time() - total_start) / 60

    # Step 3: Summary
    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        print(
            f"  {name}: bal_acc={r['balanced_accuracy']:.1%}, f1={r['f1']:.3f}, time={r['training_time_min']:.1f}min",
        )
    print(f"  Total time: {total_time:.1f} minutes")
    print(f"  GPU: {gpu_name}")
    print("=" * 60)

    if not results:
        raise SystemExit("No models were trained successfully.")

    summary = {
        "results": results,
        "total_time_min": round(total_time, 1),
        "gpu": gpu_name,
        "max_samples_per_dataset": max_samples,
    }
    with open(MODELS_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Create tarball for download
    subprocess.run(
        [
            "tar",
            "czf",
            "/home/director-ai/models_trained.tar.gz",
            "-C",
            str(MODELS_DIR),
            ".",
        ],
        check=True,
    )
    size = os.path.getsize("/home/director-ai/models_trained.tar.gz") / 1e6
    logger.info(
        "Models tarball: /home/director-ai/models_trained.tar.gz (%.0f MB)",
        size,
    )
