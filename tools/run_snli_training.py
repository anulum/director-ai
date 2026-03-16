# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — SNLI Fine-tuning (UpCloud GPU)

"""SNLI-only fine-tuning on UpCloud L40S.

Fixes from JarvisLabs failure:
- dataloader_num_workers=2 (was 4, caused OOM kill)
- gradient_accumulation_steps=2 with batch_size=12 (same effective batch=24)
- Checkpoint resume if interrupted

Usage::

    python run_snli_training.py
    python run_snli_training.py --resume  # resume from last checkpoint
"""

import json
import logging
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import balanced_accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("train-snli")

BASE_MODEL = "yaxili96/FactCG-DeBERTa-v3-Large"
FACTCG_TEMPLATE = (
    "{premise}\n\nChoose your answer: based on the paragraph above "
    'can we conclude that "{hypothesis}"?\n\nOPTIONS:\n- Yes\n- No\n'
    "I think the answer is "
)
DATA_DIR = Path("/home/director-ai/data")
MODELS_DIR = Path("/home/director-ai/models")
OUTPUT_DIR = MODELS_DIR / "factcg-snli"
MAX_SAMPLES = 9_999_999


def nli_to_binary(label_raw):
    if isinstance(label_raw, str):
        low = label_raw.lower()
        if low in ("entailment", "supports"):
            return 1
        if low in ("contradiction", "refutes"):
            return 0
        return None
    label_int = int(label_raw)
    if label_int == 0:
        return 1
    if label_int == 2:
        return 0
    return None


def prepare_snli():
    logger.info("Loading SNLI...")
    rows = []
    for split in ["train", "validation", "test"]:
        ds = load_dataset("snli", split=split, trust_remote_code=True)
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

    logger.info("SNLI: %d binary samples", len(rows))
    rng = random.Random(42)
    rng.shuffle(rows)
    if len(rows) > MAX_SAMPLES:
        rows = rows[:MAX_SAMPLES]

    n_eval = max(1, int(len(rows) * 0.1))
    train, eval_ = rows[n_eval:], rows[:n_eval]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in [("snli_train.jsonl", train), ("snli_eval.jsonl", eval_)]:
        with open(DATA_DIR / name, "w") as f:
            f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in data)
        logger.info("Wrote %d samples to %s", len(data), DATA_DIR / name)

    return len(train), len(eval_)


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tokenize_dataset(rows, tokenizer, max_length=512):
    texts = [
        FACTCG_TEMPLATE.format(premise=r["premise"], hypothesis=r["hypothesis"])
        for r in rows
        if r["premise"] and r["hypothesis"]
    ]
    labels = [r["label"] for r in rows if r["premise"] and r["hypothesis"]]
    ds = Dataset.from_dict({"text": texts, "labels": labels})

    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    ds = ds.map(_tokenize, batched=True, batch_size=256, remove_columns=["text"])
    ds.set_format("torch")
    return ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary", zero_division=0),
    }


def train_snli(resume_from_checkpoint=False):
    train_path = DATA_DIR / "snli_train.jsonl"
    eval_path = DATA_DIR / "snli_eval.jsonl"

    if not train_path.exists():
        logger.info("Data not found, preparing...")
        prepare_snli()

    train_rows = load_jsonl(train_path)
    eval_rows = load_jsonl(eval_path)
    logger.info("Samples: %d train, %d eval", len(train_rows), len(eval_rows))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    train_ds = tokenize_dataset(train_rows, tokenizer)
    eval_ds = tokenize_dataset(eval_rows, tokenizer)

    n_steps = len(train_rows) // 12 * 3  # effective batch=24 via grad_accum=2
    eval_save_steps = max(200, n_steps // 15)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=12,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=48,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=True,
        seed=42,
        eval_strategy="steps",
        eval_steps=eval_save_steps,
        save_strategy="steps",
        save_steps=eval_save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="balanced_accuracy",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    elapsed = time.time() - t0

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    eval_metrics = trainer.evaluate()
    bal_acc = eval_metrics.get("eval_balanced_accuracy", 0.0)
    logger.info("Best balanced accuracy: %.1f%%", bal_acc * 100)
    logger.info("Training time: %.1f minutes", elapsed / 60)

    result = {
        "name": "snli",
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
    with open(OUTPUT_DIR / "training_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 60)
    print("  SNLI TRAINING COMPLETE")
    print("=" * 60)
    print(
        f"  bal_acc={bal_acc:.1%}, f1={result['f1']:.3f}, time={result['training_time_min']:.1f}min",
    )
    print(
        f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
    )
    print("=" * 60)

    subprocess.run(
        [
            "tar",
            "czf",
            "/home/director-ai/factcg-snli.tar.gz",
            "-C",
            str(MODELS_DIR),
            "factcg-snli",
        ],
        check=True,
    )
    size = os.path.getsize("/home/director-ai/factcg-snli.tar.gz") / 1e6
    logger.info("Model tarball: /home/director-ai/factcg-snli.tar.gz (%.0f MB)", size)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    cli_args = parser.parse_args()

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram = (
        torch.cuda.get_device_properties(0).total_memory / 1e9
        if torch.cuda.is_available()
        else 0
    )
    print(f"GPU: {gpu_name}, VRAM: {vram:.1f} GB")

    train_snli(resume_from_checkpoint=cli_args.resume)
