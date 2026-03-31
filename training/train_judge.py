# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Train Binary Judge Classifier
"""
Fine-tune DeBERTa-v3-base as a 2-class (approve/reject) judge for
borderline NLI cases. Replaces external LLM judge (GPT-4o-mini / Haiku).

Input format per sample::

    NLI divergence: 0.45
    Context: {premise[:400]}
    Response: {hypothesis[:400]}

Output: 2-class softmax (0=approve, 1=reject).

Hyperparameters tuned for GTX 1060 6GB (gradient checkpointing + FP16).

Usage::

    python training/train_judge.py
    python training/train_judge.py --epochs 3 --batch-size 8
    python training/train_judge.py --base-model microsoft/deberta-v3-base
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data_judge"
OUTPUT_DIR = Path(__file__).parent / "output" / "deberta-v3-base-judge"
BASE_MODEL = "microsoft/deberta-v3-base"
MAX_LENGTH = 384
LABEL_NAMES = ["approve", "reject"]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary", pos_label=1),
        "precision": precision_score(labels, preds, pos_label=1),
        "recall": recall_score(labels, preds, pos_label=1),
        "f1_approve": f1_score(labels, preds, average="binary", pos_label=0),
    }


def main():
    parser = argparse.ArgumentParser(description="Train binary judge classifier")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.15)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", action="store_false", dest="fp16")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_false",
        dest="gradient_checkpointing",
    )
    args = parser.parse_args()

    logger.info("Loading judge dataset from %s", DATA_DIR)
    ds = load_from_disk(str(DATA_DIR))
    train_ds = ds["train"]
    eval_ds = ds["eval"]
    logger.info("Train: %d, Eval: %d", len(train_ds), len(eval_ds))

    logger.info("Loading tokenizer: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)

    def tokenize(example):
        return tokenizer(
            example["text"],
            max_length=args.max_length,
            truncation=True,
            padding=False,
        )

    logger.info("Tokenizing...")
    train_ds = train_ds.map(tokenize, batched=True, desc="Tokenizing train")
    eval_ds = eval_ds.map(tokenize, batched=True, desc="Tokenizing eval")

    logger.info("Loading model: %s (2-class)", args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label={0: "approve", 1: "reject"},
        label2id={"approve": 0, "reject": 1},
        ignore_mismatched_sizes=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    effective_batch = args.batch_size * args.grad_accum
    total_steps = (len(train_ds) * args.epochs) // effective_batch
    logger.info("Effective batch: %d, Total steps: ~%d", effective_batch, total_steps)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=args.fp16 and torch.cuda.is_available(),
        logging_steps=100,
        seed=args.seed,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Evaluating best model...")
    final_metrics = trainer.evaluate()
    logger.info("Final metrics: %s", final_metrics)

    logger.info("Saving model to %s", OUTPUT_DIR)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    import json

    (OUTPUT_DIR / "judge_metrics.json").write_text(json.dumps(final_metrics, indent=2))
    logger.info("Done. Model saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
