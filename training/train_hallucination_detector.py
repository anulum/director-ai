#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Fine-Tune DeBERTa for Hallucination Detection
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Fine-tune ``microsoft/deberta-v3-large`` (continuing from MoritzLaurer's
mnli-fever-anli-ling-wanli checkpoint) on the unified hallucination
detection dataset built by ``data_pipeline.py``.

Usage::

    python training/train_hallucination_detector.py
    # Output: training/output/deberta-v3-large-hallucination/

Requires: ``pip install director-ai[train]``
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output" / "deberta-v3-large-hallucination"
BASE_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
MAX_LENGTH = 512
LABEL_NAMES = ["entailment", "neutral", "contradiction"]


def compute_metrics(eval_pred):
    """Compute accuracy and per-class F1 for the Trainer."""
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_per = f1_score(labels, preds, average=None, labels=[0, 1, 2])
    return {
        "accuracy": acc,
        "f1": f1_macro,
        "f1_entailment": f1_per[0],
        "f1_neutral": f1_per[1],
        "f1_contradiction": f1_per[2],
    }


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy loss."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        if class_weights is not None:
            self._class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self._class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self._class_weights is not None:
            w = self._class_weights.to(logits.device)
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=w)
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"{DATA_DIR} not found — run data_pipeline.py first")

    logger.info("Loading dataset from %s", DATA_DIR)
    dataset = load_from_disk(str(DATA_DIR))

    logger.info("Loading tokenizer and model: %s", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=3)

    def tokenize(batch):
        return tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

    logger.info("Tokenizing ...")
    tokenized = dataset.map(
        tokenize, batched=True, remove_columns=["premise", "hypothesis", "source"]
    )

    # Class weights
    train_labels = np.array(tokenized["train"]["label"])
    weights = compute_class_weight(
        "balanced", classes=np.array([0, 1, 2]), y=train_labels
    )
    logger.info("Class weights: %s", dict(zip(LABEL_NAMES, weights)))

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        warmup_ratio=0.06,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        report_to="none",
        dataloader_num_workers=4,
        label_names=["labels"],
    )

    trainer = WeightedTrainer(
        class_weights=weights,
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["eval"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    logger.info("Starting training ...")
    trainer.train()

    logger.info("Saving best model to %s", OUTPUT_DIR)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    logger.info("Final evaluation:")
    metrics = trainer.evaluate()
    for k, v in sorted(metrics.items()):
        logger.info("  %s: %.4f", k, v)


if __name__ == "__main__":
    main()
