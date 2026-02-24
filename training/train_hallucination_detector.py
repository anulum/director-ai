#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Fine-Tune DeBERTa for Hallucination Detection
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Fine-tune DeBERTa-v3-base on a balanced 100K subset of the unified
hallucination detection dataset built by ``data_pipeline.py``.

Usage::

    python training/train_hallucination_detector.py
    # Output: training/output/deberta-v3-base-hallucination/

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
OUTPUT_DIR = Path(__file__).parent / "output" / "deberta-v3-base-hallucination"
BASE_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
MAX_LENGTH = 256
SUBSET_SIZE = 100_000
EVAL_RATIO = 0.1
LABEL_NAMES = ["entailment", "neutral", "contradiction"]


def compute_metrics(eval_pred):
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


def subsample_balanced(dataset, n_total, seed=42):
    """Stratified subsample: equal per-source, then per-label within source."""
    sources = set(dataset["source"])
    per_source = n_total // len(sources)
    indices = []
    rng = np.random.default_rng(seed)

    for src in sorted(sources):
        src_mask = np.array(dataset["source"]) == src
        src_indices = np.where(src_mask)[0]
        take = min(per_source, len(src_indices))
        chosen = rng.choice(src_indices, size=take, replace=False)
        indices.extend(chosen.tolist())

    rng.shuffle(indices)
    return dataset.select(indices[:n_total])


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"{DATA_DIR} not found — run data_pipeline.py first")

    logger.info("Loading dataset from %s", DATA_DIR)
    full_dataset = load_from_disk(str(DATA_DIR))

    # Subsample from the train split
    train_full = full_dataset["train"]
    logger.info("Full train set: %d examples, subsampling to %d", len(train_full), SUBSET_SIZE)
    train_sub = subsample_balanced(train_full, SUBSET_SIZE)

    eval_size = int(SUBSET_SIZE * EVAL_RATIO)
    eval_full = full_dataset["eval"]
    eval_sub = eval_full.select(range(min(eval_size, len(eval_full))))

    logger.info("Subset: train=%d, eval=%d", len(train_sub), len(eval_sub))

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
    tok_train = train_sub.map(tokenize, batched=True, remove_columns=["premise", "hypothesis", "source"])
    tok_eval = eval_sub.map(tokenize, batched=True, remove_columns=["premise", "hypothesis", "source"])

    # Class weights from the subset
    train_labels = np.array(tok_train["label"])
    weights = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=train_labels)
    logger.info("Class weights: %s", dict(zip(LABEL_NAMES, weights)))

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,
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
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=0,
        label_names=["labels"],
    )

    trainer = WeightedTrainer(
        class_weights=weights,
        model=model,
        args=training_args,
        train_dataset=tok_train,
        eval_dataset=tok_eval,
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
