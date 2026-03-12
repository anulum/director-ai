#!/usr/bin/env python3
"""Fine-tune FactCG-DeBERTa-v3-Large on XNLI English for cross-lingual NLI.

XNLI (Conneau et al. 2018): ~15K English NLI pairs (subset of MultiNLI
professionally translated to 14 languages). English split only.

Usage:
    python run_xnli_training.py [--resume]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import balanced_accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "yaxili96/FactCG-DeBERTa-v3-Large"
OUTPUT_DIR = "/home/director-ai/models/factcg-xnli"
TEMPLATE = (
    "{text_a}\n\nChoose your answer: based on the paragraph above "
    'can we conclude that "{text_b}"?\n\nOPTIONS:\n- Yes\n- No\n'
    "I think the answer is "
)


def load_xnli():
    ds = load_dataset("facebook/xnli", "en")

    def convert(example):
        # XNLI: 0=entailment, 1=neutral, 2=contradiction → binary
        label = 1 if example["label"] == 0 else 0
        text = TEMPLATE.format(text_a=example["premise"], text_b=example["hypothesis"])
        return {"text": text, "label": label}

    train = ds["train"].map(convert, remove_columns=ds["train"].column_names)
    val = ds["validation"].map(convert, remove_columns=ds["validation"].column_names)
    test = ds["test"].map(convert, remove_columns=ds["test"].column_names)

    print(f"XNLI loaded: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def tokenize_fn(tokenizer, max_length=512):
    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length, padding=False)
    return _tok


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": float((preds == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    print("=== XNLI Fine-Tuning ===")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    train_ds, val_ds, test_ds = load_xnli()
    tok_fn = tokenize_fn(tokenizer)
    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tok_fn, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tok_fn, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR, num_train_epochs=5, per_device_train_batch_size=16,
        per_device_eval_batch_size=32, learning_rate=2e-5, weight_decay=0.01,
        warmup_ratio=0.1, eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="balanced_accuracy", greater_is_better=True,
        fp16=True, logging_steps=50, report_to="none",
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_ds,
        eval_dataset=val_ds, tokenizer=tokenizer, compute_metrics=compute_metrics,
    )

    start = time.time()
    trainer.train(resume_from_checkpoint=True) if args.resume else trainer.train()
    elapsed = time.time() - start

    test_result = trainer.evaluate(test_ds)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    result = {
        "dataset": "xnli_en", "base_model": BASE_MODEL,
        "test_balanced_accuracy": test_result["eval_balanced_accuracy"],
        "test_accuracy": test_result["eval_accuracy"],
        "training_time_minutes": round(elapsed / 60, 1), "epochs": 5, "status": "COMPLETE",
    }
    with open(os.path.join(OUTPUT_DIR, "training_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nCOMPLETE — bal_acc={result['test_balanced_accuracy']:.4f}")


if __name__ == "__main__":
    main()
