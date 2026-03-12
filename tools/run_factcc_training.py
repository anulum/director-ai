#!/usr/bin/env python3
"""Fine-tune FactCG-DeBERTa-v3-Large on FactCC for summarization consistency.

FactCC (Kryscinski et al. 2020): ~93K document-summary sentence pairs with
factual consistency labels. Directly targets AggreFact-CNN/XSum subsets.

Usage:
    python run_factcc_training.py [--resume]
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
OUTPUT_DIR = "/home/director-ai/models/factcg-factcc"
TEMPLATE = (
    "{text_a}\n\nChoose your answer: based on the paragraph above "
    'can we conclude that "{text_b}"?\n\nOPTIONS:\n- Yes\n- No\n'
    "I think the answer is "
)


def load_factcc():
    ds = load_dataset("gfhayworth/factcc", trust_remote_code=True)

    def convert(example):
        doc = example.get("text", "") or example.get("document", "") or ""
        claim = example.get("claim", "") or example.get("summary", "") or ""
        label_raw = example.get("label", 0)
        # FactCC: CORRECT=1, INCORRECT=0 (or similar)
        if isinstance(label_raw, str):
            label = 1 if label_raw.upper() in ("CORRECT", "CONSISTENT", "1") else 0
        else:
            label = int(label_raw)
        doc_trunc = doc[:2000]
        text = TEMPLATE.format(text_a=doc_trunc, text_b=claim)
        return {"text": text, "label": label}

    splits = list(ds.keys())
    train_raw = ds["train"] if "train" in splits else ds[splits[0]]
    train = train_raw.map(convert, remove_columns=train_raw.column_names)

    if "validation" in splits:
        val = ds["validation"].map(convert, remove_columns=ds["validation"].column_names)
    else:
        split = train.train_test_split(test_size=0.1, seed=42)
        train = split["train"]
        val = split["test"]

    if "test" in splits:
        test = ds["test"].map(convert, remove_columns=ds["test"].column_names)
    else:
        val_split = val.train_test_split(test_size=0.5, seed=42)
        val = val_split["train"]
        test = val_split["test"]

    print(f"FactCC loaded: train={len(train)}, val={len(val)}, test={len(test)}")
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

    print("=== FactCC Fine-Tuning ===")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    train_ds, val_ds, test_ds = load_factcc()
    tok_fn = tokenize_fn(tokenizer)
    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tok_fn, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tok_fn, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR, num_train_epochs=3, per_device_train_batch_size=16,
        per_device_eval_batch_size=32, gradient_accumulation_steps=2,
        learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.1,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="balanced_accuracy",
        greater_is_better=True, fp16=True, dataloader_num_workers=2,
        logging_steps=100, report_to="none",
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
        "dataset": "factcc", "base_model": BASE_MODEL,
        "test_balanced_accuracy": test_result["eval_balanced_accuracy"],
        "test_accuracy": test_result["eval_accuracy"],
        "training_time_minutes": round(elapsed / 60, 1), "epochs": 3, "status": "COMPLETE",
    }
    with open(os.path.join(OUTPUT_DIR, "training_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nCOMPLETE — bal_acc={result['test_balanced_accuracy']:.4f}")


if __name__ == "__main__":
    main()
