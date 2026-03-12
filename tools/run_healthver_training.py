#!/usr/bin/env python3
"""Fine-tune FactCG-DeBERTa-v3-Large on PUBHEALTH for health claim verification.

PUBHEALTH (Kotonya & Toni 2020): ~12K health claims with evidence text
and expert verdicts. Covers COVID, nutrition, disease, drug safety.
Complements MedNLI (clinical) with public health misinformation detection.

Binary: true=1, false=0, mixture/unproven dropped.

Usage:
    python run_healthver_training.py [--resume]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import balanced_accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "yaxili96/FactCG-DeBERTa-v3-Large"
OUTPUT_DIR = "/home/director-ai/models/factcg-healthver"
TEMPLATE = (
    "{text_a}\n\nChoose your answer: based on the paragraph above "
    'can we conclude that "{text_b}"?\n\nOPTIONS:\n- Yes\n- No\n'
    "I think the answer is "
)

LABEL_MAP = {"true": 1, "false": 0, "mixture": None, "unproven": None}


def load_healthver():
    """Load PUBHEALTH and convert to binary NLI."""
    rng = random.Random(42)
    ds = load_dataset("OpenMed/PubHealth-Processed")

    rows_by_split = {"train": [], "validation": [], "test": []}
    for split in rows_by_split:
        if split not in ds:
            continue
        for ex in ds[split]:
            main_text = (ex.get("main_text") or "").strip()
            claim = (ex.get("claim") or "").strip()
            label_str = (ex.get("label") or "").strip().lower()
            label = LABEL_MAP.get(label_str)
            if label is None or not main_text or not claim:
                continue
            text = TEMPLATE.format(text_a=main_text[:2000], text_b=claim)
            rows_by_split[split].append({"text": text, "label": label})

    rng.shuffle(rows_by_split["train"])
    for s, rows in rows_by_split.items():
        print(f"  {s}: {len(rows)}")
    return tuple(Dataset.from_list(rows_by_split[s]) for s in ["train", "validation", "test"])


def tokenize_fn(tokenizer, max_length=512):
    def _tok(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=max_length, padding=False
        )

    return _tok


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    bal_acc = balanced_accuracy_score(labels, preds)
    acc = (preds == labels).mean()
    return {"accuracy": float(acc), "balanced_accuracy": float(bal_acc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    print("=== PUBHEALTH Fine-Tuning (Health Claim Verification) ===")
    print(f"Base model: {BASE_MODEL}")
    print(f"Output: {OUTPUT_DIR}")
    print(
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    train_ds, val_ds, test_ds = load_healthver()
    tok_fn = tokenize_fn(tokenizer)
    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tok_fn, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tok_fn, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="balanced_accuracy",
        greater_is_better=True,
        fp16=True,
        dataloader_num_workers=2,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    start = time.time()
    if args.resume:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    elapsed = time.time() - start

    print(f"\nTraining time: {elapsed / 60:.1f} min")

    test_result = trainer.evaluate(test_ds)
    print(f"Test balanced accuracy: {test_result['eval_balanced_accuracy']:.4f}")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    result = {
        "dataset": "pubhealth",
        "base_model": BASE_MODEL,
        "test_balanced_accuracy": test_result["eval_balanced_accuracy"],
        "test_accuracy": test_result["eval_accuracy"],
        "training_time_minutes": round(elapsed / 60, 1),
        "epochs": 5,
        "status": "COMPLETE",
    }
    result_path = os.path.join(OUTPUT_DIR, "training_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nCOMPLETE — bal_acc={result['test_balanced_accuracy']:.4f}")
    print(f"Result saved to {result_path}")


if __name__ == "__main__":
    main()
