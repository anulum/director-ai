#!/usr/bin/env python3
"""Fine-tune FactCG-DeBERTa-v3-Large on ClimateFEVER for climate claims.

ClimateFEVER (Diggelmann et al. 2020): ~1.5K climate claims with Wikipedia
evidence sentences. Each claim has multiple evidence passages with
per-evidence labels. Flattened to (evidence, claim) pairs.

Binary: SUPPORTS=1, REFUTES=0, NOT_ENOUGH_INFO/DISPUTED dropped.

Usage:
    python run_climatefever_training.py [--resume]
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
OUTPUT_DIR = "/home/director-ai/models/factcg-climatefever"
TEMPLATE = (
    "{text_a}\n\nChoose your answer: based on the paragraph above "
    'can we conclude that "{text_b}"?\n\nOPTIONS:\n- Yes\n- No\n'
    "I think the answer is "
)

EVIDENCE_LABEL_MAP = {0: 1, 2: 0}  # 0=SUPPORTS→1, 2=REFUTES→0, 1=NOT_ENOUGH_INFO→skip


def load_climatefever():
    """Load ClimateFEVER and flatten to (evidence, claim) pairs."""
    rng = random.Random(42)
    ds = load_dataset("climate_fever")

    rows = []
    for ex in ds["test"]:  # ClimateFEVER only has a "test" split
        claim = ex["claim"]
        evidences = ex.get("evidences", [])
        for ev in evidences:
            ev_label = ev.get("evidence_label", "")
            label = EVIDENCE_LABEL_MAP.get(ev_label)
            if label is None:
                continue
            ev_text = (ev.get("evidence") or "").strip()
            if not ev_text:
                continue
            text = TEMPLATE.format(text_a=ev_text, text_b=claim)
            rows.append({"text": text, "label": label})

    rng.shuffle(rows)
    print(f"ClimateFEVER flattened: {len(rows)} (evidence, claim) pairs")

    # 70/15/15 split (single-split dataset)
    n = len(rows)
    i1, i2 = int(n * 0.70), int(n * 0.85)
    train_rows, val_rows, test_rows = rows[:i1], rows[i1:i2], rows[i2:]
    print(f"  train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")
    return (
        Dataset.from_list(train_rows),
        Dataset.from_list(val_rows),
        Dataset.from_list(test_rows),
    )


def tokenize_fn(tokenizer, max_length=512):
    def _tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
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

    print("=== ClimateFEVER Fine-Tuning (Climate Domain) ===")
    print(f"Base model: {BASE_MODEL}")
    print(f"Output: {OUTPUT_DIR}")
    print(
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    train_ds, val_ds, test_ds = load_climatefever()
    tok_fn = tokenize_fn(tokenizer)
    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tok_fn, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tok_fn, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=10,
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
        logging_steps=20,
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
        "dataset": "climate_fever",
        "base_model": BASE_MODEL,
        "test_balanced_accuracy": test_result["eval_balanced_accuracy"],
        "test_accuracy": test_result["eval_accuracy"],
        "training_time_minutes": round(elapsed / 60, 1),
        "epochs": 10,
        "status": "COMPLETE",
    }
    result_path = os.path.join(OUTPUT_DIR, "training_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nCOMPLETE — bal_acc={result['test_balanced_accuracy']:.4f}")
    print(f"Result saved to {result_path}")


if __name__ == "__main__":
    main()
