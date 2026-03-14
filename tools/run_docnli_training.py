#!/usr/bin/env python3
"""Fine-tune FactCG-DeBERTa-v3-Large on DocNLI for document-level entailment.

DocNLI (Yin et al. 2021): ~900K document-hypothesis pairs from summarization,
QA, and other NLI sources. The most directly relevant dataset for AggreFact.

Usage:
    python run_docnli_training.py [--resume]
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
OUTPUT_DIR = "/home/director-ai/models/factcg-docnli"
TEMPLATE = (
    "{text_a}\n\nChoose your answer: based on the paragraph above "
    'can we conclude that "{text_b}"?\n\nOPTIONS:\n- Yes\n- No\n'
    "I think the answer is "
)


def load_docnli():
    """Load DocNLI and convert to binary NLI format."""
    ds = load_dataset("saattrupdan/doc-nli")
    # 100K subset: full 900K requires ~31h at batch=8; 100K ≈ 5h on RTX 6000 24GB
    subset_size = 100_000
    if len(ds["train"]) > subset_size:
        ds = ds.copy()
        ds["train"] = ds["train"].shuffle(seed=42).select(range(subset_size))

    def convert(example):
        premise = example.get("premise", "") or ""
        hypothesis = example.get("hypothesis", "") or ""
        # doc-nli: "entailment" / "not_entailment"
        label_str = example.get("label", "")
        if isinstance(label_str, int):
            label = 1 if label_str == 0 else 0
        else:
            label = 1 if label_str == "entailment" else 0
        # Truncate long premises to first 2000 chars for template
        premise_trunc = premise[:2000]
        text = TEMPLATE.format(text_a=premise_trunc, text_b=hypothesis)
        return {"text": text, "label": label}

    train = ds["train"].map(convert, remove_columns=ds["train"].column_names)

    if "validation" in ds:
        val = ds["validation"].map(
            convert,
            remove_columns=ds["validation"].column_names,
        )
    else:
        split = train.train_test_split(test_size=0.05, seed=42)
        train = split["train"]
        val = split["test"]

    if "test" in ds:
        test = ds["test"].map(convert, remove_columns=ds["test"].column_names)
    else:
        val_split = val.train_test_split(test_size=0.5, seed=42)
        val = val_split["train"]
        test = val_split["test"]

    print(f"DocNLI loaded: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


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

    print("=== DocNLI Fine-Tuning ===")
    print(f"Base model: {BASE_MODEL}")
    print(f"Output: {OUTPUT_DIR}")
    print(
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    train_ds, val_ds, test_ds = load_docnli()
    tok_fn = tokenize_fn(tokenizer)
    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tok_fn, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tok_fn, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=4,  # effective batch 32; batch=16 OOMs on 24GB
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
        logging_steps=200,
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
        "dataset": "doc-nli",
        "base_model": BASE_MODEL,
        "test_balanced_accuracy": test_result["eval_balanced_accuracy"],
        "test_accuracy": test_result["eval_accuracy"],
        "training_time_minutes": round(elapsed / 60, 1),
        "epochs": 3,
        "train_subset": 100_000,
        "status": "COMPLETE",
    }
    result_path = os.path.join(OUTPUT_DIR, "training_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nCOMPLETE — bal_acc={result['test_balanced_accuracy']:.4f}")


if __name__ == "__main__":
    main()
