#!/usr/bin/env python3
"""Fine-tune FactCG-DeBERTa-v3-Large on FRANK for summarization factual consistency.

FRANK (Pagnoni et al. 2021): ~2.2K human-annotated summary-article pairs with
fine-grained factuality labels. Directly measures what AggreFact measures.

Usage:
    python run_frank_training.py [--resume]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from datasets import load_dataset
from sklearn.metrics import balanced_accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "yaxili96/FactCG-DeBERTa-v3-Large"
OUTPUT_DIR = "/home/director-ai/models/factcg-frank"
TEMPLATE = (
    "{text_a}\n\nChoose your answer: based on the paragraph above "
    'can we conclude that "{text_b}"?\n\nOPTIONS:\n- Yes\n- No\n'
    "I think the answer is "
)


def load_frank():
    # Try multiple HF dataset names for FRANK
    for name in ["frankbenchmark/frank", "nbroad/frank", "frank"]:
        try:
            ds = load_dataset(name, trust_remote_code=True)
            print(f"Loaded FRANK from {name}")
            break
        except Exception as e:
            print(f"Failed {name}: {e}")
            continue
    else:
        # Fallback: use SummaC benchmark which includes FRANK-like data
        print("FRANK not found, falling back to SummaC")
        ds = load_dataset("mteb/summac")

    splits = list(ds.keys())
    raw = ds[splits[0]]
    cols = raw.column_names
    print(f"Columns: {cols}")

    def convert(example):
        # Detect column names
        doc = ""
        claim = ""
        label_raw = 0

        for k in ["article", "document", "text", "source", "premise"]:
            if example.get(k):
                doc = str(example[k])[:2000]
                break
        for k in ["summary", "claim", "hypothesis", "sentence"]:
            if example.get(k):
                claim = str(example[k])
                break
        for k in ["label", "Factual", "factual", "is_factual"]:
            if k in example:
                label_raw = example[k]
                break

        if isinstance(label_raw, str):
            label = (
                1
                if label_raw.lower() in ("factual", "1", "true", "correct", "support")
                else 0
            )
        elif isinstance(label_raw, float):
            label = 1 if label_raw > 0.5 else 0
        else:
            label = int(label_raw)

        text = TEMPLATE.format(text_a=doc, text_b=claim)
        return {"text": text, "label": label}

    if len(splits) >= 3:
        train = ds["train"].map(convert, remove_columns=ds["train"].column_names)
        val_key = "validation" if "validation" in splits else splits[1]
        test_key = "test" if "test" in splits else splits[2]
        val = ds[val_key].map(convert, remove_columns=ds[val_key].column_names)
        test = ds[test_key].map(convert, remove_columns=ds[test_key].column_names)
    else:
        mapped = raw.map(convert, remove_columns=cols)
        split = mapped.train_test_split(test_size=0.2, seed=42)
        val_split = split["test"].train_test_split(test_size=0.5, seed=42)
        train = split["train"]
        val = val_split["train"]
        test = val_split["test"]

    print(f"FRANK loaded: train={len(train)}, val={len(val)}, test={len(test)}")
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
    return {
        "accuracy": float((preds == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    print("=== FRANK Fine-Tuning ===")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    train_ds, val_ds, test_ds = load_frank()
    tok_fn = tokenize_fn(tokenizer)
    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tok_fn, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tok_fn, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
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
        logging_steps=10,
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
    trainer.train(resume_from_checkpoint=True) if args.resume else trainer.train()
    elapsed = time.time() - start

    test_result = trainer.evaluate(test_ds)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    result = {
        "dataset": "frank",
        "base_model": BASE_MODEL,
        "test_balanced_accuracy": test_result["eval_balanced_accuracy"],
        "test_accuracy": test_result["eval_accuracy"],
        "training_time_minutes": round(elapsed / 60, 1),
        "epochs": 10,
        "status": "COMPLETE",
    }
    with open(os.path.join(OUTPUT_DIR, "training_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nCOMPLETE — bal_acc={result['test_balanced_accuracy']:.4f}")


if __name__ == "__main__":
    main()
