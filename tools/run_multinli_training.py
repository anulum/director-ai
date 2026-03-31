# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — run_multinli_training
#!/usr/bin/env python3
"""Fine-tune FactCG-DeBERTa-v3-Large on MultiNLI for multi-genre coverage.

MultiNLI (Williams et al. 2018): ~433K premise-hypothesis pairs across
10 genres (fiction, government, travel, telephone, letters, 9/11, face-to-face,
slate, verbatim, oup). Broadest genre coverage of any NLI dataset.

Binary: entailment=1, contradiction=0, neutral dropped.
No public test labels — validation_matched used as test.

Usage:
    python run_multinli_training.py [--resume]
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
OUTPUT_DIR = "/home/director-ai/models/factcg-multinli"
TEMPLATE = (
    "{text_a}\n\nChoose your answer: based on the paragraph above "
    'can we conclude that "{text_b}"?\n\nOPTIONS:\n- Yes\n- No\n'
    "I think the answer is "
)

LABEL_MAP = {0: 1, 1: None, 2: 0}  # entailment=1, neutral=skip, contradiction=0


def load_multinli():
    """Load MultiNLI and convert to binary NLI."""
    rng = random.Random(42)
    ds = load_dataset("nyu-mll/multi_nli")

    train_rows = []
    for ex in ds["train"]:
        label = LABEL_MAP.get(ex["label"])
        if label is None:
            continue
        text = TEMPLATE.format(text_a=ex["premise"], text_b=ex["hypothesis"])
        train_rows.append({"text": text, "label": label})

    rng.shuffle(train_rows)

    # validation_matched as test (no public test labels)
    test_rows = []
    for ex in ds["validation_matched"]:
        label = LABEL_MAP.get(ex["label"])
        if label is None:
            continue
        text = TEMPLATE.format(text_a=ex["premise"], text_b=ex["hypothesis"])
        test_rows.append({"text": text, "label": label})

    # validation_mismatched as val (cross-genre)
    val_rows = []
    for ex in ds["validation_mismatched"]:
        label = LABEL_MAP.get(ex["label"])
        if label is None:
            continue
        text = TEMPLATE.format(text_a=ex["premise"], text_b=ex["hypothesis"])
        val_rows.append({"text": text, "label": label})

    print(
        f"MultiNLI loaded: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}",
    )
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

    print("=== MultiNLI Fine-Tuning (Multi-Genre Coverage) ===")
    print(f"Base model: {BASE_MODEL}")
    print(f"Output: {OUTPUT_DIR}")
    print(
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    train_ds, val_ds, test_ds = load_multinli()
    tok_fn = tokenize_fn(tokenizer)
    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tok_fn, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tok_fn, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        learning_rate=1.5e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="balanced_accuracy",
        greater_is_better=True,
        fp16=True,
        dataloader_num_workers=4,
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
        "dataset": "multi_nli",
        "base_model": BASE_MODEL,
        "test_balanced_accuracy": test_result["eval_balanced_accuracy"],
        "test_accuracy": test_result["eval_accuracy"],
        "training_time_minutes": round(elapsed / 60, 1),
        "epochs": 3,
        "status": "COMPLETE",
    }
    result_path = os.path.join(OUTPUT_DIR, "training_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nCOMPLETE — bal_acc={result['test_balanced_accuracy']:.4f}")
    print(f"Result saved to {result_path}")


if __name__ == "__main__":
    main()
