# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — run_scifact_training
#!/usr/bin/env python3
"""Fine-tune FactCG-DeBERTa-v3-Large on SciFact for scientific claim verification.

SciFact (Wadden et al. 2020): ~1.4K scientific claims with evidence sentences.
Directly targets fact verification — closest proxy to AggreFact task.

Usage:
    python run_scifact_training.py [--resume]
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
OUTPUT_DIR = "/home/director-ai/models/factcg-scifact"
TEMPLATE = (
    "{text_a}\n\nChoose your answer: based on the paragraph above "
    'can we conclude that "{text_b}"?\n\nOPTIONS:\n- Yes\n- No\n'
    "I think the answer is "
)


def load_scifact():
    _ds = load_dataset("allenai/scifact", "corpus")
    claims = load_dataset("allenai/scifact", "claims")

    # Claims have: claim, evidence_doc_id, evidence_label, cited_doc_ids
    # Labels: SUPPORT, CONTRADICT, NOT_ENOUGH_INFO
    train_examples = []
    for ex in claims["train"]:
        label = 1 if ex.get("evidence_label") == "SUPPORT" else 0
        claim_text = ex["claim"]
        # Use claim as both premise context and hypothesis for binary classification
        text = TEMPLATE.format(text_a=claim_text, text_b=claim_text)
        train_examples.append({"text": text, "label": label})

    # Since SciFact is small, also try loading via the simpler approach
    if len(train_examples) < 100:
        # Fallback: load as flat dataset
        try:
            ds2 = load_dataset("allenai/scifact_entailment")

            def convert(example):
                label = 1 if example["label"] == 0 else 0  # entailment=supported
                text = TEMPLATE.format(
                    text_a=example["premise"],
                    text_b=example["hypothesis"],
                )
                return {"text": text, "label": label}

            mapped = ds2["train"].map(convert, remove_columns=ds2["train"].column_names)
            from datasets import Dataset

            split = mapped.train_test_split(test_size=0.2, seed=42)
            val_split = split["test"].train_test_split(test_size=0.5, seed=42)
            print(
                f"SciFact (entailment) loaded: train={len(split['train'])}, val={len(val_split['train'])}, test={len(val_split['test'])}",
            )
            return split["train"], val_split["train"], val_split["test"]
        except Exception:
            pass

    from datasets import Dataset

    full = Dataset.from_list(train_examples)
    split = full.train_test_split(test_size=0.2, seed=42)
    val_split = split["test"].train_test_split(test_size=0.5, seed=42)
    print(
        f"SciFact loaded: train={len(split['train'])}, val={len(val_split['train'])}, test={len(val_split['test'])}",
    )
    return split["train"], val_split["train"], val_split["test"]


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

    print("=== SciFact Fine-Tuning ===")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    train_ds, val_ds, test_ds = load_scifact()
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
    trainer.train(resume_from_checkpoint=True) if args.resume else trainer.train()
    elapsed = time.time() - start

    test_result = trainer.evaluate(test_ds)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    result = {
        "dataset": "scifact",
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
