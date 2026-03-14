#!/usr/bin/env python3
"""Fine-tune FactCG-DeBERTa-v3-Large on ReCoRD for reading comprehension.

ReCoRD (Zhang et al. 2018): ~101K cloze-style queries over news articles.
Each example has a passage, query with @placeholder, and entity answers.
Maps to NLI: passage=premise, query+answer=hypothesis.

Usage:
    python run_record_training.py [--resume]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from datasets import Dataset, load_dataset
from sklearn.metrics import balanced_accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "yaxili96/FactCG-DeBERTa-v3-Large"
OUTPUT_DIR = "/home/director-ai/models/factcg-record"
TEMPLATE = (
    "{text_a}\n\nChoose your answer: based on the paragraph above "
    'can we conclude that "{text_b}"?\n\nOPTIONS:\n- Yes\n- No\n'
    "I think the answer is "
)


def load_record():
    ds = load_dataset("super_glue", "record")
    # ReCoRD: passage, query (with @placeholder), entities, answers
    # Convert: correct entity fill = label 1, wrong entity fill = label 0

    examples = []
    for split_name in ["train"]:
        for ex in ds[split_name]:
            passage = ex["passage"]
            query = ex["query"]
            entities = ex["entities"]
            answers = (
                ex["answers"] if isinstance(ex["answers"], list) else [ex["answers"]]
            )
            answer_set = set(answers)

            for ent in entities:
                filled = query.replace("@placeholder", ent)
                label = 1 if ent in answer_set else 0
                text = TEMPLATE.format(text_a=passage[:1500], text_b=filled)
                examples.append({"text": text, "label": label})

    full = Dataset.from_list(examples)
    # Balance: subsample majority class
    pos = full.filter(lambda x: x["label"] == 1)
    neg = full.filter(lambda x: x["label"] == 0)
    if len(neg) > 2 * len(pos):
        neg = neg.shuffle(seed=42).select(range(2 * len(pos)))
    from datasets import concatenate_datasets

    balanced = concatenate_datasets([pos, neg]).shuffle(seed=42)

    split = balanced.train_test_split(test_size=0.1, seed=42)
    val_split = split["test"].train_test_split(test_size=0.5, seed=42)

    print(
        f"ReCoRD loaded: train={len(split['train'])}, val={len(val_split['train'])}, test={len(val_split['test'])}",
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

    print("=== ReCoRD Fine-Tuning ===")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    train_ds, val_ds, test_ds = load_record()
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
        logging_steps=100,
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
        "dataset": "record_superglue",
        "base_model": BASE_MODEL,
        "test_balanced_accuracy": test_result["eval_balanced_accuracy"],
        "test_accuracy": test_result["eval_accuracy"],
        "training_time_minutes": round(elapsed / 60, 1),
        "epochs": 3,
        "status": "COMPLETE",
    }
    with open(os.path.join(OUTPUT_DIR, "training_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nCOMPLETE — bal_acc={result['test_balanced_accuracy']:.4f}")


if __name__ == "__main__":
    main()
