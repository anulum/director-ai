# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — run_finance_training
#!/usr/bin/env python3
"""Fine-tune FactCG-DeBERTa-v3-Large on financial NLI data.

Converts Financial PhraseBank (sentiment-labeled financial sentences) into
binary NLI pairs, then fine-tunes for financial domain guardrails.

Strategy:
- Entailment pair: (sentence, "This financial statement is {correct_sentiment}") → 1
- Contradiction pair: (sentence, "This financial statement is {wrong_sentiment}") → 0

Usage:
    python run_finance_training.py [--resume]
"""

from __future__ import annotations

import argparse
import json
import os
import random
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
OUTPUT_DIR = "/home/director-ai/models/factcg-finance"
TEMPLATE = (
    "{text_a}\n\nChoose your answer: based on the paragraph above "
    'can we conclude that "{text_b}"?\n\nOPTIONS:\n- Yes\n- No\n'
    "I think the answer is "
)

SENTIMENT_MAP = {
    "positive": [
        "indicates positive financial performance",
        "suggests favorable market conditions",
        "reflects financial growth or improvement",
    ],
    "negative": [
        "indicates negative financial performance",
        "suggests unfavorable market conditions",
        "reflects financial decline or deterioration",
    ],
    "neutral": [
        "presents factual financial information without strong sentiment",
        "describes a routine financial event",
        "states neutral financial data",
    ],
}

OPPOSITE = {
    "positive": "negative",
    "negative": "positive",
    "neutral": "positive",  # neutral contradicts a strong positive claim
}


def load_finance_nli():
    """Load Financial PhraseBank and convert to binary NLI pairs."""
    ds = load_dataset("takala/financial_phrasebank", "sentences_allagree")
    data = ds["train"]

    label_names = {0: "negative", 1: "neutral", 2: "positive"}
    pairs = []
    rng = random.Random(42)

    for example in data:
        sentence = example["sentence"]
        sentiment = label_names[example["label"]]

        # Entailment: sentence + correct sentiment hypothesis
        hyp_correct = rng.choice(SENTIMENT_MAP[sentiment])
        text_ent = TEMPLATE.format(text_a=sentence, text_b=hyp_correct)
        pairs.append({"text": text_ent, "label": 1})

        # Contradiction: sentence + wrong sentiment hypothesis
        wrong_sentiment = OPPOSITE[sentiment]
        hyp_wrong = rng.choice(SENTIMENT_MAP[wrong_sentiment])
        text_con = TEMPLATE.format(text_a=sentence, text_b=hyp_wrong)
        pairs.append({"text": text_con, "label": 0})

    rng.shuffle(pairs)
    split = int(len(pairs) * 0.85)
    val_split = int(len(pairs) * 0.925)
    train_data = pairs[:split]
    val_data = pairs[split:val_split]
    test_data = pairs[val_split:]

    from datasets import Dataset

    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)
    test_ds = Dataset.from_list(test_data)
    print(
        f"Finance NLI: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)} "
        f"(from {len(data)} Financial PhraseBank sentences)",
    )
    return train_ds, val_ds, test_ds


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

    print("=== Finance NLI Fine-Tuning ===")
    print(f"Base model: {BASE_MODEL}")
    print(f"Output: {OUTPUT_DIR}")
    print(
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    train_ds, val_ds, test_ds = load_finance_nli()
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
        "dataset": "financial_phrasebank_nli",
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
