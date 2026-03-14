#!/usr/bin/env python3
"""A/B test: standard vs Sieving fine-tuning on ClimateFEVER.

ClimateFEVER is tiny (~1.5K pairs after flattening) — trains in minutes.
Runs both standard and Sieving fine-tuning, then compares balanced
accuracy on test set and on a synthetic noise-injected test set.

Usage:
    python run_sieving_demo.py [--noise-ratio 0.10]
"""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch
from datasets import Dataset, load_dataset
from sieving import SievingCollator, SievingTrainer
from sklearn.metrics import balanced_accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)

BASE_MODEL = "yaxili96/FactCG-DeBERTa-v3-Large"
TEMPLATE = (
    "{text_a}\n\nChoose your answer: based on the paragraph above "
    'can we conclude that "{text_b}"?\n\nOPTIONS:\n- Yes\n- No\n'
    "I think the answer is "
)

EVIDENCE_LABEL_MAP = {0: 1, 2: 0}  # 0=SUPPORTS→1, 2=REFUTES→0, 1=NOT_ENOUGH_INFO→skip


def load_climatefever():
    rng = random.Random(42)
    ds = load_dataset("climate_fever")
    rows = []
    for ex in ds["test"]:
        claim = ex["claim"]
        for ev in ex.get("evidences", []):
            label = EVIDENCE_LABEL_MAP.get(ev.get("evidence_label", ""))
            if label is None:
                continue
            ev_text = (ev.get("evidence") or "").strip()
            if not ev_text:
                continue
            rows.append(
                {
                    "text": TEMPLATE.format(text_a=ev_text, text_b=claim),
                    "label": label,
                },
            )
    rng.shuffle(rows)
    n = len(rows)
    i1, i2 = int(n * 0.70), int(n * 0.85)
    return (
        Dataset.from_list(rows[:i1]),
        Dataset.from_list(rows[i1:i2]),
        Dataset.from_list(rows[i2:]),
    )


def tokenize(ds, tokenizer):
    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=512, padding=False)

    return ds.map(_tok, batched=True, remove_columns=["text"])


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": float((preds == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
    }


def inject_typos(text: str, ratio: float = 0.05) -> str:
    """Simulate noisy user input by randomly swapping adjacent chars."""
    chars = list(text)
    rng = random.Random(hash(text))
    for i in range(len(chars) - 1):
        if rng.random() < ratio and chars[i].isalpha():
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def make_noisy_test(test_rows, tokenizer, typo_ratio=0.05):
    """Create a noisy copy of the test set with simulated typos."""
    noisy = []
    for ex in test_rows:
        noisy.append(
            {"text": inject_typos(ex["text"], typo_ratio), "label": ex["label"]},
        )
    return tokenize(Dataset.from_list(noisy), tokenizer)


def run_experiment(
    name,
    trainer_cls,
    collator,
    model,
    tokenizer,
    train_ds,
    val_ds,
    test_ds,
    noisy_test_ds,
    output_dir,
    epochs,
):
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="balanced_accuracy",
        greater_is_better=True,
        fp16=True,
        logging_steps=20,
        report_to="none",
    )

    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    clean_result = trainer.evaluate(test_ds)
    noisy_result = trainer.evaluate(noisy_test_ds)

    return {
        "name": name,
        "clean_bal_acc": clean_result["eval_balanced_accuracy"],
        "noisy_bal_acc": noisy_result["eval_balanced_accuracy"],
        "robustness_gap": clean_result["eval_balanced_accuracy"]
        - noisy_result["eval_balanced_accuracy"],
        "training_minutes": round(elapsed / 60, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-ratio", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    print(f"=== Sieving A/B Demo (noise_ratio={args.noise_ratio}) ===")
    print(
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Loading ClimateFEVER...")
    train_ds_raw, val_ds_raw, test_ds_raw = load_climatefever()
    print(
        f"  train={len(train_ds_raw)}, val={len(val_ds_raw)}, test={len(test_ds_raw)}",
    )

    train_ds = tokenize(train_ds_raw, tokenizer)
    val_ds = tokenize(val_ds_raw, tokenizer)
    test_ds = tokenize(test_ds_raw, tokenizer)
    noisy_test_ds = make_noisy_test(test_ds_raw, tokenizer, typo_ratio=0.05)

    # Experiment A: Standard fine-tuning
    print("\n--- Experiment A: Standard (no sieving) ---")
    model_a = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
    )
    standard_collator = SievingCollator(tokenizer, noise_ratio=0.0)
    result_a = run_experiment(
        "standard",
        SievingTrainer,
        standard_collator,
        model_a,
        tokenizer,
        train_ds,
        val_ds,
        test_ds,
        noisy_test_ds,
        "/home/director-ai/models/sieving-demo-standard",
        args.epochs,
    )

    # Experiment B: Sieving fine-tuning
    print(f"\n--- Experiment B: Sieving (noise_ratio={args.noise_ratio}) ---")
    model_b = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
    )
    sieving_collator = SievingCollator(tokenizer, noise_ratio=args.noise_ratio)
    result_b = run_experiment(
        f"sieving_{args.noise_ratio}",
        SievingTrainer,
        sieving_collator,
        model_b,
        tokenizer,
        train_ds,
        val_ds,
        test_ds,
        noisy_test_ds,
        "/home/director-ai/models/sieving-demo-sieving",
        args.epochs,
    )

    # Summary
    print("\n" + "=" * 60)
    print("SIEVING A/B COMPARISON")
    print("=" * 60)
    for r in [result_a, result_b]:
        print(f"\n{r['name']}:")
        print(f"  Clean test bal_acc:   {r['clean_bal_acc']:.4f}")
        print(f"  Noisy test bal_acc:   {r['noisy_bal_acc']:.4f}")
        print(f"  Robustness gap:       {r['robustness_gap']:.4f}")
        print(f"  Training time:        {r['training_minutes']} min")

    delta_clean = result_b["clean_bal_acc"] - result_a["clean_bal_acc"]
    delta_noisy = result_b["noisy_bal_acc"] - result_a["noisy_bal_acc"]
    delta_gap = result_a["robustness_gap"] - result_b["robustness_gap"]
    print("\nSieving effect:")
    print(f"  Clean accuracy delta:      {delta_clean:+.4f}")
    print(f"  Noisy accuracy delta:      {delta_noisy:+.4f}")
    print(f"  Robustness gap reduction:  {delta_gap:+.4f}")

    results = {
        "noise_ratio": args.noise_ratio,
        "epochs": args.epochs,
        "dataset": "climate_fever",
        "standard": result_a,
        "sieving": result_b,
        "delta_clean": delta_clean,
        "delta_noisy": delta_noisy,
        "delta_robustness_gap": delta_gap,
    }
    out_path = "/home/director-ai/sieving_ab_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
