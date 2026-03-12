"""Forge A/B demo: validate R-Drop + FGM + Focal on ClimateFEVER.

Experiments:
  A — Forge only (R-Drop α=4.0, FGM ε=1.0, Focal γ=2.0)
  B — Full stack (Forge + Sieving noise=0.10)

Compares against sieving_ab_results.json baseline (standard=78.87%, sieving=80.90%).
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
from datasets import load_dataset
from forge import ForgeConfig, ForgeTrainer
from sieving import SievingCollator
from sklearn.metrics import balanced_accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)

MODEL = "microsoft/deberta-v3-large"
MAX_LEN = 512
SEED = 42
OUTPUT_BASE = "/home/director-ai/models"


def load_climatefever(tokenizer, max_len=MAX_LEN):
    ds = load_dataset("climate_fever", trust_remote_code=True)
    train_raw = ds["test"]  # ClimateFEVER only has test split; we split it

    label_map = {"SUPPORTS": 1, "REFUTES": 0, "NOT_ENOUGH_INFO": 0, "DISPUTED": 0}

    def preprocess(example):
        claim = example["claim"]
        evidences = example.get("evidences", [])
        if evidences and isinstance(evidences[0], dict):
            evidence_text = " ".join(e.get("evidence", "") for e in evidences[:3])
        elif evidences and isinstance(evidences[0], list):
            evidence_text = " ".join(
                e.get("evidence", "")
                for sublist in evidences[:3]
                for e in (sublist if isinstance(sublist, list) else [sublist])
            )
        else:
            evidence_text = ""

        tok = tokenizer(
            claim,
            evidence_text[:2000],
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )
        tok["labels"] = label_map.get(example["claim_label"], 0)
        return tok

    processed = train_raw.map(preprocess, remove_columns=train_raw.column_names)
    processed.set_format("torch")

    split = processed.train_test_split(test_size=0.3, seed=SEED)
    return split["train"], split["test"]


def add_noise_to_dataset(dataset, tokenizer, noise_ratio=0.15):
    """Corrupt tokens to test robustness (same as sieving demo)."""
    vocab_size = tokenizer.vocab_size
    noisy = dataset.map(
        lambda ex: {
            "input_ids": torch.where(
                torch.rand_like(ex["input_ids"].float()) < noise_ratio,
                torch.randint(0, vocab_size, ex["input_ids"].shape),
                ex["input_ids"],
            )
        },
        batched=False,
    )
    noisy.set_format("torch")
    return noisy


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"balanced_accuracy": balanced_accuracy_score(labels, preds)}


def run_experiment(name, train_ds, eval_ds, tokenizer, forge_cfg, use_sieving=False):
    print(f"\n{'=' * 60}")
    print(f"  Experiment: {name}")
    print(f"  Forge: {forge_cfg.active_techniques() or ['none']}")
    print(f"  Sieving: {use_sieving}")
    print(f"{'=' * 60}\n")

    output_dir = os.path.join(OUTPUT_BASE, f"forge-demo-{name}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL,
        num_labels=2,
    )

    collator = None
    if use_sieving:
        collator = SievingCollator(tokenizer, noise_ratio=0.10)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
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
        seed=SEED,
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = ForgeTrainer(
        model=model,
        args=args,
        forge_config=forge_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        data_collator=collator,
    )

    t0 = time.time()
    trainer.train()
    train_min = (time.time() - t0) / 60

    clean_metrics = trainer.evaluate(eval_ds)
    clean_acc = clean_metrics["eval_balanced_accuracy"]

    noisy_ds = add_noise_to_dataset(eval_ds, tokenizer, noise_ratio=0.15)
    noisy_metrics = trainer.evaluate(noisy_ds)
    noisy_acc = noisy_metrics["eval_balanced_accuracy"]

    result = {
        "name": name,
        "clean_bal_acc": round(clean_acc, 4),
        "noisy_bal_acc": round(noisy_acc, 4),
        "robustness_gap": round(clean_acc - noisy_acc, 4),
        "training_minutes": round(train_min, 1),
    }
    print(f"\n  Results for {name}:")
    print(f"    Clean accuracy:     {clean_acc:.4f}")
    print(f"    Noisy accuracy:     {noisy_acc:.4f}")
    print(f"    Robustness gap:     {clean_acc - noisy_acc:.4f}")
    print(f"    Training time:      {train_min:.1f} min")

    del model, trainer
    torch.cuda.empty_cache()
    return result


def main():
    print(f"Loading tokenizer: {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    print("Loading ClimateFEVER...")
    train_ds, eval_ds = load_climatefever(tokenizer)
    print(f"  Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # Experiment A: R-Drop + Focal (no FGM — conflicts with fp16 AMP scaler on Turing GPUs)
    forge_cfg = ForgeConfig(rdrop_alpha=4.0, focal_gamma=2.0)
    result_forge = run_experiment(
        "rdrop-focal", train_ds, eval_ds, tokenizer, forge_cfg, use_sieving=False
    )

    # Experiment B: R-Drop + Focal + Sieving
    result_full = run_experiment(
        "rdrop-focal-sieving", train_ds, eval_ds, tokenizer, forge_cfg, use_sieving=True
    )

    # Load previous sieving-only results for comparison
    prev = {}
    prev_path = "/home/director-ai/sieving_ab_results.json"
    if os.path.exists(prev_path):
        with open(prev_path) as f:
            prev = json.load(f)

    summary = {
        "model": MODEL,
        "dataset": "climate_fever",
        "epochs": 10,
        "forge_config": {"rdrop_alpha": 4.0, "focal_gamma": 2.0},
        "rdrop_focal": result_forge,
        "rdrop_focal_sieving": result_full,
        "previous_standard": prev.get("standard", {}),
        "previous_sieving": prev.get("sieving", {}),
    }

    # Deltas vs standard baseline
    std_clean = prev.get("standard", {}).get("clean_bal_acc", 0)
    if std_clean:
        summary["delta_forge_vs_standard"] = round(
            result_forge["clean_bal_acc"] - std_clean, 4
        )
        summary["delta_fullstack_vs_standard"] = round(
            result_full["clean_bal_acc"] - std_clean, 4
        )
        summary["delta_fullstack_vs_sieving"] = round(
            result_full["clean_bal_acc"]
            - prev.get("sieving", {}).get("clean_bal_acc", 0),
            4,
        )

    out_path = "/home/director-ai/forge_ab_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    if std_clean:
        print(f"  Standard baseline:     {std_clean:.4f}")
        print(
            f"  Sieving only:          {prev.get('sieving', {}).get('clean_bal_acc', 'N/A')}"
        )
    print(f"  Forge only:            {result_forge['clean_bal_acc']:.4f}")
    print(f"  Forge + Sieving:       {result_full['clean_bal_acc']:.4f}")
    if std_clean:
        print(f"  Forge vs standard:     {summary['delta_forge_vs_standard']:+.4f}")
        print(f"  Full stack vs standard:{summary['delta_fullstack_vs_standard']:+.4f}")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
