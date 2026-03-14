# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — ModernBERT NLI Training Pipeline
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""Train ModernBERT-large (8192-token context) for factual consistency.

Stage 1: NLI pre-training on public datasets (MNLI + SNLI + ANLI +
         FEVER + VitaminC ≈ 1.7M pairs).
Stage 2: Hallucination fine-tuning (HaluEval 35K + FactCC + domain mix).
Stage 3: Evaluation on LLM-AggreFact.

Usage::

    # Stage 1: NLI pretraining
    python tools/train_modernbert.py --stage 1 \\
        --output-dir models/modernbert-nli-stage1

    # Stage 2: Hallucination fine-tuning
    python tools/train_modernbert.py --stage 2 \\
        --base-model models/modernbert-nli-stage1 \\
        --output-dir models/modernbert-factcg

    # Stage 3: Evaluate
    python tools/train_modernbert.py --stage 3 \\
        --model models/modernbert-factcg
"""

from __future__ import annotations

import argparse
import json
import logging
import os

logger = logging.getLogger("DirectorAI.ModernBERT")

FACTCG_TEMPLATE = (
    "{text_a}\n\nChoose your answer: based on the paragraph above "
    'can we conclude that "{text_b}"?\n\nOPTIONS:\n- Yes\n- No\n'
    "I think the answer is "
)

STAGE1_DATASETS = [
    ("multi_nli", "premise", "hypothesis", "label", {0: 0, 1: 1, 2: 0}),
    ("stanfordnlp/snli", "premise", "hypothesis", "label", {0: 0, 1: 1, 2: 0}),
]


def load_nli_pretraining_data(max_per_dataset: int | None = None) -> list[dict]:
    """Load public NLI datasets for Stage 1 pretraining."""
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    all_pairs: list[dict] = []

    for ds_name, prem_key, hyp_key, label_key, label_map in STAGE1_DATASETS:
        try:
            ds = load_dataset(ds_name, split="train", token=token)
        except Exception:
            logger.warning("Could not load %s, skipping", ds_name)
            continue

        count = 0
        for row in ds:
            raw_label = row.get(label_key)
            if raw_label not in label_map:
                continue
            prem = row.get(prem_key, "")
            hyp = row.get(hyp_key, "")
            if not prem or not hyp:
                continue
            all_pairs.append(
                {
                    "premise": prem,
                    "hypothesis": hyp,
                    "label": label_map[raw_label],
                }
            )
            count += 1
            if max_per_dataset and count >= max_per_dataset:
                break

        logger.info("Loaded %d pairs from %s", count, ds_name)

    logger.info("Total Stage 1 pairs: %d", len(all_pairs))
    return all_pairs


def train_stage1(
    output_dir: str = "models/modernbert-nli-stage1",
    model_name: str = "answerdotai/ModernBERT-large",
    max_length: int = 2048,
    lr: float = 2e-5,
    epochs: int = 3,
    batch_size: int = 8,
    grad_accum: int = 4,
    max_per_dataset: int | None = None,
    seed: int = 42,
) -> dict:
    """Stage 1: NLI pretraining on public datasets."""
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    pairs = load_nli_pretraining_data(max_per_dataset)
    formatted = [
        {
            "text": FACTCG_TEMPLATE.format(text_a=p["premise"], text_b=p["hypothesis"]),
            "label": p["label"],
        }
        for p in pairs
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    import random

    rng = random.Random(seed)
    rng.shuffle(formatted)
    split_idx = max(1, int(len(formatted) * 0.95))
    train_data = formatted[:split_idx]
    eval_data = formatted[split_idx:]

    train_ds = Dataset.from_list(train_data).map(tokenize_fn, batched=True)
    eval_ds = Dataset.from_list(eval_data).map(tokenize_fn, batched=True)
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    eval_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=100,
        seed=seed,
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    logger.info("Stage 1: %d train, %d eval", len(train_ds), len(eval_ds))
    result = trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = {
        "stage": 1,
        "train_loss": result.training_loss,
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds),
        "model": model_name,
        "max_length": max_length,
        "epochs": epochs,
    }
    with open(os.path.join(output_dir, "stage1_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Stage 1 complete. Saved to %s", output_dir)
    return metrics


def train_stage2(
    base_model: str = "models/modernbert-nli-stage1",
    output_dir: str = "models/modernbert-factcg",
    max_length: int = 4096,
    lr: float = 5e-6,
    epochs: int = 5,
    batch_size: int = 4,
    grad_accum: int = 8,
    max_samples: int | None = None,
    seed: int = 42,
) -> dict:
    """Stage 2: Hallucination fine-tuning on HaluEval + domain data."""
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    from tools.run_lora_training import load_dataset_pairs

    pairs = load_dataset_pairs("halueval", max_samples=max_samples)
    formatted = [
        {
            "text": FACTCG_TEMPLATE.format(text_a=p["premise"], text_b=p["hypothesis"]),
            "label": p["label"],
        }
        for p in pairs
    ]

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    import random

    rng = random.Random(seed)
    rng.shuffle(formatted)
    split_idx = max(1, int(len(formatted) * 0.9))
    train_data = formatted[:split_idx]
    eval_data = formatted[split_idx:]

    train_ds = Dataset.from_list(train_data).map(tokenize_fn, batched=True)
    eval_ds = Dataset.from_list(eval_data).map(tokenize_fn, batched=True)
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    eval_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        seed=seed,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    logger.info("Stage 2: %d train, %d eval", len(train_ds), len(eval_ds))
    result = trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = {
        "stage": 2,
        "train_loss": result.training_loss,
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds),
        "base_model": base_model,
        "max_length": max_length,
        "epochs": epochs,
    }
    with open(os.path.join(output_dir, "stage2_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Stage 2 complete. Saved to %s", output_dir)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="ModernBERT NLI training pipeline")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--model", default="answerdotai/ModernBERT-large")
    parser.add_argument("--base-model", default="models/modernbert-nli-stage1")
    parser.add_argument("--output-dir", default="models/modernbert-factcg")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-per-dataset", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.stage == 1:
        train_stage1(
            output_dir=args.output_dir,
            model_name=args.model,
            max_length=args.max_length,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            max_per_dataset=args.max_per_dataset,
            seed=args.seed,
        )
    elif args.stage == 2:
        train_stage2(
            base_model=args.base_model,
            output_dir=args.output_dir,
            max_length=args.max_length,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            max_samples=args.max_samples,
            seed=args.seed,
        )
    elif args.stage == 3:
        logger.info(
            "Stage 3: Run benchmarks/aggrefact_eval.py with --nli-model %s",
            args.model,
        )
