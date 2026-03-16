# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ModernBERT NLI Training Pipeline

"""Train ModernBERT-large (8192-token context) for factual consistency.

Stage 1: NLI pre-training on public datasets (MNLI + SNLI + ANLI +
         VitaminC ≈ 1.5M pairs).
Stage 2: Hallucination fine-tuning (HaluEval 35K, 5 epochs).
Stage 3: Evaluation on LLM-AggreFact at multiple context lengths.

Usage::

    # Stage 1: NLI pretraining (skip if using pre-trained checkpoint)
    python tools/train_modernbert.py --stage 1 \\
        --output-dir models/modernbert-nli-stage1

    # Stage 2: Hallucination fine-tuning
    python tools/train_modernbert.py --stage 2 \\
        --base-model models/modernbert-nli-stage1 \\
        --output-dir models/modernbert-factcg

    # Stage 2 with pre-trained NLI checkpoint (skips Stage 1)
    python tools/train_modernbert.py --stage 2 \\
        --base-model MoritzLaurer/ModernBERT-large-zeroshot-v2.0 \\
        --skip-stage1 --output-dir models/modernbert-factcg

    # Stage 3: Evaluate at multiple context lengths
    python tools/train_modernbert.py --stage 3 \\
        --model models/modernbert-factcg
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys

logger = logging.getLogger("DirectorAI.ModernBERT")

FACTCG_TEMPLATE = (
    "{text_a}\n\nChoose your answer: based on the paragraph above "
    'can we conclude that "{text_b}"?\n\nOPTIONS:\n- Yes\n- No\n'
    "I think the answer is "
)

# (dataset_name, split, premise_col, hypothesis_col, label_col, label_map)
# Label convention: 1 = supported/entailment, 0 = not-supported
# Standard NLI numeric: 0=entailment, 1=neutral, 2=contradiction
STAGE1_DATASETS = [
    ("multi_nli", "train", "premise", "hypothesis", "label", {0: 1, 1: 0, 2: 0}),
    ("stanfordnlp/snli", "train", "premise", "hypothesis", "label", {0: 1, 1: 0, 2: 0}),
    ("anli", "train_r1", "premise", "hypothesis", "label", {0: 1, 1: 0, 2: 0}),
    ("anli", "train_r2", "premise", "hypothesis", "label", {0: 1, 1: 0, 2: 0}),
    ("anli", "train_r3", "premise", "hypothesis", "label", {0: 1, 1: 0, 2: 0}),
    ("tals/vitaminc", "train", "evidence", "claim", "label", {0: 1, 1: 0, 2: 0}),
    (
        "pietrolesci/nli_fever",
        "train",
        "premise",
        "hypothesis",
        "label",
        {0: 1, 1: 0, 2: 0},
    ),
]


def load_nli_pretraining_data(max_per_dataset: int | None = None) -> list[dict]:
    """Load public NLI datasets for Stage 1 pretraining."""
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    all_pairs: list[dict] = []

    for ds_name, split, prem_key, hyp_key, label_key, label_map in STAGE1_DATASETS:
        try:
            ds = load_dataset(ds_name, split=split, token=token)
        except Exception:
            logger.warning("Could not load %s (split=%s), skipping", ds_name, split)
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

        logger.info("Loaded %d pairs from %s (split=%s)", count, ds_name, split)

    logger.info("Total Stage 1 pairs: %d", len(all_pairs))
    return all_pairs


def _compute_metrics(eval_pred):
    """Balanced accuracy + accuracy for Trainer eval."""
    import numpy as np
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "accuracy": accuracy_score(labels, preds),
    }


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
    gradient_checkpointing: bool = False,
    resume_from: str | None = None,
) -> dict:
    """Stage 1: NLI pretraining on public datasets."""
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
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
        attn_implementation="sdpa",
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
        )

    import random

    rng = random.Random(seed)
    rng.shuffle(formatted)
    split_idx = max(1, int(len(formatted) * 0.95))
    train_data = formatted[:split_idx]
    eval_data = formatted[split_idx:]

    train_ds = Dataset.from_list(train_data).map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
    )
    eval_ds = Dataset.from_list(eval_data).map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
    )

    collator = DataCollatorWithPadding(tokenizer)

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
        metric_for_best_model="balanced_accuracy",
        greater_is_better=True,
        logging_steps=100,
        seed=seed,
        bf16=True,
        fp16=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=_compute_metrics,
    )

    logger.info("Stage 1: %d train, %d eval", len(train_ds), len(eval_ds))
    result = trainer.train(resume_from_checkpoint=resume_from)

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
    skip_stage1: bool = False,
    gradient_checkpointing: bool = False,
    resume_from: str | None = None,
) -> dict:
    """Stage 2: Hallucination fine-tuning on HaluEval."""
    from datasets import Dataset
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
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

    if skip_stage1:
        cfg = AutoConfig.from_pretrained(base_model)
        if getattr(cfg, "num_labels", 2) != 2:
            logger.info(
                "Checkpoint has %d labels, replacing head with 2-class",
                cfg.num_labels,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                base_model,
                num_labels=2,
                ignore_mismatched_sizes=True,
                attn_implementation="sdpa",
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                base_model,
                attn_implementation="sdpa",
            )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            attn_implementation="sdpa",
        )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
        )

    import random

    rng = random.Random(seed)
    rng.shuffle(formatted)
    split_idx = max(1, int(len(formatted) * 0.9))
    train_data = formatted[:split_idx]
    eval_data = formatted[split_idx:]

    train_ds = Dataset.from_list(train_data).map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
    )
    eval_ds = Dataset.from_list(eval_data).map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
    )

    collator = DataCollatorWithPadding(tokenizer)

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
        metric_for_best_model="balanced_accuracy",
        greater_is_better=True,
        logging_steps=50,
        seed=seed,
        bf16=True,
        fp16=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=_compute_metrics,
    )

    logger.info("Stage 2: %d train, %d eval", len(train_ds), len(eval_ds))
    result = trainer.train(resume_from_checkpoint=resume_from)

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


def run_stage3(model_path: str) -> None:
    """Stage 3: Evaluate on AggreFact at 512, 2048, and 4096 tokens."""
    eval_script = os.path.join(os.path.dirname(__file__), "eval_aggrefact.py")
    for max_len in (512, 2048, 4096):
        tag = f"modernbert_{max_len}"
        logger.info("Evaluating at max_length=%d", max_len)
        subprocess.run(
            [
                sys.executable,
                eval_script,
                tag,
                model_path,
                "--max-length",
                str(max_len),
                "--batch-size",
                "16",
            ],
            check=True,
        )


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
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
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
            gradient_checkpointing=args.gradient_checkpointing,
            resume_from=args.resume_from_checkpoint,
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
            skip_stage1=args.skip_stage1,
            gradient_checkpointing=args.gradient_checkpointing,
            resume_from=args.resume_from_checkpoint,
        )
    elif args.stage == 3:
        run_stage3(args.model)
