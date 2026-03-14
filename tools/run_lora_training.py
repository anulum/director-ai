# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — LoRA Fine-Tuning Pipeline
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""LoRA fine-tuning for FactCG-DeBERTa-v3-Large.

Unlike full fine-tuning (which destroyed 22/23 models via catastrophic
forgetting), LoRA updates <1% of parameters, preserving base calibration.
The CommitmentBank success (+0.54pp, 250 samples) confirms small targeted
updates work — LoRA generalises this to larger datasets.

Usage::

    python tools/run_lora_training.py \\
        --dataset halueval \\
        --output-dir models/lora-halueval \\
        --rank 8 --alpha 16 --lr 1e-4 --epochs 5

    # With sieving (Phase 3B)
    python tools/run_lora_training.py \\
        --dataset halueval \\
        --sieving --sieve-rate 0.1 \\
        --output-dir models/lora-halueval-sieved

    # Domain-specific adapter (Phase 3C)
    python tools/run_lora_training.py \\
        --dataset mednli \\
        --output-dir models/lora-medical \\
        --rank 4 --alpha 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os

logger = logging.getLogger("DirectorAI.LoRATraining")

FACTCG_TEMPLATE = (
    "{text_a}\n\nChoose your answer: based on the paragraph above "
    'can we conclude that "{text_b}"?\n\nOPTIONS:\n- Yes\n- No\n'
    "I think the answer is "
)


def load_dataset_pairs(
    dataset_name: str,
    max_samples: int | None = None,
    split: str = "train",
) -> list[dict]:
    """Load (premise, hypothesis, label) pairs from supported datasets.

    Returns list of {"premise": str, "hypothesis": str, "label": int}.
    Label: 1 = supported/entailment, 0 = not supported/contradiction.
    """
    from datasets import load_dataset

    pairs: list[dict] = []
    token = os.environ.get("HF_TOKEN")

    if dataset_name == "halueval":
        halu_split = "data" if split == "train" else split
        ds = load_dataset(
            "pminervini/HaluEval", "qa_samples", split=halu_split, token=token
        )
        for row in ds:
            pairs.append(
                {
                    "premise": row.get("knowledge", ""),
                    "hypothesis": row.get("answer", ""),
                    "label": 1 if row.get("hallucination") == "no" else 0,
                }
            )
    elif dataset_name == "vitaminc":
        ds = load_dataset("tals/vitaminc", split=split, token=token)
        for row in ds:
            lbl = row.get("label", -1)
            if lbl not in (0, 1, 2):
                continue
            pairs.append(
                {
                    "premise": row.get("evidence", ""),
                    "hypothesis": row.get("claim", ""),
                    "label": 1 if lbl == 0 else 0,
                }
            )
    elif dataset_name == "mednli":
        ds = load_dataset("bigbio/mednli", split=split, token=token)
        for row in ds:
            lbl = row.get("gold_label", "")
            if lbl not in ("entailment", "contradiction", "neutral"):
                continue
            pairs.append(
                {
                    "premise": row.get("sentence1", ""),
                    "hypothesis": row.get("sentence2", ""),
                    "label": 1 if lbl == "entailment" else 0,
                }
            )
    elif dataset_name == "fever":
        ds = load_dataset("fever/fever_v1", split=split, token=token)
        for row in ds:
            lbl = row.get("label", "")
            if lbl not in ("SUPPORTS", "REFUTES"):
                continue
            pairs.append(
                {
                    "premise": row.get("evidence_sentence", row.get("claim", "")),
                    "hypothesis": row.get("claim", ""),
                    "label": 1 if lbl == "SUPPORTS" else 0,
                }
            )
    elif dataset_name == "contractnli":
        ds = load_dataset("kiddothe2b/ContractNLI", split=split, token=token)
        for row in ds:
            lbl = row.get("label", -1)
            pairs.append(
                {
                    "premise": row.get("premise", ""),
                    "hypothesis": row.get("hypothesis", ""),
                    "label": 1 if lbl == 0 else 0,
                }
            )
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported: halueval, vitaminc, mednli, fever, contractnli",
        )

    pairs = [p for p in pairs if p["premise"] and p["hypothesis"]]
    if max_samples:
        pairs = pairs[:max_samples]
    logger.info("Loaded %d pairs from %s/%s", len(pairs), dataset_name, split)
    return pairs


def format_for_factcg(pairs: list[dict]) -> list[dict]:
    """Convert pairs to FactCG instruction template format."""
    formatted = []
    for p in pairs:
        text = FACTCG_TEMPLATE.format(text_a=p["premise"], text_b=p["hypothesis"])
        formatted.append({"text": text, "label": p["label"]})
    return formatted


class SievingCollator:
    """Data collator that corrupts tokens during training (Phase 3B).

    Randomly replaces input tokens with [MASK] or random vocabulary
    tokens. Forces the model to learn semantic features rather than
    surface patterns. Based on the sieving technique prototype in
    tools/sieving.py.
    """

    def __init__(self, tokenizer, sieve_rate: float = 0.1, seed: int = 42):
        import random

        self.tokenizer = tokenizer
        self.sieve_rate = sieve_rate
        self.rng = random.Random(seed)
        self.vocab_size = tokenizer.vocab_size
        self.mask_id = getattr(tokenizer, "mask_token_id", 103)
        special = set()
        for attr in ("cls_token_id", "sep_token_id", "pad_token_id", "mask_token_id"):
            tid = getattr(tokenizer, attr, None)
            if tid is not None:
                special.add(tid)
        self.special_ids = special

    def __call__(self, features: list[dict]) -> dict:

        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        # tokenizer.pad() passes "label" through unchanged, but
        # DeBERTa's forward() expects "labels" (plural).
        if "label" in batch:
            batch["labels"] = batch.pop("label")
        if self.sieve_rate <= 0:
            return batch

        input_ids = batch["input_ids"].clone()
        for i in range(input_ids.size(0)):
            for j in range(input_ids.size(1)):
                tid = input_ids[i, j].item()
                if tid in self.special_ids:
                    continue
                if self.rng.random() < self.sieve_rate:
                    if self.rng.random() < 0.5:
                        input_ids[i, j] = self.mask_id
                    else:
                        input_ids[i, j] = self.rng.randint(0, self.vocab_size - 1)
        batch["input_ids"] = input_ids
        return batch


def train_lora(
    pairs: list[dict],
    model_name: str = "yaxili96/FactCG-DeBERTa-v3-Large",
    output_dir: str = "models/lora-output",
    rank: int = 8,
    alpha: int = 16,
    target_modules: list[str] | None = None,
    lr: float = 1e-4,
    epochs: int = 5,
    batch_size: int = 4,
    grad_accum: int = 4,
    sieving: bool = False,
    sieve_rate: float = 0.1,
    eval_split: float = 0.1,
    seed: int = 42,
) -> dict:
    """Train LoRA adapter on the given pairs.

    Returns training metrics dict.
    """
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    if target_modules is None:
        target_modules = ["query_proj", "value_proj"]

    logger.info("Loading base model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        low_cpu_mem_usage=False,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.1,
        target_modules=target_modules,
        bias="none",
        # PEFT defaults modules_to_save=["classifier","score"] for SEQ_CLS,
        # which fully retrains the classification head and destroys FactCG's
        # calibration. Keep the head frozen — LoRA encoder changes are small
        # enough for the existing head to handle.
        modules_to_save=[],
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "LoRA: %d/%d params trainable (%.2f%%)",
        trainable,
        total,
        100 * trainable / total,
    )

    formatted = format_for_factcg(pairs)

    # Train/eval split
    import random

    rng = random.Random(seed)
    rng.shuffle(formatted)
    split_idx = max(1, int(len(formatted) * (1 - eval_split)))
    train_data = formatted[:split_idx]
    eval_data = formatted[split_idx:]

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )

    from datasets import Dataset

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

    collator = None
    if sieving:
        collator = SievingCollator(tokenizer, sieve_rate=sieve_rate, seed=seed)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    logger.info(
        "Starting LoRA training: %d train, %d eval", len(train_ds), len(eval_ds)
    )
    result = trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = {
        "train_loss": result.training_loss,
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 4),
        "lora_rank": rank,
        "lora_alpha": alpha,
        "target_modules": target_modules,
        "learning_rate": lr,
        "epochs": epochs,
        "sieving": sieving,
        "sieve_rate": sieve_rate if sieving else 0,
    }

    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Training complete. Adapter saved to %s", output_dir)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Director-AI NLI")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset: halueval, vitaminc, mednli, fever, contractnli",
    )
    parser.add_argument("--model", default="yaxili96/FactCG-DeBERTa-v3-Large")
    parser.add_argument("--output-dir", default="models/lora-output")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument(
        "--target-modules", nargs="+", default=["query_proj", "value_proj"]
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument(
        "--sieving",
        action="store_true",
        help="Enable sieving noise injection (Phase 3B)",
    )
    parser.add_argument("--sieve-rate", type=float, default=0.1)
    parser.add_argument("--eval-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = load_dataset_pairs(args.dataset, max_samples=args.max_samples)

    result = train_lora(
        pairs=pairs,
        model_name=args.model,
        output_dir=args.output_dir,
        rank=args.rank,
        alpha=args.alpha,
        target_modules=args.target_modules,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        sieving=args.sieving,
        sieve_rate=args.sieve_rate,
        eval_split=args.eval_split,
        seed=args.seed,
    )

    print("\nTraining metrics:")
    for k, v in result.items():
        print(f"  {k}: {v}")
