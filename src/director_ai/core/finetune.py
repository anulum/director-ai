# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Domain Adaptation / Fine-tuning Pipeline
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Fine-tune the NLI model on domain-specific labeled data.

Supports FactCG-DeBERTa-v3-Large and any HuggingFace NLI model.
Input: JSONL file with ``premise``, ``hypothesis``, ``label`` fields.
Labels: 1 = supported (entailment), 0 = not supported (contradiction).

Usage (CLI)::

    director-ai finetune train.jsonl --output ./my-model --epochs 3
    director-ai finetune train.jsonl --eval eval.jsonl --lr 2e-5

Usage (Python)::

    from director_ai.core.finetune import finetune_nli
    result = finetune_nli("train.jsonl", output_dir="./my-model")
    print(result.best_balanced_accuracy)

    # Use the fine-tuned model:
    from director_ai.core.nli import NLIScorer
    scorer = NLIScorer(model_name="./my-model")

Requires ``pip install director-ai[finetune]`` (adds transformers, accelerate).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("DirectorAI.FineTune")

_DEFAULT_BASE_MODEL = "yaxili96/FactCG-DeBERTa-v3-Large"

# FactCG instruction template — must match inference-time template
_FACTCG_TEMPLATE = (
    "{premise}\n\nChoose your answer: based on the paragraph above "
    'can we conclude that "{hypothesis}"?\n\nOPTIONS:\n- Yes\n- No\n'
    "I think the answer is "
)


@dataclass
class FinetuneConfig:
    """Fine-tuning hyperparameters."""

    base_model: str = _DEFAULT_BASE_MODEL
    output_dir: str = "./director-finetuned"
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_length: int = 512
    fp16: bool = True
    seed: int = 42
    eval_steps: int = 100
    save_strategy: str = "epoch"
    gradient_accumulation_steps: int = 1


@dataclass
class FinetuneResult:
    """Training result summary."""

    output_dir: str = ""
    epochs_completed: int = 0
    train_samples: int = 0
    eval_samples: int = 0
    best_balanced_accuracy: float = 0.0
    final_loss: float = 0.0
    eval_metrics: dict = field(default_factory=dict)


def _load_jsonl(path: str | Path) -> list[dict]:
    """Load JSONL dataset with premise/hypothesis/label fields."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            premise = row.get("premise") or row.get("doc") or row.get("context", "")
            hypothesis = row.get("hypothesis") or row.get("claim") or row.get("response", "")
            label = row.get("label")
            if not premise or not hypothesis or label is None:
                logger.warning("Skipping line %d: missing premise/hypothesis/label", line_num)
                continue
            rows.append({
                "premise": premise,
                "hypothesis": hypothesis,
                "label": int(label),
            })
    logger.info("Loaded %d samples from %s", len(rows), path)
    return rows


def _prepare_dataset(rows: list[dict], tokenizer, max_length: int, is_factcg: bool):
    """Tokenize rows into a HuggingFace Dataset."""
    from datasets import Dataset

    texts, labels = [], []
    for row in rows:
        if is_factcg:
            text = _FACTCG_TEMPLATE.format(
                premise=row["premise"], hypothesis=row["hypothesis"],
            )
            texts.append(text)
        else:
            texts.append((row["premise"], row["hypothesis"]))
        labels.append(row["label"])

    if is_factcg:
        encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_length, return_tensors="np",
        )
    else:
        premises = [t[0] for t in texts]
        hypotheses = [t[1] for t in texts]
        encodings = tokenizer(
            premises, hypotheses, truncation=True, padding="max_length",
            max_length=max_length, return_tensors="np",
        )

    return Dataset.from_dict({
        "input_ids": encodings["input_ids"].tolist(),
        "attention_mask": encodings["attention_mask"].tolist(),
        "labels": labels,
    })


def _compute_metrics(eval_pred):
    """Compute balanced accuracy + per-class metrics during training."""
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    bal_acc = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary", zero_division=0)
    return {"balanced_accuracy": bal_acc, "f1": f1}


def finetune_nli(
    train_path: str | Path,
    eval_path: str | Path | None = None,
    config: FinetuneConfig | None = None,
) -> FinetuneResult:
    """Fine-tune an NLI model on domain-specific data.

    Parameters
    ----------
    train_path : path to JSONL training data
    eval_path : optional path to JSONL evaluation data
    config : hyperparameters (defaults to FinetuneConfig())

    Returns
    -------
    FinetuneResult with output_dir and best metrics.
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    if config is None:
        config = FinetuneConfig()

    train_rows = _load_jsonl(train_path)
    if not train_rows:
        raise ValueError(f"No valid samples in {train_path}")

    eval_rows = _load_jsonl(eval_path) if eval_path else None

    logger.info("Base model: %s", config.base_model)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model, num_labels=2,
    )

    is_factcg = "factcg" in config.base_model.lower()

    train_dataset = _prepare_dataset(train_rows, tokenizer, config.max_length, is_factcg)
    eval_dataset = _prepare_dataset(eval_rows, tokenizer, config.max_length, is_factcg) if eval_rows else None

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        fp16=config.fp16,
        seed=config.seed,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=config.eval_steps if eval_dataset else None,
        save_strategy=config.save_strategy,
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="balanced_accuracy" if eval_dataset else None,
        greater_is_better=True,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=_compute_metrics if eval_dataset else None,
    )

    logger.info("Starting fine-tuning: %d train, %d eval, %d epochs",
                len(train_rows), len(eval_rows or []), config.epochs)
    train_result = trainer.train()

    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info("Model saved to %s", config.output_dir)

    result = FinetuneResult(
        output_dir=config.output_dir,
        epochs_completed=config.epochs,
        train_samples=len(train_rows),
        eval_samples=len(eval_rows or []),
        final_loss=train_result.training_loss,
    )

    if eval_dataset:
        eval_metrics = trainer.evaluate()
        result.eval_metrics = eval_metrics
        result.best_balanced_accuracy = eval_metrics.get("eval_balanced_accuracy", 0.0)
        logger.info("Best balanced accuracy: %.1f%%", result.best_balanced_accuracy * 100)

    return result
