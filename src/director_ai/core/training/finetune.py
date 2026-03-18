# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Â© Concepts 1996â€“2026 Miroslav Ĺ otek. All rights reserved.
# Â© Code 2020â€“2026 Miroslav Ĺ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI â€” Domain Adaptation / Fine-tuning Pipeline

"""Fine-tune the NLI model on domain-specific labeled data.

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
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("DirectorAI.FineTune")

_DEFAULT_BASE_MODEL = "yaxili96/FactCG-DeBERTa-v3-Large"

# FactCG instruction template â€” must match inference-time template
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

    # Phase E: training optimization
    mix_general_data: bool = False
    general_data_path: str | None = None
    general_data_ratio: float = 0.2
    early_stopping_patience: int = 0
    class_weighted_loss: bool = False
    auto_benchmark: bool = False
    auto_onnx_export: bool = False


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
    regression_report: dict = field(default_factory=dict)
    onnx_path: str = ""
    mixed_general_samples: int = 0


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
            hypothesis = (
                row.get("hypothesis") or row.get("claim") or row.get("response", "")
            )
            label = row.get("label")
            if not premise or not hypothesis or label is None:
                logger.warning(
                    "Skipping line %d: missing premise/hypothesis/label",
                    line_num,
                )
                continue
            rows.append(
                {
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "label": int(label),
                },
            )
    logger.info("Loaded %d samples from %s", len(rows), path)
    return rows


def _mix_general_data(
    domain_rows: list[dict],
    general_path: str | Path | None,
    ratio: float,
    seed: int,
) -> tuple[list[dict], int]:
    """Mix general-purpose NLI data into domain data to prevent catastrophic forgetting.

    Returns (mixed_rows, n_general_added).
    """
    if general_path is None:
        pkg_dir = Path(__file__).parent.parent
        general_path = pkg_dir / "data" / "aggrefact_benchmark_1k.jsonl"

    general_path = Path(general_path)
    if not general_path.exists():
        logger.warning("General data not found at %s, skipping mix", general_path)
        return domain_rows, 0

    general_rows = _load_jsonl(general_path)
    if not general_rows:
        return domain_rows, 0

    n_general = int(len(domain_rows) * ratio / (1 - ratio))
    rng = random.Random(seed)
    if n_general < len(general_rows):
        general_sample = rng.sample(general_rows, n_general)
    else:
        general_sample = general_rows

    mixed = domain_rows + general_sample
    rng.shuffle(mixed)
    logger.info(
        "Mixed %d general samples into %d domain samples (%.0f%% general)",
        len(general_sample),
        len(domain_rows),
        len(general_sample) / len(mixed) * 100,
    )
    return mixed, len(general_sample)


def _compute_class_weights(rows: list[dict]) -> list[float]:
    """Compute inverse-frequency class weights for imbalanced datasets."""
    counts = Counter(r["label"] for r in rows)
    total = sum(counts.values())
    n_classes = len(counts)
    weights = []
    for label in sorted(counts.keys()):
        weights.append(total / (n_classes * counts[label]))
    return weights


def _prepare_dataset(rows: list[dict], tokenizer, max_length: int, is_factcg: bool):
    """Tokenize rows into a HuggingFace Dataset via batched map (OOM-safe)."""
    from datasets import Dataset

    if is_factcg:
        texts = [
            _FACTCG_TEMPLATE.format(premise=r["premise"], hypothesis=r["hypothesis"])
            for r in rows
        ]
        ds = Dataset.from_dict({"text": texts, "labels": [r["label"] for r in rows]})

        def _tok(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )

        ds = ds.map(_tok, batched=True, batch_size=256, remove_columns=["text"])
    else:
        premises = [r["premise"] for r in rows]
        hypotheses = [r["hypothesis"] for r in rows]
        ds = Dataset.from_dict(
            {
                "premise": premises,
                "hypothesis": hypotheses,
                "labels": [r["label"] for r in rows],
            },
        )

        def _tok_pair(batch):
            return tokenizer(
                batch["premise"],
                batch["hypothesis"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )

        ds = ds.map(
            _tok_pair,
            batched=True,
            batch_size=256,
            remove_columns=["premise", "hypothesis"],
        )

    ds.set_format("torch")
    return ds


def _compute_metrics(eval_pred):
    """Compute balanced accuracy + per-class metrics during training."""
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    bal_acc = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary", zero_division=0)
    return {"balanced_accuracy": bal_acc, "f1": f1}


def _make_weighted_trainer_class(class_weights: list[float]):
    """Create a Trainer subclass that applies class-weighted cross-entropy loss."""
    import torch
    from transformers import Trainer

    weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # type: ignore[override]
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor.to(logits.device))
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


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
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    if config is None:
        config = FinetuneConfig()

    train_rows = _load_jsonl(train_path)
    if not train_rows:
        raise ValueError(f"No valid samples in {train_path}")

    eval_rows = _load_jsonl(eval_path) if eval_path else None

    # Phase E: mix general data to prevent catastrophic forgetting
    n_general_mixed = 0
    if config.mix_general_data:
        train_rows, n_general_mixed = _mix_general_data(
            train_rows,
            config.general_data_path,
            config.general_data_ratio,
            config.seed,
        )

    # Phase E: compute class weights for imbalanced datasets
    class_weights = None
    if config.class_weighted_loss:
        class_weights = _compute_class_weights(train_rows)
        logger.info("Class weights: %s", class_weights)

    logger.info("Base model: %s", config.base_model)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        num_labels=2,
    )

    is_factcg = "factcg" in config.base_model.lower()

    train_dataset = _prepare_dataset(
        train_rows,
        tokenizer,
        config.max_length,
        is_factcg,
    )
    eval_dataset = (
        _prepare_dataset(eval_rows, tokenizer, config.max_length, is_factcg)
        if eval_rows
        else None
    )

    # save_strategy must match eval_strategy when load_best_model_at_end=True
    save_strat = "steps" if eval_dataset else config.save_strategy

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        fp16=config.fp16,
        seed=config.seed,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=config.eval_steps if eval_dataset else None,
        save_strategy=save_strat,
        save_steps=float(config.eval_steps) if eval_dataset else None,  # type: ignore[arg-type]
        save_total_limit=2,
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="balanced_accuracy" if eval_dataset else None,
        greater_is_better=True,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=4,
    )

    # Phase E: early stopping callback
    callbacks = []
    if config.early_stopping_patience > 0 and eval_dataset:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
            ),
        )

    # Phase E: class-weighted loss via custom Trainer
    trainer_cls = Trainer
    if class_weights is not None:
        trainer_cls = _make_weighted_trainer_class(class_weights)

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=_compute_metrics if eval_dataset else None,
        callbacks=callbacks or None,
    )

    logger.info(
        "Starting fine-tuning: %d train, %d eval, %d epochs",
        len(train_rows),
        len(eval_rows or []),
        config.epochs,
    )
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
        mixed_general_samples=n_general_mixed,
    )

    if eval_dataset:
        eval_metrics = trainer.evaluate()
        result.eval_metrics = eval_metrics
        result.best_balanced_accuracy = eval_metrics.get("eval_balanced_accuracy", 0.0)
        logger.info(
            "Best balanced accuracy: %.1f%%",
            result.best_balanced_accuracy * 100,
        )

    # Phase E: auto-benchmark against baseline
    if config.auto_benchmark:
        try:
            from director_ai.core.training.finetune_benchmark import (
                benchmark_finetuned_model,
            )

            report = benchmark_finetuned_model(
                config.output_dir,
                eval_path=eval_path,
            )
            result.regression_report = {
                "recommendation": report.recommendation,
                "general_accuracy": report.general_accuracy,
                "domain_accuracy": report.domain_accuracy,
                "regression_pp": report.regression_pp,
            }
            logger.info(
                "Anti-regression: %s (%.1fpp)",
                report.recommendation,
                report.regression_pp,
            )
        except Exception as exc:
            logger.warning("Auto-benchmark failed: %s", exc)

    # Phase E: auto ONNX export
    if config.auto_onnx_export:
        try:
            from director_ai.core.scoring.nli import export_onnx

            onnx_dir = str(Path(config.output_dir) / "onnx")
            export_onnx(config.output_dir, onnx_dir)
            result.onnx_path = onnx_dir
            logger.info("ONNX exported to %s", onnx_dir)
        except Exception as exc:
            logger.warning("ONNX export failed: %s", exc)

    return result
