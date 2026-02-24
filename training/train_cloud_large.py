#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Cloud Fine-Tune DeBERTa-v3-large
# For UpCloud L40S (48GB VRAM) or any high-VRAM GPU
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Fine-tune DeBERTa-v3-large on the full 734K-example hallucination
detection dataset. Designed for cloud GPUs with 24GB+ VRAM.

Usage::

    python train_cloud_large.py
    # Output: ./output/deberta-v3-large-hallucination/
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import ClassLabel, Dataset, DatasetDict, load_dataset
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("./output/deberta-v3-large-hallucination")
DATA_DIR = Path("./data")
BASE_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
MAX_LENGTH = 512
LABEL_NAMES = ["entailment", "neutral", "contradiction"]

LABEL_ENTAILMENT = 0
LABEL_NEUTRAL = 1
LABEL_CONTRADICTION = 2


# ── Data Pipeline (inline, no external files needed) ────────────────

def _load_halueval():
    examples = []
    for task in ("qa", "dialogue", "summarization"):
        logger.info("Loading HaluEval/%s ...", task)
        ds = load_dataset("pminervini/HaluEval", task, split="data")
        for row in ds:
            if task == "qa":
                premise = row.get("knowledge") or row.get("question", "")
                right, halluc = row.get("right_answer", ""), row.get("hallucinated_answer", "")
            elif task == "dialogue":
                premise = row.get("dialogue_history") or row.get("knowledge", "")
                right, halluc = row.get("right_response", ""), row.get("hallucinated_response", "")
            else:
                premise = row.get("document", "")
                right, halluc = row.get("right_summary", ""), row.get("hallucinated_summary", "")
            if premise and right:
                examples.append({"premise": premise, "hypothesis": right, "label": LABEL_ENTAILMENT, "source": f"halueval_{task}"})
            if premise and halluc:
                examples.append({"premise": premise, "hypothesis": halluc, "label": LABEL_CONTRADICTION, "source": f"halueval_{task}"})
    logger.info("HaluEval: %d examples", len(examples))
    return examples


def _load_fever():
    logger.info("Loading FEVER ...")
    ds = load_dataset("pietrolesci/nli_fever", split="train")
    label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
    examples = []
    for row in ds:
        premise, hypothesis = row.get("premise", ""), row.get("hypothesis", "")
        raw = row.get("label")
        label = raw if isinstance(raw, int) else label_map.get(raw.lower()) if isinstance(raw, str) else None
        if label is not None and premise and hypothesis:
            examples.append({"premise": premise, "hypothesis": hypothesis, "label": label, "source": "fever"})
    logger.info("FEVER: %d examples", len(examples))
    return examples


def _load_vitaminc():
    logger.info("Loading VitaminC ...")
    ds = load_dataset("tals/vitaminc", split="train")
    label_map = {"SUPPORTS": 0, "REFUTES": 2, "NOT ENOUGH INFO": 1}
    examples = []
    for row in ds:
        premise, hypothesis = row.get("evidence", ""), row.get("claim", "")
        raw = row.get("label")
        label = raw if isinstance(raw, int) else label_map.get(raw.upper()) if isinstance(raw, str) else None
        if label is not None and premise and hypothesis:
            examples.append({"premise": premise, "hypothesis": hypothesis, "label": label, "source": "vitaminc"})
    logger.info("VitaminC: %d examples", len(examples))
    return examples


def _load_anli_r3():
    logger.info("Loading ANLI Round 3 ...")
    ds = load_dataset("anli", split="train_r3")
    examples = []
    for row in ds:
        premise, hypothesis, label = row.get("premise", ""), row.get("hypothesis", ""), row.get("label")
        if label is not None and premise and hypothesis:
            examples.append({"premise": premise, "hypothesis": hypothesis, "label": int(label), "source": "anli_r3"})
    logger.info("ANLI R3: %d examples", len(examples))
    return examples


def build_or_load_dataset():
    """Build dataset from HuggingFace sources, or load from disk if already built."""
    if DATA_DIR.exists() and (DATA_DIR / "dataset_dict.json").exists():
        logger.info("Loading cached dataset from %s", DATA_DIR)
        from datasets import load_from_disk
        return load_from_disk(str(DATA_DIR))

    all_examples = _load_halueval() + _load_fever() + _load_vitaminc() + _load_anli_r3()
    logger.info("Total examples: %d", len(all_examples))

    ds = Dataset.from_list(all_examples)
    ds = ds.cast_column("label", ClassLabel(names=LABEL_NAMES))
    split = ds.train_test_split(test_size=0.1, seed=42, stratify_by_column="label")
    dataset = DatasetDict({"train": split["train"], "eval": split["test"]})

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(DATA_DIR))

    stats = {
        "total": len(all_examples),
        "train": len(dataset["train"]),
        "eval": len(dataset["eval"]),
        "label_distribution": dict(Counter(ex["label"] for ex in all_examples)),
        "source_distribution": dict(Counter(ex["source"] for ex in all_examples)),
    }
    (DATA_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    logger.info("Stats: %s", json.dumps(stats, indent=2))
    return dataset


# ── Training ────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1_per = f1_score(labels, preds, average=None, labels=[0, 1, 2])
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
        "f1_entailment": f1_per[0],
        "f1_neutral": f1_per[1],
        "f1_contradiction": f1_per[2],
    }


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self._class_weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        w = self._class_weights.to(logits.device) if self._class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=w)
        return (loss, outputs) if return_outputs else loss


def main():
    dataset = build_or_load_dataset()
    logger.info("Train: %d, Eval: %d", len(dataset["train"]), len(dataset["eval"]))

    logger.info("Loading model: %s", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=3)

    def tokenize(batch):
        return tokenizer(batch["premise"], batch["hypothesis"], truncation=True, max_length=MAX_LENGTH, padding="max_length")

    logger.info("Tokenizing ...")
    tokenized = dataset.map(tokenize, batched=True, num_proc=4, remove_columns=["premise", "hypothesis", "source"])

    train_labels = np.array(tokenized["train"]["label"])
    weights = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=train_labels)
    logger.info("Class weights: %s", dict(zip(LABEL_NAMES, weights)))

    # L40S 48GB: batch 16 fits comfortably, accumulate 2 for effective 32
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        warmup_ratio=0.06,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=True,
        logging_steps=100,
        report_to="none",
        dataloader_num_workers=4,
        label_names=["labels"],
    )

    trainer = WeightedTrainer(
        class_weights=weights,
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["eval"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    logger.info("Starting training ...")
    trainer.train()

    logger.info("Saving best model to %s", OUTPUT_DIR)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    logger.info("Final evaluation:")
    metrics = trainer.evaluate()
    for k, v in sorted(metrics.items()):
        logger.info("  %s: %.4f", k, v)

    # Save metrics to file for easy retrieval
    metrics_path = OUTPUT_DIR / "final_metrics.json"
    metrics_path.write_text(json.dumps({k: round(v, 4) for k, v in metrics.items()}, indent=2))
    logger.info("Metrics saved to %s", metrics_path)


if __name__ == "__main__":
    main()
