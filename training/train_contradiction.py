# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — LoRA fine-tune of the streaming-halt contradiction model

"""LoRA fine-tune a 3-class NLI model to separate contradiction from unsupported.

The off-the-shelf MNLI model tops out around 0.51 recall on clean grounding
contradictions (benchmarks/contradiction_recall) because it is not specialised
for the contradiction-vs-merely-unsupported boundary the streaming halt needs.
This fine-tunes ``MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`` — a
strong 3-class NLI base whose ``id2label`` already matches our label scheme
(0 entailment, 1 neutral, 2 contradiction) — on the AggreFact-derived dataset
from ``build_contradiction_dataset`` (supported→entailment, injected→
contradiction, unsupported and cross-document→neutral).

LoRA is used because full fine-tuning of a 435M model does not fit the local
6 GB card; the base stays frozen and only low-rank adapters plus the
classification head train. Pascal GPUs lack bf16 and DeBERTa-v3 is numerically
unstable in fp16, so training runs in fp32 with gradient checkpointing.

Reproduce with ``python -m training.train_contradiction`` (after
``python -m training.build_contradiction_dataset``).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.optimization import Adafactor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
LABEL_NAMES = ["entailment", "neutral", "contradiction"]
_DATA_DIR = Path(__file__).parent / "data_contradiction"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, preds, average=None, labels=[0, 1, 2])
    rec = recall_score(labels, preds, average=None, labels=[0, 1, 2])
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", labels=[0, 1, 2]),
        "f1_entailment": f1[0],
        "f1_neutral": f1[1],
        "f1_contradiction": f1[2],
        "recall_contradiction": rec[2],
        "recall_neutral": rec[1],
    }


class WeightedTrainer(Trainer):
    """Cross-entropy with class weights and label smoothing for imbalance."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self._class_weights = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None
            else None
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = (
            self._class_weights.to(logits.device, dtype=torch.float32)
            if self._class_weights is not None
            else None
        )
        loss = torch.nn.functional.cross_entropy(
            logits.float(), labels, weight=weight, label_smoothing=0.05
        )
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    p = argparse.ArgumentParser(description="Contradiction-halt LoRA fine-tune")
    # Adafactor on this Pascal GPU: fused/elementwise AdamW corrupts weights to
    # NaN on sm_61 + torch 2.5.1+cu121, while Adafactor is numerically stable.
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-length", type=int, default=320)
    p.add_argument("--eval-cap", type=int, default=4000)
    p.add_argument("--max-train", type=int, default=0, help="0 = all")
    p.add_argument("--logging-steps", type=int, default=25)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="Enable gradient checkpointing (NaN-prone with DeBERTa-v3; off by "
        "default — LoRA on a frozen base fits without it).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent / "output" / "contradiction-lora"),
    )
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(str(_DATA_DIR))
    train_ds = dataset["train"]
    if args.max_train:
        train_ds = train_ds.select(range(min(args.max_train, len(train_ds))))
    eval_ds = dataset["eval"].select(range(min(args.eval_cap, len(dataset["eval"]))))
    logger.info(
        "train=%d eval=%d lr=%.1e rank=%d epochs=%d max_len=%d",
        len(train_ds), len(eval_ds), args.lr, args.rank, args.epochs, args.max_length,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=3
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=["query_proj", "key_proj", "value_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize(batch):
        return tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    cols = ["premise", "hypothesis", "source"]
    tok_train = train_ds.map(tokenize, batched=True, remove_columns=cols)
    tok_eval = eval_ds.map(tokenize, batched=True, remove_columns=cols)

    train_labels = np.array(tok_train["label"])
    weights = compute_class_weight(
        "balanced", classes=np.array([0, 1, 2]), y=train_labels
    )
    logger.info(
        "class weights: %s",
        dict(zip(LABEL_NAMES, [round(float(w), 3) for w in weights], strict=True)),
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.06,
        weight_decay=0.01,
        max_grad_norm=args.max_grad_norm,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        # Pascal: no bf16; DeBERTa-v3 is unstable in fp16 -> fp32.
        fp16=False,
        bf16=False,
        gradient_checkpointing=args.grad_checkpoint,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        report_to="none",
        dataloader_num_workers=2,
        label_names=["labels"],
    )

    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        weight_decay=0.01,
    )

    trainer = WeightedTrainer(
        class_weights=weights,
        model=model,
        args=training_args,
        train_dataset=tok_train,
        eval_dataset=tok_eval,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )

    device = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    )
    logger.info("Starting contradiction LoRA fine-tune on %s", device)
    trainer.train()

    logger.info("Saving adapter to %s", output_dir)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = trainer.evaluate()
    metrics["config"] = {
        "base_model": BASE_MODEL,
        "lr": args.lr,
        "rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "device": device,
    }
    (output_dir / "final_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            logger.info("  %s: %.4f", k, v)


if __name__ == "__main__":
    main()
