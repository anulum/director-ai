# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — token-level RAGTruth hallucination detector training

"""Train a token-level hallucination detector on RAGTruth (LettuceDetect-style).

RAGTruth hallucinations are localised character spans inside otherwise-grounded
responses; whole-response and claim-level NLI miss the "baseless addition" spans
that dominate the corpus (catch rate ~25% on isolated spans, F1 36.6% overall).
A token classifier learns the span pattern directly: it reads ``[context][SEP]
[response]`` and predicts, per response token, whether that token lies inside a
hallucinated span. The context is truncated (``only_first``) so the response is
always fully present and labellable.

Data: ``wandb/RAGTruth-processed`` (train 15090 / test 2700). The gold spans are
the JSON ``hallucination_labels`` character offsets into the response.

Env overrides: EPOCHS, BATCH_SIZE, GRAD_ACCUM, MAX_LENGTH, LR, BASE_MODEL,
MAX_TRAIN, OUTPUT_DIR, GRAD_CHECKPOINT, POS_WEIGHT_SCALE, FOCAL_GAMMA,
TRAIN_HARD_NEGATIVES, HARD_NEGATIVE_MAX_WEIGHT, HARD_NEGATIVE_FP_PENALTY.

Usage::

    python training/train_ragtruth_token.py --validate   # check span alignment
    python training/train_ragtruth_token.py               # full training
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("ragtruth_token")

BASE_MODEL = os.environ.get("BASE_MODEL", "answerdotai/ModernBERT-base")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "2048"))
OUTPUT_DIR = Path(
    os.environ.get("OUTPUT_DIR", "training/output/ragtruth-token-modernbert")
)
DATASET = "wandb/RAGTruth-processed"
LABEL_NAMES = ["supported", "hallucinated"]
TEST_SPLIT_NAMES = {"test", "validation", "eval"}


def parse_spans(raw: object) -> list[tuple[int, int]]:
    """Return (start, end) character spans from a RAGTruth hallucination_labels cell."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return []
    spans: list[tuple[int, int]] = []
    for item in raw or []:
        if isinstance(item, dict) and "start" in item and "end" in item:
            spans.append((int(item["start"]), int(item["end"])))
    return spans


def _overlaps(cs: int, ce: int, spans: list[tuple[int, int]]) -> bool:
    """True when token char range [cs, ce) intersects any hallucination span."""
    return any(not (ce <= s or cs >= e) for s, e in spans)


def _load_hard_negative_weights(
    path: str,
    *,
    max_weight: float,
) -> dict[int, float]:
    """Load non-test hard-negative row weights keyed by source row index."""
    if not path:
        return {}
    if max_weight < 1.0:
        raise ValueError("HARD_NEGATIVE_MAX_WEIGHT must be >= 1.0")

    weights: dict[int, float] = {}
    with Path(path).open() as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            split = str(row.get("source_split", "")).strip().lower()
            if split in TEST_SPLIT_NAMES:
                raise ValueError(
                    f"{path}:{line_no} comes from source_split={split!r}; "
                    "refusing to train on benchmark/eval hard negatives"
                )
            if "row_index" not in row:
                raise ValueError(f"{path}:{line_no} is missing row_index")
            row_index = int(row["row_index"])
            candidate = float(row.get("candidate_weight", max_weight))
            weights[row_index] = max(1.0, min(max_weight, candidate))
    return weights


def build_encoder(tokenizer, hard_negative_weights: dict[int, float] | None = None):
    """Return an ``encode(example) -> dict`` that aligns gold spans to token labels."""

    def encode(example: dict, row_index: int | None = None) -> dict:
        ctx = example["context"] or ""
        resp = example["output"] or ""
        spans = parse_spans(example.get("hallucination_labels"))
        enc = tokenizer(
            ctx,
            resp,
            truncation="only_first",
            max_length=MAX_LENGTH,
            return_offsets_mapping=True,
        )
        seq_ids = enc.sequence_ids()
        labels: list[int] = []
        for seq_id, (cs, ce) in zip(seq_ids, enc["offset_mapping"], strict=True):
            if seq_id != 1:  # context / special tokens are ignored in the loss
                labels.append(-100)
            else:
                labels.append(1 if _overlaps(cs, ce, spans) else 0)
        enc["labels"] = labels
        if hard_negative_weights is not None:
            enc["hard_negative_weight"] = hard_negative_weights.get(
                int(row_index or 0), 1.0
            )
        enc.pop("offset_mapping")
        return enc

    return encode


def validate(n: int = 6) -> None:
    """Print span→token alignment for a few examples so it can be eyeballed."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    encode = build_encoder(tok)
    ds = load_dataset(DATASET, split="train")
    shown = 0
    for ex in ds:
        spans = parse_spans(ex.get("hallucination_labels"))
        if not spans:
            continue
        enc = encode(ex)
        ids = enc["input_ids"]
        labels = enc["labels"]
        hall_tokens = [tok.decode([ids[i]]) for i, lab in enumerate(labels) if lab == 1]
        gold_text = [ex["output"][s:e] for s, e in spans]
        n_resp = sum(1 for lab in labels if lab != -100)
        print(f"\n=== example {shown} (task={ex['task_type']}) ===")
        print(f"  response tokens: {n_resp}, hallucinated tokens: {len(hall_tokens)}")
        print(f"  GOLD span text : {gold_text}")
        print(f"  TOKENS labelled: {''.join(hall_tokens)!r}")
        shown += 1
        if shown >= n:
            break


def compute_metrics(eval_pred):
    """Token-level precision/recall/F1 on the hallucinated class (ignores -100)."""
    from sklearn.metrics import precision_recall_fscore_support

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    mask = labels != -100
    y_true = labels[mask]
    y_pred = preds[mask]
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average="binary", zero_division=0
    )
    acc = float((y_true == y_pred).mean())
    return {"token_precision": p, "token_recall": r, "token_f1": f1, "token_acc": acc}


def _compute_weighted_token_loss(
    *,
    logits,
    labels,
    class_weights,
    focal_gamma: float,
    hard_negative_weight=None,
    hard_negative_fp_penalty: float = 0.0,
):
    """Compute weighted token loss with optional hard-negative FP suppression."""
    import torch

    flat_labels = labels.reshape(-1)
    flat_logits = logits.reshape(-1, 2)
    loss_values = torch.nn.functional.cross_entropy(
        flat_logits,
        flat_labels,
        weight=class_weights.to(logits.device),
        ignore_index=-100,
        reduction="none",
    ).reshape(labels.shape)

    if focal_gamma > 0:
        valid = labels != -100
        safe_labels = labels.clamp_min(0)
        probs = torch.softmax(flat_logits, dim=-1)
        pt = probs.gather(1, safe_labels.reshape(-1, 1)).squeeze(1)
        pt = pt.reshape(labels.shape)
        focal = torch.ones_like(loss_values)
        focal[valid] = (1.0 - pt[valid]).clamp_min(1e-8).pow(focal_gamma)
        loss_values = loss_values * focal

    hard_negative_multiplier = None
    if hard_negative_weight is not None:
        hard_negative_multiplier = (
            hard_negative_weight.to(logits.device).float().view(-1, 1)
        )
        loss_values = loss_values * hard_negative_multiplier

    if hard_negative_fp_penalty > 0 and hard_negative_multiplier is not None:
        probs = torch.softmax(logits, dim=-1)
        supported_hallucination_prob = probs[..., 1]
        hard_negative_rows = hard_negative_multiplier > 1.0
        supported_tokens = labels == 0
        penalty_mask = supported_tokens & hard_negative_rows
        penalty_values = torch.zeros_like(loss_values)
        row_strength = hard_negative_multiplier.clamp_min(1.0) - 1.0
        penalty_values[penalty_mask] = (
            supported_hallucination_prob[penalty_mask]
            * row_strength.expand_as(loss_values)[penalty_mask]
            * hard_negative_fp_penalty
        )
        loss_values = loss_values + penalty_values

    valid_loss = loss_values[labels != -100]
    return valid_loss.mean() if valid_loss.numel() else loss_values.mean()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    epochs = float(os.environ.get("EPOCHS", "3"))
    batch_size = int(os.environ.get("BATCH_SIZE", "4"))
    grad_accum = int(os.environ.get("GRAD_ACCUM", "4"))
    lr = float(os.environ.get("LR", "3e-5"))
    max_train = int(os.environ.get("MAX_TRAIN", "0"))  # 0 = all
    grad_ckpt = os.environ.get("GRAD_CHECKPOINT", "1") == "1"
    pos_weight_scale = float(os.environ.get("POS_WEIGHT_SCALE", "1.0"))
    focal_gamma = float(os.environ.get("FOCAL_GAMMA", "0.0"))
    train_hard_negatives = os.environ.get("TRAIN_HARD_NEGATIVES", "")
    hard_negative_max_weight = float(os.environ.get("HARD_NEGATIVE_MAX_WEIGHT", "5.0"))
    hard_negative_fp_penalty = float(os.environ.get("HARD_NEGATIVE_FP_PENALTY", "0.0"))
    if pos_weight_scale <= 0:
        raise ValueError("POS_WEIGHT_SCALE must be > 0")
    if focal_gamma < 0:
        raise ValueError("FOCAL_GAMMA must be >= 0")
    if hard_negative_fp_penalty < 0:
        raise ValueError("HARD_NEGATIVE_FP_PENALTY must be >= 0")
    hard_negative_weights = _load_hard_negative_weights(
        train_hard_negatives, max_weight=hard_negative_max_weight
    )
    if hard_negative_weights:
        logger.info(
            "loaded %d train hard-negative row weights from %s; max_weight=%.2f",
            len(hard_negative_weights),
            train_hard_negatives,
            hard_negative_max_weight,
        )

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    encode = build_encoder(tok, hard_negative_weights or None)

    eval_steps = int(os.environ.get("EVAL_STEPS", "600"))
    eval_subset = int(os.environ.get("EVAL_SUBSET", "500"))  # 0 = full test each eval

    train = load_dataset(DATASET, split="train")
    test = load_dataset(DATASET, split="test")
    if max_train:
        train = train.select(range(min(max_train, len(train))))
    cols = train.column_names
    train = train.map(
        encode,
        remove_columns=cols,
        desc="encode train",
        with_indices=bool(hard_negative_weights),
    )
    test_full = test.map(
        build_encoder(tok),
        remove_columns=test.column_names,
        desc="encode test",
    )
    # A small held-out slice keeps in-training evals fast on the 6 GB GTX 1060;
    # the full 2700-row test is scored once at the end for the headline number.
    test = (
        test_full.select(range(min(eval_subset, len(test_full))))
        if eval_subset
        else test_full
    )

    # Class weighting: hallucinated tokens are a small minority of response tokens.
    pos = sum(int((np.array(r["labels"]) == 1).sum()) for r in train)
    neg = sum(int((np.array(r["labels"]) == 0).sum()) for r in train)
    raw_pos_weight = max(1.0, neg / max(1, pos))
    pos_weight = max(1.0, raw_pos_weight * pos_weight_scale)
    logger.info(
        "token balance: supported=%d hallucinated=%d -> raw_pos_weight=%.2f "
        "scale=%.3f effective_pos_weight=%.2f focal_gamma=%.3f "
        "hard_negative_fp_penalty=%.3f",
        neg,
        pos,
        raw_pos_weight,
        pos_weight_scale,
        pos_weight,
        focal_gamma,
        hard_negative_fp_penalty,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label={0: "supported", 1: "hallucinated"},
        label2id={"supported": 0, "hallucinated": 1},
    )
    if grad_ckpt:
        model.gradient_checkpointing_enable()

    class_weights = torch.tensor([1.0, pos_weight])

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            hard_negative_weight = inputs.pop("hard_negative_weight", None)
            outputs = model(**inputs)
            loss = _compute_weighted_token_loss(
                logits=outputs.logits,
                labels=labels,
                class_weights=class_weights,
                focal_gamma=focal_gamma,
                hard_negative_weight=hard_negative_weight,
                hard_negative_fp_penalty=hard_negative_fp_penalty,
            )
            return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="token_f1",
        greater_is_better=True,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=test,
        processing_class=tok,
        data_collator=DataCollatorForTokenClassification(tok),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    # Headline number on the COMPLETE 2700-row test split, not the eval slice.
    metrics = trainer.evaluate(eval_dataset=test_full, metric_key_prefix="test")
    logger.info("FINAL token metrics (full test): %s", metrics)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tok.save_pretrained(str(OUTPUT_DIR))
    with open(OUTPUT_DIR / "token_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    model_path = OUTPUT_DIR / "model.safetensors"
    run_config = {
        "base_model": BASE_MODEL,
        "dataset": DATASET,
        "max_length": MAX_LENGTH,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "learning_rate": lr,
        "max_train": max_train,
        "grad_checkpoint": grad_ckpt,
        "eval_steps": eval_steps,
        "eval_subset": eval_subset,
        "raw_pos_weight": raw_pos_weight,
        "pos_weight_scale": pos_weight_scale,
        "effective_pos_weight": pos_weight,
        "focal_gamma": focal_gamma,
        "train_hard_negatives": train_hard_negatives or None,
        "hard_negative_count": len(hard_negative_weights),
        "hard_negative_max_weight": hard_negative_max_weight,
        "hard_negative_fp_penalty": hard_negative_fp_penalty,
        "model_sha256": _sha256(model_path) if model_path.is_file() else None,
    }
    with open(OUTPUT_DIR / "training_run_config.json", "w") as fh:
        json.dump(run_config, fh, indent=2, sort_keys=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="check span alignment only")
    a = ap.parse_args()
    if a.validate:
        validate()
    else:
        main()
