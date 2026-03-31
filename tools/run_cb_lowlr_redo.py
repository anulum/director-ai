# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — run_cb_lowlr_redo
#!/usr/bin/env python3
"""CB-lowLR redo: CommitmentBank at LR=5e-6, with verified save + auto-benchmark.

Previous run lost the model (trainer.save_model() failure). This version
verifies the checkpoint exists before proceeding to AggreFact scoring.

Usage (on GPU instance):
    python tools/run_cb_lowlr_redo.py

Expects:
    - benchmarks/aggrefact_test.jsonl in $WORKDIR
    - CUDA GPU with >= 8GB VRAM
    - ~15 min training + ~40 min scoring = ~55 min total
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import balanced_accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "yaxili96/FactCG-DeBERTa-v3-Large"
WORKDIR = Path(os.environ.get("DIRECTOR_WORKDIR", "/home/user/director-ai"))
OUTPUT_DIR = WORKDIR / "models" / "factcg-cb-lowlr"
SCORES_DIR = WORKDIR / "scores"
TEMPLATE = (
    "{text_a}\n\nChoose your answer: based on the paragraph above "
    'can we conclude that "{text_b}"?\n\nOPTIONS:\n- Yes\n- No\n'
    "I think the answer is "
)


def train_cb_lowlr():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_DIR.mkdir(parents=True, exist_ok=True)

    ds_raw = load_dataset("super_glue", "cb")

    def convert(ex):
        label = 1 if ex["label"] == 0 else 0
        return {
            "text": TEMPLATE.format(text_a=ex["premise"], text_b=ex["hypothesis"]),
            "label": label,
        }

    train = ds_raw["train"].map(convert, remove_columns=ds_raw["train"].column_names)
    val_split = ds_raw["validation"].train_test_split(test_size=0.5, seed=42)
    val = val_split["train"].map(
        convert,
        remove_columns=val_split["train"].column_names,
    )
    test = val_split["test"].map(convert, remove_columns=val_split["test"].column_names)
    print(f"CB: train={len(train)}, val={len(val)}, test={len(test)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=512, padding=False)

    train = train.map(tok_fn, batched=True, remove_columns=["text"])
    val = val.map(tok_fn, batched=True, remove_columns=["text"])
    test = test.map(tok_fn, batched=True, remove_columns=["text"])

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        return {
            "accuracy": float((preds == labels).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        }

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=20,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=32,
        learning_rate=5e-6,
        weight_decay=0.01,
        warmup_ratio=0.2,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="balanced_accuracy",
        greater_is_better=True,
        fp16=True,
        logging_steps=10,
        report_to="none",
    )

    t0 = time.time()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=val,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    test_result = trainer.evaluate(test)

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Verify save succeeded
    safetensors = OUTPUT_DIR / "model.safetensors"
    if not safetensors.exists():
        print(f"ERROR: model.safetensors not found at {safetensors}", file=sys.stderr)
        sys.exit(1)
    size_mb = safetensors.stat().st_size / 1e6
    if size_mb < 1000:
        print(f"ERROR: model.safetensors too small ({size_mb:.0f} MB)", file=sys.stderr)
        sys.exit(1)
    print(f"Model saved: {safetensors} ({size_mb:.0f} MB)")

    result = {
        "dataset": "cb_superglue",
        "base_model": BASE_MODEL,
        "learning_rate": 5e-6,
        "test_balanced_accuracy": test_result["eval_balanced_accuracy"],
        "test_accuracy": test_result["eval_accuracy"],
        "training_time_minutes": round((time.time() - t0) / 60, 1),
        "epochs": 20,
        "status": "COMPLETE",
    }
    result_path = OUTPUT_DIR / "training_result.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(f"CB-lowLR COMPLETE — bal_acc={result['test_balanced_accuracy']:.4f}")

    del trainer, model
    torch.cuda.empty_cache()
    return result


def score_on_aggrefact():
    """Score the CB-lowLR model on AggreFact 29K."""
    sys.path.insert(0, str(WORKDIR))
    import benchmarks.aggrefact_eval as ae
    from benchmarks._load_aggrefact_patch import _load_aggrefact_local

    ae._load_aggrefact = _load_aggrefact_local
    from benchmarks.aggrefact_eval import _BinaryNLIPredictor

    model_path = str(OUTPUT_DIR)
    out_path = SCORES_DIR / "factcg-cb-lowlr.json"

    rows = _load_aggrefact_local()
    pred = _BinaryNLIPredictor(model_name=model_path, max_length=512)
    by_ds: dict = {}
    t0 = time.perf_counter()

    for i, row in enumerate(rows):
        doc, claim = row.get("doc", ""), row.get("claim", "")
        lbl, ds = row.get("label"), row.get("dataset", "unknown")
        if lbl is None or not doc or not claim:
            continue
        prob = pred.score(doc, claim)
        by_ds.setdefault(ds, []).append((int(lbl), float(prob)))
        if (i + 1) % 2000 == 0:
            elapsed = time.perf_counter() - t0
            eta = (len(rows) - i - 1) * elapsed / (i + 1) / 60
            print(f"cb-lowlr: {i + 1}/{len(rows)} ({eta:.0f} min remaining)")

    out_path.write_text(json.dumps(by_ds))
    elapsed = time.perf_counter() - t0
    print(f"Scoring DONE — {out_path} ({elapsed / 60:.1f} min)")

    del pred
    torch.cuda.empty_cache()


def main():
    print("=" * 60)
    print("CB-lowLR Redo: train + AggreFact benchmark")
    print(
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
    )
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    train_cb_lowlr()
    score_on_aggrefact()
    print("\nALL DONE — download scores/factcg-cb-lowlr.json + models/factcg-cb-lowlr/")


if __name__ == "__main__":
    main()
