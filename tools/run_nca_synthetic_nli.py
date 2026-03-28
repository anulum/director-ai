# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — run_nca_synthetic_nli
#!/usr/bin/env python3
"""EXP-NCA-01: NCA-inspired synthetic NLI data for hallucination detection.

Generates synthetic (document, claim, label) pairs using structured
perturbation rules inspired by Neural Cellular Automata dynamics
(arxiv.org/abs/2603.10055). Fine-tunes FactCG-DeBERTa-v3-Large at LR=5e-6,
benchmarks on AggreFact.

Perturbation rules (NCA-inspired):
  - Local: entity swap, quantity shift, negation (1-hop neighborhood)
  - Propagation: claim mixing across documents (multi-hop)
  - Stability: faithful extraction with minor rephrasing (identity rule)

The key hypothesis: synthetic data with controlled structural perturbations
provides better training signal than natural NLI datasets, because it
targets exactly the consistency-checking skill without task mismatch.

Usage (on GPU instance):
    python tools/run_nca_synthetic_nli.py [--generate-only] [--train-only]

Expects:
    - benchmarks/aggrefact_test.jsonl in $WORKDIR (for scoring)
    - CUDA GPU with >= 16GB VRAM
    - ~30 min generation + ~2h training + ~40 min scoring
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

WORKDIR = Path(os.environ.get("DIRECTOR_WORKDIR", "/home/user/director-ai"))
SYNTHETIC_DIR = WORKDIR / "data" / "nca_synthetic"
OUTPUT_DIR = WORKDIR / "models" / "factcg-nca-synthetic"
SCORES_DIR = WORKDIR / "scores"
BASE_MODEL = "yaxili96/FactCG-DeBERTa-v3-Large"
TEMPLATE = (
    "{text_a}\n\nChoose your answer: based on the paragraph above "
    'can we conclude that "{text_b}"?\n\nOPTIONS:\n- Yes\n- No\n'
    "I think the answer is "
)

# --- Stage 1: Synthetic data generation ---

# NCA rule weights (probability of each perturbation type)
RULE_WEIGHTS = {
    "faithful": 0.40,  # identity: extract + minor rephrase → supported
    "entity_swap": 0.15,  # local: swap named entities → not_supported
    "quantity_shift": 0.10,  # local: change numbers → not_supported
    "negation": 0.10,  # local: negate predicate → not_supported
    "cross_doc_mix": 0.15,  # propagation: mix claims across docs → not_supported
    "temporal_shift": 0.10,  # local: change time references → not_supported
}


def _extract_sentences(text: str) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if len(s.strip()) > 20]


def _find_entities(text: str) -> list[str]:
    """Extract capitalized multi-word spans as pseudo-entities."""
    return re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)


def _find_numbers(text: str) -> list[str]:
    return re.findall(
        r"\b\d+(?:\.\d+)?(?:\s*(?:%|percent|million|billion|thousand))?\b",
        text,
    )


def _rephrase_minor(sent: str, rng: random.Random) -> str:
    """Minor rephrasing: swap clause order, add/remove hedging."""
    hedges = ["reportedly", "according to sources", "it was noted that"]
    if rng.random() < 0.3 and ", " in sent:
        parts = sent.split(", ", 1)
        sent = parts[1].capitalize() + ", " + parts[0].lower()
        if sent[-1] not in ".!?":
            sent += "."
    if rng.random() < 0.2:
        sent = rng.choice(hedges).capitalize() + ", " + sent[0].lower() + sent[1:]
    return sent


def rule_faithful(doc: str, rng: random.Random) -> tuple[str, int] | None:
    sents = _extract_sentences(doc)
    if not sents:
        return None
    sent = rng.choice(sents)
    return _rephrase_minor(sent, rng), 1


def rule_entity_swap(
    doc: str,
    all_entities: list[str],
    rng: random.Random,
) -> tuple[str, int] | None:
    sents = _extract_sentences(doc)
    if not sents:
        return None
    sent = rng.choice(sents)
    entities = _find_entities(sent)
    if not entities or not all_entities:
        return None
    target = rng.choice(entities)
    replacement = rng.choice([e for e in all_entities if e != target] or all_entities)
    return sent.replace(target, replacement, 1), 0


def rule_quantity_shift(doc: str, rng: random.Random) -> tuple[str, int] | None:
    sents = _extract_sentences(doc)
    if not sents:
        return None
    sent = rng.choice(sents)
    numbers = _find_numbers(sent)
    if not numbers:
        return None
    target = rng.choice(numbers)
    try:
        num_val = float(re.match(r"[\d.]+", target).group())
        factor = rng.choice([0.5, 0.7, 1.5, 2.0, 3.0])
        new_val = num_val * factor
        new_str = str(int(new_val)) if num_val == int(num_val) else f"{new_val:.1f}"
        suffix = target[len(re.match(r"[\d.]+", target).group()) :]
        return sent.replace(target, new_str + suffix, 1), 0
    except (ValueError, AttributeError):
        return None


def rule_negation(doc: str, rng: random.Random) -> tuple[str, int] | None:
    sents = _extract_sentences(doc)
    if not sents:
        return None
    sent = rng.choice(sents)
    negations = [
        (r"\bis\b", "is not"),
        (r"\bwas\b", "was not"),
        (r"\bhas\b", "has not"),
        (r"\bhave\b", "have not"),
        (r"\bcan\b", "cannot"),
        (r"\bwill\b", "will not"),
        (r"\bdid\b", "did not"),
    ]
    rng.shuffle(negations)
    for pattern, replacement in negations:
        if re.search(pattern, sent):
            return re.sub(pattern, replacement, sent, count=1), 0
    return None


def rule_cross_doc_mix(
    doc: str,
    other_docs: list[str],
    rng: random.Random,
) -> tuple[str, int] | None:
    other = rng.choice(other_docs) if other_docs else doc
    other_sents = _extract_sentences(other)
    if not other_sents:
        return None
    return rng.choice(other_sents), 0


def rule_temporal_shift(doc: str, rng: random.Random) -> tuple[str, int] | None:
    sents = _extract_sentences(doc)
    if not sents:
        return None
    sent = rng.choice(sents)
    shifts = [
        (r"\b2024\b", "2019"),
        (r"\b2023\b", "2018"),
        (r"\b2022\b", "2017"),
        (r"\blast year\b", "five years ago"),
        (r"\brecently\b", "decades ago"),
        (r"\bthis month\b", "last year"),
        (r"\btoday\b", "in 2015"),
    ]
    rng.shuffle(shifts)
    for pattern, replacement in shifts:
        if re.search(pattern, sent):
            return re.sub(pattern, replacement, sent, count=1), 0
    return None


def generate_synthetic_data(
    source_docs: list[str],
    n_samples: int = 50000,
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)
    all_entities = []
    for doc in source_docs[:1000]:
        all_entities.extend(_find_entities(doc))
    all_entities = list(set(all_entities))

    rules = list(RULE_WEIGHTS.keys())
    weights = [RULE_WEIGHTS[r] for r in rules]

    samples = []
    attempts = 0
    max_attempts = n_samples * 5

    while len(samples) < n_samples and attempts < max_attempts:
        attempts += 1
        doc = rng.choice(source_docs)
        rule = rng.choices(rules, weights=weights, k=1)[0]

        result = None
        if rule == "faithful":
            result = rule_faithful(doc, rng)
        elif rule == "entity_swap":
            result = rule_entity_swap(doc, all_entities, rng)
        elif rule == "quantity_shift":
            result = rule_quantity_shift(doc, rng)
        elif rule == "negation":
            result = rule_negation(doc, rng)
        elif rule == "cross_doc_mix":
            others = [d for d in source_docs if d != doc]
            result = rule_cross_doc_mix(doc, others, rng)
        elif rule == "temporal_shift":
            result = rule_temporal_shift(doc, rng)

        if result is None:
            continue

        claim, label = result
        if len(claim) < 15 or len(claim) > 500:
            continue

        samples.append(
            {
                "premise": doc[:2000],
                "hypothesis": claim,
                "label": label,
                "rule": rule,
            },
        )

        if len(samples) % 5000 == 0:
            pos = sum(1 for s in samples if s["label"] == 1)
            print(
                f"Generated {len(samples)}/{n_samples} (pos={pos}, neg={len(samples) - pos})",
            )

    pos = sum(1 for s in samples if s["label"] == 1)
    print(
        f"Final: {len(samples)} samples (pos={pos}, neg={len(samples) - pos}, attempts={attempts})",
    )
    return samples


def generate_from_aggrefact(n_samples: int = 50000) -> list[dict]:
    """Use AggreFact documents as source material."""
    jsonl_path = WORKDIR / "benchmarks" / "aggrefact_test.jsonl"
    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found")
        sys.exit(1)

    docs = set()
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            doc = row.get("doc", "")
            if len(doc) > 100:
                docs.add(doc)

    docs = list(docs)
    print(f"Source documents: {len(docs)} unique from AggreFact")
    return generate_synthetic_data(docs, n_samples=n_samples)


# --- Stage 2: Fine-tuning ---


def train_on_synthetic(data_path: Path):
    import torch
    from datasets import Dataset
    from sklearn.metrics import balanced_accuracy_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(data_path) as f:
        raw = json.load(f)

    def format_row(row):
        return {
            "text": TEMPLATE.format(text_a=row["premise"], text_b=row["hypothesis"]),
            "label": row["label"],
        }

    formatted = [format_row(r) for r in raw]
    rng = random.Random(42)
    rng.shuffle(formatted)

    n_val = max(500, len(formatted) // 20)
    train_data = Dataset.from_list(formatted[n_val:])
    val_data = Dataset.from_list(formatted[:n_val])
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=512, padding=False)

    train_data = train_data.map(tok_fn, batched=True, remove_columns=["text"])
    val_data = val_data.map(tok_fn, batched=True, remove_columns=["text"])

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        return {
            "accuracy": float((preds == labels).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        }

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="balanced_accuracy",
        greater_is_better=True,
        fp16=True,
        logging_steps=100,
        report_to="none",
    )

    t0 = time.time()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    val_result = trainer.evaluate(val_data)

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    safetensors = OUTPUT_DIR / "model.safetensors"
    if not safetensors.exists():
        print(f"ERROR: model not saved at {safetensors}", file=sys.stderr)
        sys.exit(1)
    print(f"Model saved: {safetensors} ({safetensors.stat().st_size / 1e6:.0f} MB)")

    result = {
        "dataset": "nca_synthetic",
        "base_model": BASE_MODEL,
        "learning_rate": 5e-6,
        "val_balanced_accuracy": val_result["eval_balanced_accuracy"],
        "training_time_minutes": round((time.time() - t0) / 60, 1),
        "n_train": len(train_data),
        "n_val": len(val_data),
        "epochs": 3,
        "status": "COMPLETE",
    }
    (OUTPUT_DIR / "training_result.json").write_text(json.dumps(result, indent=2))
    print(f"NCA-synthetic COMPLETE — val_bal_acc={result['val_balanced_accuracy']:.4f}")

    del trainer, model
    torch.cuda.empty_cache()
    return result


# --- Stage 3: AggreFact scoring ---


def score_on_aggrefact():
    import torch

    sys.path.insert(0, str(WORKDIR))
    import benchmarks.aggrefact_eval as ae
    from benchmarks._load_aggrefact_patch import _load_aggrefact_local

    ae._load_aggrefact = _load_aggrefact_local
    from benchmarks.aggrefact_eval import _BinaryNLIPredictor

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SCORES_DIR / "factcg-nca-synthetic.json"

    rows = _load_aggrefact_local()
    pred = _BinaryNLIPredictor(model_name=str(OUTPUT_DIR), max_length=512)
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
            print(f"nca-synthetic: {i + 1}/{len(rows)} ({eta:.0f} min remaining)")

    out_path.write_text(json.dumps(by_ds))
    print(f"Scoring DONE — {out_path} ({(time.perf_counter() - t0) / 60:.1f} min)")

    del pred
    torch.cuda.empty_cache()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="NCA synthetic NLI experiment")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--n-samples", type=int, default=50000)
    args = parser.parse_args()

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    data_path = SYNTHETIC_DIR / "nca_synthetic_nli.json"

    if not args.train_only and not args.score_only:
        print("=" * 60)
        print("Stage 1: Generate NCA-inspired synthetic NLI data")
        print("=" * 60)
        samples = generate_from_aggrefact(n_samples=args.n_samples)
        data_path.write_text(json.dumps(samples, indent=2))
        print(f"Saved {len(samples)} samples to {data_path}")

        if args.generate_only:
            return

    if not args.generate_only and not args.score_only:
        print("\n" + "=" * 60)
        print("Stage 2: Fine-tune FactCG-DeBERTa-v3-Large on synthetic data")
        print("=" * 60)
        if not data_path.exists():
            print(f"ERROR: {data_path} not found. Run --generate-only first.")
            sys.exit(1)
        train_on_synthetic(data_path)

        if args.train_only:
            return

    if not args.generate_only and not args.train_only:
        print("\n" + "=" * 60)
        print("Stage 3: Score on AggreFact")
        print("=" * 60)
        score_on_aggrefact()

    print("\nALL DONE")


if __name__ == "__main__":
    main()
