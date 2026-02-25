#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Training Data Pipeline
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Build unified NLI training dataset from four sources (~100K examples):

- HaluEval (QA + Dialogue + Summarization) → ~30K
- FEVER (claims + evidence) → ~40K
- VitaminC (claims + evidence) → ~30K
- ANLI Round 3 (hardest NLI split) → ~20K

All normalised to (premise, hypothesis, label) with 3-class labels:
    0 = entailment, 1 = neutral, 2 = contradiction.

Usage::

    python training/data_pipeline.py
    # Output: training/data/ (HuggingFace Dataset on disk)
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "data"

LABEL_ENTAILMENT = 0
LABEL_NEUTRAL = 1
LABEL_CONTRADICTION = 2


def _load_halueval():
    """Load HaluEval QA + Dialogue + Summarization from HuggingFace."""
    from datasets import load_dataset

    examples = []

    for task in ("qa", "dialogue", "summarization"):
        logger.info("Loading HaluEval/%s ...", task)
        ds = load_dataset("pminervini/HaluEval", task, split="data")

        for row in ds:
            if task == "qa":
                premise = row.get("knowledge") or row.get("question", "")
                right = row.get("right_answer", "")
                halluc = row.get("hallucinated_answer", "")
            elif task == "dialogue":
                premise = row.get("dialogue_history") or row.get("knowledge", "")
                right = row.get("right_response", "")
                halluc = row.get("hallucinated_response", "")
            else:
                premise = row.get("document", "")
                right = row.get("right_summary", "")
                halluc = row.get("hallucinated_summary", "")

            if premise and right:
                examples.append(
                    {
                        "premise": premise,
                        "hypothesis": right,
                        "label": LABEL_ENTAILMENT,
                        "source": f"halueval_{task}",
                    }
                )
            if premise and halluc:
                examples.append(
                    {
                        "premise": premise,
                        "hypothesis": halluc,
                        "label": LABEL_CONTRADICTION,
                        "source": f"halueval_{task}",
                    }
                )

    logger.info("HaluEval: %d examples", len(examples))
    return examples


def _load_fever():
    """Load FEVER dataset (claims + evidence + labels)."""
    from datasets import load_dataset

    logger.info("Loading FEVER ...")
    ds = load_dataset("pietrolesci/nli_fever", split="train")

    label_map = {
        "entailment": LABEL_ENTAILMENT,
        "neutral": LABEL_NEUTRAL,
        "contradiction": LABEL_CONTRADICTION,
    }

    examples = []
    for row in ds:
        premise = row.get("premise", "")
        hypothesis = row.get("hypothesis", "")
        raw_label = row.get("label")

        if isinstance(raw_label, int):
            label = raw_label
        elif isinstance(raw_label, str):
            label = label_map.get(raw_label.lower())
        else:
            continue

        if label is None or not premise or not hypothesis:
            continue

        examples.append(
            {
                "premise": premise,
                "hypothesis": hypothesis,
                "label": label,
                "source": "fever",
            }
        )

    logger.info("FEVER: %d examples", len(examples))
    return examples


def _load_vitaminc():
    """Load VitaminC (fact verification with evidence)."""
    from datasets import load_dataset

    logger.info("Loading VitaminC ...")
    ds = load_dataset("tals/vitaminc", split="train")

    label_map = {
        "SUPPORTS": LABEL_ENTAILMENT,
        "REFUTES": LABEL_CONTRADICTION,
        "NOT ENOUGH INFO": LABEL_NEUTRAL,
    }

    examples = []
    for row in ds:
        premise = row.get("evidence", "")
        hypothesis = row.get("claim", "")
        raw_label = row.get("label")

        if isinstance(raw_label, int):
            label = raw_label
        elif isinstance(raw_label, str):
            label = label_map.get(raw_label.upper())
        else:
            continue

        if label is None or not premise or not hypothesis:
            continue

        examples.append(
            {
                "premise": premise,
                "hypothesis": hypothesis,
                "label": label,
                "source": "vitaminc",
            }
        )

    logger.info("VitaminC: %d examples", len(examples))
    return examples


def _load_anli_r3():
    """Load ANLI Round 3 (hardest adversarial NLI split)."""
    from datasets import load_dataset

    logger.info("Loading ANLI Round 3 ...")
    ds = load_dataset("anli", split="train_r3")

    examples = []
    for row in ds:
        premise = row.get("premise", "")
        hypothesis = row.get("hypothesis", "")
        label = row.get("label")

        if label is None or not premise or not hypothesis:
            continue

        examples.append(
            {
                "premise": premise,
                "hypothesis": hypothesis,
                "label": int(label),
                "source": "anli_r3",
            }
        )

    logger.info("ANLI R3: %d examples", len(examples))
    return examples


VITAMINC_CAP = 100_000  # cap VitaminC to ~30% of total (was 50.6%)


def build_dataset():
    """Build unified training dataset from all sources."""
    from datasets import Dataset, DatasetDict

    halueval = _load_halueval()
    fever = _load_fever()
    vitaminc = _load_vitaminc()
    anli = _load_anli_r3()

    # Cap VitaminC to prevent it dominating training (~370K → 100K)
    if len(vitaminc) > VITAMINC_CAP:
        import random

        random.seed(42)
        vitaminc = random.sample(vitaminc, VITAMINC_CAP)
        logger.info("VitaminC capped to %d examples", VITAMINC_CAP)

    all_examples = halueval + fever + vitaminc + anli
    logger.info("Total examples: %d", len(all_examples))

    ds = Dataset.from_list(all_examples)

    # Cast label to ClassLabel so stratified split works
    from datasets import ClassLabel

    ds = ds.cast_column(
        "label", ClassLabel(names=["entailment", "neutral", "contradiction"])
    )

    # Stratified 90/10 split by label
    split = ds.train_test_split(test_size=0.1, seed=42, stratify_by_column="label")
    dataset = DatasetDict({"train": split["train"], "eval": split["test"]})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(OUTPUT_DIR))
    logger.info("Saved to %s", OUTPUT_DIR)

    # Stats
    stats = {
        "total": len(all_examples),
        "train": len(dataset["train"]),
        "eval": len(dataset["eval"]),
        "label_distribution": dict(Counter(ex["label"] for ex in all_examples)),
        "source_distribution": dict(Counter(ex["source"] for ex in all_examples)),
    }
    stats_path = OUTPUT_DIR / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Stats: %s", json.dumps(stats, indent=2))

    return dataset


if __name__ == "__main__":
    build_dataset()
