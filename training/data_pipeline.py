# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Training Data Pipeline
"""
Build unified NLI training dataset from seven sources:

- HaluEval (QA + Dialogue + Summarization) → ~60K
- FEVER (claims + evidence) → ~203K
- VitaminC (claims + evidence, capped) → ~100K
- ANLI Round 3 (hardest NLI split) → ~100K
- RAGTruth (RAG hallucination labels) → ~variable
- SummaC (summarisation consistency) → ~variable
- LLM-AggreFact (11 sub-datasets, gated) → ~29K

All normalised to (premise, hypothesis, label) with 3-class labels:
    0 = entailment, 1 = neutral, 2 = contradiction.

Usage::

    python training/data_pipeline.py
    python training/data_pipeline.py --include-ragtruth --include-summac
    python training/data_pipeline.py --include-aggrefact  # requires HF_TOKEN
    python training/data_pipeline.py --all  # all sources
    # Output: training/data/ (HuggingFace Dataset on disk)
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
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


def _load_ragtruth():
    """Load RAGTruth (RAG hallucination detection) from HuggingFace.

    Uses wandb/RAGTruth-processed with per-response hallucination labels.
    Maps: no hallucination → entailment, hallucination → contradiction.
    """
    from datasets import load_dataset

    logger.info("Loading RAGTruth (wandb/RAGTruth-processed) ...")
    ds = load_dataset("wandb/RAGTruth-processed", split="test")

    examples = []
    for row in ds:
        context = row.get("context", "")
        response = row.get("output", "")
        if not context or not response:
            continue

        labels_raw = row.get("hallucination_labels_processed", "{}")
        if isinstance(labels_raw, str):
            labels = ast.literal_eval(labels_raw)
        else:
            labels = labels_raw

        is_hallucinated = (labels.get("evident_conflict", 0) > 0) or (
            labels.get("baseless_info", 0) > 0
        )

        examples.append(
            {
                "premise": context[:2000],
                "hypothesis": response[:2000],
                "label": LABEL_CONTRADICTION if is_hallucinated else LABEL_ENTAILMENT,
                "source": "ragtruth",
            }
        )

    logger.info("RAGTruth: %d examples", len(examples))
    return examples


def _load_summac():
    """Load SummaC (summarisation consistency) from HuggingFace.

    Binary labels: 1 = consistent (entailment), 0 = inconsistent (contradiction).
    """
    from datasets import load_dataset

    logger.info("Loading SummaC (mteb/summac) ...")
    ds = load_dataset("mteb/summac")

    label_map = {1: LABEL_ENTAILMENT, 0: LABEL_CONTRADICTION}

    examples = []
    for split_name in ds:
        for row in ds[split_name]:
            doc = str(row.get("text", "") or row.get("document", ""))
            claim = str(row.get("claim", "") or row.get("summary", ""))
            label = row.get("label")

            if not doc or not claim or label not in label_map:
                continue

            examples.append(
                {
                    "premise": doc[:2000],
                    "hypothesis": claim,
                    "label": label_map[label],
                    "source": "summac",
                }
            )

    logger.info("SummaC: %d examples", len(examples))
    return examples


def _load_aggrefact():
    """Load LLM-AggreFact (gated, requires HF_TOKEN).

    11 sub-datasets: summarisation, RAG, and grounding tasks.
    Binary labels: 1 = supported (entailment), 0 = not supported (contradiction).
    """
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.warning("HF_TOKEN not set — skipping LLM-AggreFact (gated dataset)")
        return []

    logger.info("Loading LLM-AggreFact (lytang/LLM-AggreFact) ...")
    ds = load_dataset("lytang/LLM-AggreFact", split="test", token=token)

    label_map = {1: LABEL_ENTAILMENT, 0: LABEL_CONTRADICTION}

    examples = []
    for row in ds:
        doc = row.get("doc", "")
        claim = row.get("claim", "")
        label = row.get("label")
        ds_name = row.get("dataset", "aggrefact")

        if not doc or not claim or label not in label_map:
            continue

        examples.append(
            {
                "premise": doc[:2000],
                "hypothesis": claim,
                "label": label_map[label],
                "source": f"aggrefact_{ds_name}",
            }
        )

    logger.info("LLM-AggreFact: %d examples", len(examples))
    return examples


VITAMINC_CAP = 100_000  # cap VitaminC to ~30% of total (was 50.6%)


def build_dataset(
    include_ragtruth: bool = False,
    include_summac: bool = False,
    include_aggrefact: bool = False,
):
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

    if include_ragtruth:
        ragtruth = _load_ragtruth()
        all_examples += ragtruth

    if include_summac:
        try:
            summac = _load_summac()
            all_examples += summac
        except Exception as exc:
            logger.warning("SummaC loading failed (may be unavailable): %s", exc)

    if include_aggrefact:
        aggrefact = _load_aggrefact()
        all_examples += aggrefact

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
    parser = argparse.ArgumentParser(description="Build unified NLI training dataset")
    parser.add_argument(
        "--include-ragtruth", action="store_true", help="Include RAGTruth dataset"
    )
    parser.add_argument(
        "--include-summac", action="store_true", help="Include SummaC dataset"
    )
    parser.add_argument(
        "--include-aggrefact",
        action="store_true",
        help="Include LLM-AggreFact (requires HF_TOKEN)",
    )
    parser.add_argument("--all", action="store_true", help="Include all sources")
    args = parser.parse_args()

    if args.all:
        args.include_ragtruth = True
        args.include_summac = True
        args.include_aggrefact = True

    build_dataset(
        include_ragtruth=args.include_ragtruth,
        include_summac=args.include_summac,
        include_aggrefact=args.include_aggrefact,
    )
