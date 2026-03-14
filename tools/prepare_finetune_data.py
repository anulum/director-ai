# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Fine-tuning Data Preparation
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""Download HuggingFace datasets and convert to unified JSONL for fine-tuning.

Produces three training sets:

  1. ``data/aggrefact_train.jsonl`` + ``data/aggrefact_eval.jsonl``
     — General factuality (LLM-AggreFact train split)
  2. ``data/medical_train.jsonl`` + ``data/medical_eval.jsonl``
     — Medical domain (MedNLI + PubMedQA)
  3. ``data/legal_train.jsonl`` + ``data/legal_eval.jsonl``
     — Legal domain (ContractNLI)

Output format (one JSON object per line)::

    {"premise": "...", "hypothesis": "...", "label": 1}

Labels: 1 = supported (entailment), 0 = not supported (contradiction).
Neutral samples are excluded.

Usage::

    pip install datasets
    export HF_TOKEN=hf_...
    python tools/prepare_finetune_data.py
    python tools/prepare_finetune_data.py --dataset aggrefact
    python tools/prepare_finetune_data.py --dataset medical --eval-ratio 0.1
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

logger = logging.getLogger("DirectorAI.DataPrep")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# HuggingFace dataset IDs
AGGREFACT_HF = "lytang/LLM-AggreFact"
MEDNLI_HF_IDS = ["medhalt/mednli", "bigbio/mednli"]
PUBMEDQA_HF = "qiaojin/PubMedQA"
CONTRACTNLI_HF_IDS = ["kiddothe2b/contract-nli", "law-ai/contract-nli"]


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    logger.info("Wrote %d samples to %s", len(rows), path)


def _split(
    rows: list[dict],
    eval_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_eval = max(1, int(len(rows) * eval_ratio))
    return rows[n_eval:], rows[:n_eval]


def prepare_aggrefact(eval_ratio: float = 0.1, seed: int = 42) -> None:
    """LLM-AggreFact: doc/claim/label. Label 1=supported, 0=not-supported."""
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    logger.info("Loading LLM-AggreFact (gated)...")

    rows = []
    for split in ["train", "test"]:
        try:
            ds = load_dataset(AGGREFACT_HF, split=split, token=token)
            for item in ds:
                doc = item.get("doc", "")
                claim = item.get("claim", "")
                label = item.get("label")
                if not doc or not claim or label is None:
                    continue
                rows.append(
                    {
                        "premise": doc,
                        "hypothesis": claim,
                        "label": int(label),
                    },
                )
        except Exception as exc:
            logger.warning("AggreFact split '%s' unavailable: %s", split, exc)

    logger.info("AggreFact total: %d samples", len(rows))
    train, eval_ = _split(rows, eval_ratio, seed)
    _write_jsonl(train, DATA_DIR / "aggrefact_train.jsonl")
    _write_jsonl(eval_, DATA_DIR / "aggrefact_eval.jsonl")


def _load_mednli_binary() -> list[dict]:
    """MedNLI: entailment (label=0) → 1, contradiction (label=2) → 0. Skip neutral."""
    from datasets import load_dataset

    rows = []
    for hf_id in MEDNLI_HF_IDS:
        try:
            for split in ["train", "validation", "test"]:
                try:
                    ds = load_dataset(hf_id, split=split, trust_remote_code=True)
                except Exception:
                    continue
                for item in ds:
                    premise = item.get("premise") or item.get("sentence1", "")
                    hypothesis = item.get("hypothesis") or item.get("sentence2", "")
                    label_raw = item.get("label")
                    if not premise or not hypothesis or label_raw is None:
                        continue
                    if isinstance(label_raw, str):
                        if label_raw.lower() == "neutral":
                            continue
                        label = 1 if label_raw.lower() == "entailment" else 0
                    else:
                        label_int = int(label_raw)
                        if label_int == 1:  # neutral
                            continue
                        label = (
                            1 if label_int == 0 else 0
                        )  # 0=entailment→1, 2=contradiction→0
                    rows.append(
                        {"premise": premise, "hypothesis": hypothesis, "label": label},
                    )
            if rows:
                logger.info(
                    "Loaded MedNLI from %s: %d binary samples",
                    hf_id,
                    len(rows),
                )
                return rows
        except Exception as exc:
            logger.debug("MedNLI source %s failed: %s", hf_id, exc)
    raise RuntimeError("MedNLI not available")


def _load_pubmedqa_binary() -> list[dict]:
    """PubMedQA: yes → supported (1), no/maybe → not-supported (0)."""
    from datasets import load_dataset

    ds = load_dataset(PUBMEDQA_HF, "pqa_labeled", split="train", trust_remote_code=True)
    rows = []
    for item in ds:
        contexts = item.get("context", {})
        if isinstance(contexts, dict):
            premise = " ".join(contexts.get("contexts", []))
        else:
            premise = str(contexts)
        hypothesis = item.get("long_answer", "")
        decision = item.get("final_decision", "")
        if not premise or not hypothesis:
            continue
        label = 1 if decision.lower() == "yes" else 0
        rows.append({"premise": premise, "hypothesis": hypothesis, "label": label})
    logger.info("Loaded PubMedQA: %d binary samples", len(rows))
    return rows


def prepare_medical(eval_ratio: float = 0.1, seed: int = 42) -> None:
    """Merge MedNLI + PubMedQA into medical training set."""
    rows = []
    try:
        rows.extend(_load_mednli_binary())
    except Exception as exc:
        logger.error("MedNLI failed: %s", exc)
    try:
        rows.extend(_load_pubmedqa_binary())
    except Exception as exc:
        logger.error("PubMedQA failed: %s", exc)

    if not rows:
        raise RuntimeError("No medical data loaded")

    logger.info("Medical total: %d samples", len(rows))
    train, eval_ = _split(rows, eval_ratio, seed)
    _write_jsonl(train, DATA_DIR / "medical_train.jsonl")
    _write_jsonl(eval_, DATA_DIR / "medical_eval.jsonl")


def _load_contractnli_binary() -> list[dict]:
    """ContractNLI: entailment → 1, contradiction → 0. Skip neutral."""
    from datasets import load_dataset

    rows = []
    for hf_id in CONTRACTNLI_HF_IDS:
        try:
            for split in ["train", "validation", "test"]:
                try:
                    ds = load_dataset(hf_id, split=split, trust_remote_code=True)
                except Exception:
                    continue
                for item in ds:
                    premise = item.get("premise", "") or item.get("context", "")
                    hypothesis = item.get("hypothesis", "") or item.get("statement", "")
                    label_raw = item.get("label")
                    if not premise or not hypothesis or label_raw is None:
                        continue
                    if isinstance(label_raw, str):
                        if label_raw.lower() == "neutral":
                            continue
                        label = 1 if label_raw.lower() == "entailment" else 0
                    else:
                        label_int = int(label_raw)
                        if label_int == 1:
                            continue
                        label = 1 if label_int == 0 else 0
                    rows.append(
                        {"premise": premise, "hypothesis": hypothesis, "label": label},
                    )
            if rows:
                logger.info(
                    "Loaded ContractNLI from %s: %d binary samples",
                    hf_id,
                    len(rows),
                )
                return rows
        except Exception as exc:
            logger.debug("ContractNLI source %s failed: %s", hf_id, exc)
    raise RuntimeError("ContractNLI not available")


def prepare_legal(eval_ratio: float = 0.1, seed: int = 42) -> None:
    """ContractNLI training set."""
    rows = _load_contractnli_binary()
    logger.info("Legal total: %d samples", len(rows))
    train, eval_ = _split(rows, eval_ratio, seed)
    _write_jsonl(train, DATA_DIR / "legal_train.jsonl")
    _write_jsonl(eval_, DATA_DIR / "legal_eval.jsonl")


PREPARERS = {
    "aggrefact": prepare_aggrefact,
    "medical": prepare_medical,
    "legal": prepare_legal,
}


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Prepare fine-tuning datasets")
    parser.add_argument(
        "--dataset",
        choices=["aggrefact", "medical", "legal", "all"],
        default="all",
    )
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    targets = list(PREPARERS.keys()) if args.dataset == "all" else [args.dataset]
    for name in targets:
        logger.info("Preparing %s...", name)
        try:
            PREPARERS[name](eval_ratio=args.eval_ratio, seed=args.seed)
        except Exception as exc:
            logger.error("%s failed: %s", name, exc)
    logger.info("Done. Output in %s", DATA_DIR)
