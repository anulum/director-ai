# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — AggreFact Dataset Loading

"""Dataset list, published reference scores, and the gated loader for
LLM-AggreFact.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("DirectorAI.Benchmark.AggreFact")

AGGREFACT_DATASETS = [
    "AggreFact-CNN",
    "AggreFact-XSum",
    "TofuEval-MediaS",
    "TofuEval-MeetB",
    "Wice",
    "Reveal",
    "ClaimVerify",
    "FactCheck-GPT",
    "ExpertQA",
    "Lfqa",
    "RAGTruth",
]

# Published reference scores (balanced accuracy %) from the leaderboard
REFERENCE_SCORES = {
    "Bespoke-MiniCheck-7B": 77.4,
    "Claude-3.5-Sonnet": 77.2,
    "Granite-Guardian-3.3-8B": 76.5,
    "FactCG-DeBERTa-L (0.4B)": 75.6,
    "MiniCheck-Flan-T5-L (0.8B)": 75.0,
    "Llama-3.3-70B": 74.5,
    "HHEM-2.1": 71.8,
}


def _load_aggrefact(max_samples: int | None = None) -> list[dict]:
    """Load LLM-AggreFact test split. Requires HF authentication."""
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    logger.info("Loading LLM-AggreFact (gated dataset)...")
    ds = load_dataset("lytang/LLM-AggreFact", split="test", token=token)
    rows = list(ds)
    if max_samples:
        rows = rows[:max_samples]
    n_ds = len(set(r["dataset"] for r in rows))
    logger.info("Loaded %d samples across %d datasets", len(rows), n_ds)
    return rows
