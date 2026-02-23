#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Upload Fine-Tuned Model to HuggingFace Hub
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Push the fine-tuned hallucination detection model to HuggingFace Hub.

Usage::

    export HF_TOKEN=hf_...
    python training/upload_to_hub.py

Requires: ``pip install director-ai[train]``
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "output" / "deberta-v3-large-hallucination"
REPO_ID = "anulum/deberta-v3-large-hallucination"

MODEL_CARD = """\
---
license: agpl-3.0
language: en
tags:
  - hallucination-detection
  - nli
  - deberta-v3
  - director-ai
datasets:
  - pminervini/HaluEval
  - pietrolesci/nli_fever
  - tals/vitaminc
  - anli
base_model: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
pipeline_tag: text-classification
---

# DeBERTa-v3-large — Hallucination Detection

Fine-tuned for hallucination detection as part of
[Director-Class AI](https://github.com/anulum/director-ai).

## Training

- **Base**: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
- **Data**: ~100K examples from HaluEval, FEVER, VitaminC, ANLI R3
- **Epochs**: 3, lr 2e-5, batch 32 (effective), class-weighted CE loss
- **Labels**: 0 = entailment, 1 = neutral, 2 = contradiction

## Usage

```python
from director_ai.core import NLIScorer

scorer = NLIScorer(model_name="anulum/deberta-v3-large-hallucination")
score = scorer.score("The capital of France is Paris.", "Paris is in Germany.")
# score ≈ 0.9 (contradiction / hallucination)
```

## License

AGPL-3.0 — commercial licensing available via [anulum.li](https://www.anulum.li).
"""


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN environment variable")

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"{MODEL_DIR} not found — run train_hallucination_detector.py first"
        )

    from huggingface_hub import HfApi

    api = HfApi(token=token)

    logger.info("Creating/updating repo %s ...", REPO_ID)
    api.create_repo(REPO_ID, exist_ok=True, private=False)

    # Write model card
    card_path = MODEL_DIR / "README.md"
    card_path.write_text(MODEL_CARD, encoding="utf-8")

    logger.info("Uploading model to %s ...", REPO_ID)
    api.upload_folder(
        folder_path=str(MODEL_DIR),
        repo_id=REPO_ID,
        commit_message="Upload fine-tuned hallucination detection model",
    )
    logger.info("Done: https://huggingface.co/%s", REPO_ID)


if __name__ == "__main__":
    main()
