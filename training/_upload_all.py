# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — _upload_all
#!/usr/bin/env python3
"""Upload all training artifacts to Hugging Face Hub."""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN", "")
if not TOKEN:
    print("Set HF_TOKEN")
    sys.exit(1)

api = HfApi(token=TOKEN)
BASE = Path(__file__).parent

# --- 1. Large model (1.7 GB) ---

LARGE_DIR = BASE / "output" / "deberta-v3-large-hallucination"
LARGE_REPO = "anulum/deberta-v3-large-hallucination"
LARGE_CARD = """\
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

# DeBERTa-v3-large -- Hallucination Detection

Fine-tuned for hallucination detection as part of
[Director-AI](https://github.com/anulum/director-ai).

## Training

- **Base**: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
- **Data**: ~100K examples from HaluEval, FEVER, VitaminC, ANLI R3
- **Epochs**: 3, lr 2e-5, batch 32 (effective), class-weighted CE loss
- **Labels**: 0 = entailment, 1 = neutral, 2 = contradiction

## Usage

```python
from director_ai.core import NLIScorer

scorer = NLIScorer(model_name="anulum/deberta-v3-large-hallucination")
score = scorer.score("The capital of France is Paris.", "Paris is in Germany.")
```

## License

AGPL-3.0 | Commercial licensing: [anulum.li](https://www.anulum.li)
"""

if LARGE_DIR.exists():
    print("=== 1/3 Uploading deberta-v3-large-hallucination (1.7 GB) ===")
    api.create_repo(LARGE_REPO, exist_ok=True, private=False)
    (LARGE_DIR / "README.md").write_text(LARGE_CARD, encoding="utf-8")
    api.upload_folder(
        folder_path=str(LARGE_DIR),
        repo_id=LARGE_REPO,
        commit_message="Upload fine-tuned DeBERTa-v3-large hallucination model",
    )
    print(f"Done: https://huggingface.co/{LARGE_REPO}")
else:
    print(f"SKIP: {LARGE_DIR} not found")

# --- 2. Base model (712 MB final, skip checkpoints) ---

BASE_DIR = BASE / "output" / "deberta-v3-base-hallucination"
BASE_REPO = "anulum/deberta-v3-base-hallucination"
BASE_CARD = """\
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
base_model: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli-ling-wanli
pipeline_tag: text-classification
---

# DeBERTa-v3-base -- Hallucination Detection

Fine-tuned for hallucination detection as part of
[Director-AI](https://github.com/anulum/director-ai).

Smaller variant (184M params) of the large model. Lower accuracy but
faster inference — suitable for CPU deployments.

## Training

- **Base**: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli-ling-wanli
- **Data**: ~100K examples from HaluEval, FEVER, VitaminC, ANLI R3
- **Epochs**: 3, lr 2e-5, batch 32 (effective), class-weighted CE loss
- **Labels**: 0 = entailment, 1 = neutral, 2 = contradiction

## Usage

```python
from director_ai.core import NLIScorer

scorer = NLIScorer(model_name="anulum/deberta-v3-base-hallucination")
score = scorer.score("The capital of France is Paris.", "Paris is in Germany.")
```

## License

AGPL-3.0 | Commercial licensing: [anulum.li](https://www.anulum.li)
"""

if BASE_DIR.exists():
    print("=== 2/3 Uploading deberta-v3-base-hallucination (712 MB) ===")
    api.create_repo(BASE_REPO, exist_ok=True, private=False)
    (BASE_DIR / "README.md").write_text(BASE_CARD, encoding="utf-8")
    # Skip checkpoint dirs (6.3 GB of optimizer states)
    api.upload_folder(
        folder_path=str(BASE_DIR),
        repo_id=BASE_REPO,
        commit_message="Upload fine-tuned DeBERTa-v3-base hallucination model",
        ignore_patterns=["checkpoint-*"],
    )
    print(f"Done: https://huggingface.co/{BASE_REPO}")
else:
    print(f"SKIP: {BASE_DIR} not found")

# --- 3. Training dataset (skip caches, ~304 MB raw) ---

DATA_DIR = BASE / "data"
DATA_REPO = "anulum/director-ai-training-data"
DATA_CARD = """\
---
license: agpl-3.0
language: en
tags:
  - hallucination-detection
  - nli
  - director-ai
size_categories:
  - 10K<n<100K
---

# Director-AI Training Dataset

Combined training data for hallucination detection models used in
[Director-AI](https://github.com/anulum/director-ai).

## Sources

- HaluEval (pminervini/HaluEval)
- FEVER (pietrolesci/nli_fever)
- VitaminC (tals/vitaminc)
- ANLI R3 (anli)

~100K examples, 3-class (entailment / neutral / contradiction).

## License

AGPL-3.0 | Commercial licensing: [anulum.li](https://www.anulum.li)
"""

if DATA_DIR.exists():
    print("=== 3/3 Uploading training dataset (~304 MB) ===")
    api.create_repo(DATA_REPO, repo_type="dataset", exist_ok=True, private=False)
    (DATA_DIR / "README.md").write_text(DATA_CARD, encoding="utf-8")
    # Skip HF cache files (preprocessed arrow caches)
    api.upload_folder(
        folder_path=str(DATA_DIR),
        repo_id=DATA_REPO,
        repo_type="dataset",
        commit_message="Upload Director-AI training dataset",
        ignore_patterns=["cache-*"],
    )
    print(f"Done: https://huggingface.co/datasets/{DATA_REPO}")
else:
    print(f"SKIP: {DATA_DIR} not found")

print("\n=== All uploads complete ===")
