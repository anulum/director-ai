# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — HaluEval dataset download/cache helpers

"""HaluEval dataset loading, extracted from ``halueval_eval``.

``halueval_eval`` carries pytest entry points and therefore imports pytest
at module level, which made every downstream benchmark that only needed the
dataset loader (e2e_eval, the FPR diagnostics, run_longcontext_bench)
un-importable in a lean benchmark venv — the 2026-07-15 JarvisLabs WCS-1
run failed on exactly that. The loader lives here, dependency-light;
``halueval_eval`` re-exports it for its historical import surface.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("DirectorAI.Benchmark.HaluEval")

_CACHE_DIR = Path(__file__).parent / ".cache"

# HaluEval dataset URLs (HuggingFace parquet files)
_DATASET_URLS = {
    "qa": (
        "https://huggingface.co/datasets/pminervini/HaluEval/resolve/main/"
        "qa/data-00000-of-00001.parquet"
    ),
    "summarization": (
        "https://huggingface.co/datasets/pminervini/HaluEval/resolve/main/"
        "summarization/data-00000-of-00001.parquet"
    ),
    "dialogue": (
        "https://huggingface.co/datasets/pminervini/HaluEval/resolve/main/"
        "dialogue/data-00000-of-00001.parquet"
    ),
}


def _download_task_data(task: str) -> list[dict]:
    """Download a HaluEval task dataset (parquet), caching locally."""
    import pandas as pd

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE_DIR / f"halueval_{task}.parquet"

    if cache_path.exists():
        logger.info("Using cached HaluEval %s dataset", task)
        df = pd.read_parquet(cache_path)
        return df.to_dict(orient="records")

    url = _DATASET_URLS.get(task)
    if not url:
        raise ValueError(f"Unknown HaluEval task: {task}")

    import requests

    logger.info("Downloading HaluEval %s dataset...", task)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    df = pd.read_parquet(cache_path)
    logger.info("Cached %d samples to %s", len(df), cache_path)
    return df.to_dict(orient="records")
