"""Local JSONL loader for LLM-AggreFact — replaces gated HuggingFace call.

Monkey-patched into aggrefact_eval._load_aggrefact at runtime on GPU instances
where HF_TOKEN is not available but the JSONL was pre-downloaded.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("DirectorAI.Benchmark.AggreFact")

_DEFAULT_PATHS = [
    Path(__file__).parent / "aggrefact_test.jsonl",
    Path("/home/director-ai/benchmarks/aggrefact_test.jsonl"),
    Path("/home/user/director-ai/benchmarks/aggrefact_test.jsonl"),
]


def _load_aggrefact_local(max_samples: int | None = None) -> list[dict]:
    """Load LLM-AggreFact from local JSONL (no HF auth required)."""
    path_env = os.environ.get("AGGREFACT_JSONL")
    candidates = ([Path(path_env)] if path_env else []) + _DEFAULT_PATHS

    for p in candidates:
        if p.is_file():
            logger.info("Loading AggreFact from %s", p)
            with open(p, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            if max_samples:
                rows = rows[:max_samples]
            n_ds = len({r.get("dataset", "unknown") for r in rows})
            logger.info("Loaded %d samples across %d datasets", len(rows), n_ds)
            return rows

    searched = [str(p) for p in candidates]
    raise FileNotFoundError(
        f"aggrefact_test.jsonl not found. Searched: {searched}\n"
        "Transfer it first: scp root@<upcloud>:/home/director-ai/benchmarks/aggrefact_test.jsonl <dest>",
    )
