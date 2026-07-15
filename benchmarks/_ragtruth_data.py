# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — RAGTruth dataset download/cache helpers

"""RAGTruth corpus loading for the WCS-1 long-context sweep (second set).

RAGTruth (Niu et al., ACL 2024; github.com/ParticleMedia/RAGTruth, MIT)
carries span-annotated hallucination labels on model responses over three
RAG tasks. The long-context sweep uses the **Summary** task: CNN/DM-style
source documents (median ~2.5 k chars, ~37 % beyond the production
3 000-char prefix) with a response-level label derived from the span
annotations (hallucinated iff any span was annotated).

Files are fetched pinned to a specific upstream commit so the corpus
cannot drift under the benchmark, and cached next to the other benchmark
datasets.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("DirectorAI.Benchmark.RAGTruth")

_CACHE_DIR = Path(__file__).parent / ".cache"

#: Upstream commit the corpus is pinned to (ParticleMedia/RAGTruth@main,
#: resolved 2026-07-15 via the GitHub API).
_RAGTRUTH_COMMIT = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"

_BASE = (
    "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/"
    f"{_RAGTRUTH_COMMIT}/dataset"
)
_FILES = {
    "source_info": f"{_BASE}/source_info.jsonl",
    "response": f"{_BASE}/response.jsonl",
}


def _download_file(name: str) -> Path:
    """Fetch one RAGTruth jsonl (pinned revision), caching locally."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE_DIR / f"ragtruth_{name}_{_RAGTRUTH_COMMIT[:8]}.jsonl"
    if cache_path.exists():
        logger.info("Using cached RAGTruth %s", name)
        return cache_path

    import requests

    url = _FILES[name]
    logger.info("Downloading RAGTruth %s ...", name)
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    logger.info("Cached %s to %s", name, cache_path)
    return cache_path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_summary_rows(
    split: str = "test",
    *,
    source_path: Path | None = None,
    response_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return Summary-task rows: ``{doc, response, hallucinated}`` dicts.

    A response counts as hallucinated iff its span-annotation list is
    non-empty — the response-level reading of RAGTruth's span labels.
    ``source_path`` / ``response_path`` are injectable for offline tests;
    unset they download the pinned corpus.
    """
    sources = _read_jsonl(source_path or _download_file("source_info"))
    responses = _read_jsonl(response_path or _download_file("response"))

    summary_docs = {
        s["source_id"]: s["source_info"]
        for s in sources
        if s.get("task_type") == "Summary" and isinstance(s.get("source_info"), str)
    }
    rows: list[dict[str, Any]] = []
    for r in responses:
        if r.get("split") != split or r.get("source_id") not in summary_docs:
            continue
        response_text = r.get("response", "") or ""
        if not response_text:
            continue
        rows.append(
            {
                "doc": summary_docs[r["source_id"]],
                "response": response_text,
                "hallucinated": bool(r.get("labels")),
            }
        )
    logger.info(
        "RAGTruth Summary %s split: %d rows (%d hallucinated)",
        split,
        len(rows),
        sum(1 for r in rows if r["hallucinated"]),
    )
    return rows
