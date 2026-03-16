# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — LLM-AggreFact × MiniCheck Benchmark

"""Evaluate MiniCheck-DeBERTa-v3-Large on LLM-AggreFact.

MiniCheck chunks documents into sentences and scores each claim against
the most-supporting sentence, so it runs its own internal pipeline —
no threshold sweep needed (it returns binary predictions directly).

Usage::

    python -m benchmarks.aggrefact_minicheck
    python -m benchmarks.aggrefact_minicheck 500  # limit samples
"""

from __future__ import annotations

import logging
import time

import torch
from sklearn.metrics import balanced_accuracy_score

from benchmarks._common import save_results
from benchmarks.aggrefact_eval import (
    AggreFactMetrics,
    _load_aggrefact,
    _print_aggrefact_results,
)

logger = logging.getLogger("DirectorAI.Benchmark.MiniCheck")


def run_minicheck_aggrefact(
    max_samples: int | None = None,
    model_name: str = "deberta-v3-large",
) -> AggreFactMetrics:
    from minicheck.minicheck import MiniCheck

    logger.info("Loading MiniCheck model: %s", model_name)
    scorer = MiniCheck(model_name=model_name, cache_dir="./ckpts")
    rows = _load_aggrefact(max_samples)

    by_dataset: dict[str, list[tuple[int, float]]] = {}
    metrics = AggreFactMetrics(threshold=0.5)
    skipped_oom = 0

    for row in rows:
        doc = row.get("doc", "")
        claim = row.get("claim", "")
        label = row.get("label")
        ds_name = row.get("dataset", "unknown")

        if label is None or not doc or not claim:
            continue

        t0 = time.perf_counter()
        try:
            pred_label, raw_prob, _, _ = scorer.score(docs=[doc], claims=[claim])
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            skipped_oom += 1
            logger.warning(
                "OOM on sample (doc len=%d, dataset=%s) — skipping",
                len(doc),
                ds_name,
            )
            continue
        metrics.inference_times.append(time.perf_counter() - t0)

        support_prob = float(raw_prob[0])
        by_dataset.setdefault(ds_name, []).append((int(label), support_prob))

    if skipped_oom:
        logger.warning("Skipped %d samples due to OOM", skipped_oom)

    for ds_name in sorted(by_dataset.keys()):
        pairs = by_dataset[ds_name]
        y_true = [p[0] for p in pairs]
        y_pred = [1 if p[1] >= 0.5 else 0 for p in pairs]
        ba = balanced_accuracy_score(y_true, y_pred)
        metrics.per_dataset[ds_name] = {
            "total": len(pairs),
            "positive": sum(y_true),
            "negative": len(y_true) - sum(y_true),
            "balanced_acc": float(ba),
        }

    return metrics


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="MiniCheck × LLM-AggreFact benchmark")
    parser.add_argument(
        "max_samples",
        nargs="?",
        type=int,
        default=None,
        help="Limit evaluation samples (default: all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deberta-v3-large",
        help="MiniCheck model name (default: deberta-v3-large)",
    )
    args = parser.parse_args()

    m = run_minicheck_aggrefact(max_samples=args.max_samples, model_name=args.model)
    _print_aggrefact_results(m, f"MiniCheck-{args.model}")

    model_tag = args.model.replace("/", "_").replace("\\", "_")
    save_results(
        {
            "benchmark": "LLM-AggreFact",
            "model": f"MiniCheck-{args.model}",
            **m.to_dict(),
        },
        f"aggrefact_minicheck_{model_tag}.json",
    )
