# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — held-out contradiction-halt evaluation (leakage-free)

"""Score a model's halt signal on the held-out contradiction eval split.

The ``contradiction_recall`` benchmark re-injects contradictions from *all*
supported AggreFact claims, which overlap the fine-tune training set — measuring
a fine-tuned model there leaks. This evaluates ``P(contradiction)`` on the
``data_contradiction`` eval split only (held out at dataset-build time), so the
off-the-shelf base and the fine-tuned model are compared on identical, unseen
data with the production halt rule (halt when ``P(contradiction) >= threshold``):

* recall   = fraction of true contradictions caught (label 2);
* false-halt = fraction of entailment/neutral wrongly halted (labels 0, 1).

Run for each model with ``--model`` (a HuggingFace id or a local merged-model
directory) and ``--device``; results go to
``benchmarks/results/contradiction_holdout_<tag>.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks._common import RESULTS_DIR
from benchmarks.contradiction_aggrefact import _auc, _device_label
from director_ai.core.scoring.contradiction import (
    DEFAULT_CONTRADICTION_MODEL,
    ContradictionScorer,
)

_DATA_DIR = Path(__file__).parent / "data_contradiction"


def run(model_id: str, *, device: int = -1, batch_size: int = 64) -> dict:
    from datasets import load_from_disk

    eval_ds = load_from_disk(str(_DATA_DIR))["eval"]
    premises = eval_ds["premise"]
    hypotheses = eval_ds["hypothesis"]
    labels = eval_ds["label"]

    scorer = ContradictionScorer.from_pretrained(model_id, device=device)
    scores: list[float] = []
    pairs = list(zip(premises, hypotheses, strict=True))
    for i in range(0, len(pairs), batch_size):
        scores.extend(scorer.contradiction_batch(pairs[i : i + batch_size]))
        if i % (batch_size * 20) == 0:
            print(f"  scored {i}/{len(pairs)}", flush=True)

    contra = [s for s, y in zip(scores, labels, strict=True) if y == 2]
    other = [s for s, y in zip(scores, labels, strict=True) if y in (0, 1)]
    neutral = [s for s, y in zip(scores, labels, strict=True) if y == 1]
    entail = [s for s, y in zip(scores, labels, strict=True) if y == 0]

    auc_labels = [1] * len(contra) + [0] * len(other)
    thresholds = {}
    for thr in (0.1, 0.2, 0.3, 0.5):
        thresholds[str(thr)] = {
            "recall_contradiction": round(
                sum(1 for s in contra if s >= thr) / len(contra), 4
            ),
            "false_halt_all": round(
                sum(1 for s in other if s >= thr) / len(other), 4
            ),
            "false_halt_neutral": round(
                sum(1 for s in neutral if s >= thr) / len(neutral), 4
            ),
            "false_halt_entailment": round(
                sum(1 for s in entail if s >= thr) / len(entail), 4
            ),
        }
    return {
        "benchmark": "contradiction_holdout",
        "model": model_id,
        "device": _device_label(device),
        "n_eval": len(pairs),
        "n_contradiction": len(contra),
        "n_neutral": len(neutral),
        "n_entailment": len(entail),
        "auc_contradiction_vs_other": round(_auc(auc_labels, contra + other), 4),
        "thresholds": thresholds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_CONTRADICTION_MODEL)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tag", default="base")
    args = parser.parse_args()
    result = run(args.model, device=args.device, batch_size=args.batch_size)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"contradiction_holdout_{args.tag}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nHeld-out contradiction halt ({args.tag}, {result['device']}):")
    print(f"  model: {result['model']}")
    print(f"  AUC contradiction-vs-other: {result['auc_contradiction_vs_other']}")
    for thr, m in result["thresholds"].items():
        print(
            f"  thr={thr}: recall={m['recall_contradiction']:.3f} "
            f"| false-halt(all)={m['false_halt_all']:.3f} "
            f"(neutral {m['false_halt_neutral']:.3f}, "
            f"entail {m['false_halt_entailment']:.3f})"
        )
    print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
