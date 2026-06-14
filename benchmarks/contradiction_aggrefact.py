# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — contradiction signal on the full LLM-AggreFact corpus

"""Evaluate the streaming-halt contradiction signal on all 29,320 LLM-AggreFact
claims.

For each (document, claim) the :class:`ContradictionScorer` returns
``P(contradiction)``. AggreFact labels each claim ``1`` (supported by the
document) or ``0`` (not supported). A streaming halt should fire on a claim that
*contradicts* the grounding, so:

* on ``label==1`` (supported) ``P(contradiction)`` is the **false-halt** signal —
  it must stay low, or the guard halts correct text;
* on ``label==0`` (unsupported) it is the **recall** signal — but AggreFact's
  unsupported class mixes genuine contradictions with merely-unsupported claims,
  and a contradiction detector deliberately does not fire on the latter, so
  recall here is a *lower bound* on contradiction recall, not an upper bound on
  what should halt. Both numbers are reported per sub-dataset and overall, with
  AUC, so the honest separation is visible at full scale rather than on a sample.

Output: ``benchmarks/results/contradiction_aggrefact.json``. Reproduce with
``python -m benchmarks.contradiction_aggrefact`` (``--max-samples N`` to subset,
``--batch-size`` to tune throughput).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from benchmarks._common import save_results, select_relevant_passages
from director_ai.core.scoring.contradiction import ContradictionScorer

_DATA = Path(__file__).resolve().parent / "aggrefact_test.jsonl"


def _load() -> list[dict]:
    return [json.loads(line) for line in _DATA.read_text(encoding="utf-8").splitlines()]


def _auc(labels: list[int], scores: list[float]) -> float:
    """Rank-based AUC (probability a positive outranks a negative)."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y == 1]
    neg = [s for s, y in zip(scores, labels, strict=True) if y == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_pos = sum(ranks[i] for i in range(len(scores)) if labels[i] == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _metrics(con_inconsistent: list[float], con_supported: list[float]) -> dict:
    # contradiction predicts the "inconsistent" (label 0) class
    labels = [1] * len(con_inconsistent) + [0] * len(con_supported)
    scores = con_inconsistent + con_supported
    out: dict[str, object] = {
        "n_supported": len(con_supported),
        "n_inconsistent": len(con_inconsistent),
        "auc_contra_vs_inconsistent": round(_auc(labels, scores), 4),
        "thresholds": {},
    }
    for thr in (0.1, 0.2, 0.3, 0.5):
        fh = sum(1 for x in con_supported if x >= thr)
        rec = sum(1 for x in con_inconsistent if x >= thr)
        out["thresholds"][str(thr)] = {  # type: ignore[index]
            "false_halt_rate": round(fh / len(con_supported), 4)
            if con_supported
            else 0.0,
            "recall": round(rec / len(con_inconsistent), 4)
            if con_inconsistent
            else 0.0,
        }
    return out


def _device_label(device: int) -> str:
    """Resolve the device into a recorded label for benchmark provenance."""
    if device < 0:
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{device} ({torch.cuda.get_device_name(device)})"
    except ImportError:
        pass
    return f"cuda:{device} (requested, unavailable — ran on cpu)"


def _batch_scores(
    scorer: ContradictionScorer,
    pairs: list[tuple[str, str]],
    batch_size: int,
) -> list[float]:
    out: list[float] = []
    for i in range(0, len(pairs), batch_size):
        out.extend(scorer.contradiction_batch(pairs[i : i + batch_size]))
        if i % (batch_size * 20) == 0:
            print(f"  scored {i}/{len(pairs)}", flush=True)
    return out


def _score_rows(
    scorer: ContradictionScorer,
    rows: list[dict],
    *,
    granularity: str,
    top_k: int,
    batch_size: int,
) -> list[float]:
    """Per-row ``P(contradiction)`` at the chosen grounding granularity.

    ``document`` scores the claim against the truncated whole document (the
    original baseline). ``passage`` mirrors the production halt: the claim is
    scored against each of the top-k most relevant document passages and the
    strongest contradiction is kept, so the premise is focused rather than
    truncated. A row with no usable passage scores 0.0 (ungrounded is not a
    contradiction).
    """
    if granularity == "document":
        pairs = [(r["doc"][:2000], r["claim"]) for r in rows]
        return _batch_scores(scorer, pairs, batch_size)

    flat: list[tuple[str, str]] = []
    owners: list[int] = []
    for ri, r in enumerate(rows):
        for passage in select_relevant_passages(r["doc"], r["claim"], top_k=top_k):
            flat.append((passage, r["claim"]))
            owners.append(ri)
    flat_scores = _batch_scores(scorer, flat, batch_size)
    per_row = [0.0] * len(rows)
    for owner, s in zip(owners, flat_scores, strict=True):
        if s > per_row[owner]:
            per_row[owner] = s
    return per_row


def run(
    max_samples: int | None = None,
    *,
    batch_size: int = 64,
    device: int = -1,
    granularity: str = "document",
    top_k: int = 5,
) -> dict:
    rows = _load()
    if max_samples is not None:
        rows = rows[:max_samples]
    scorer = ContradictionScorer.from_pretrained(device=device)

    scores = _score_rows(
        scorer, rows, granularity=granularity, top_k=top_k, batch_size=batch_size
    )

    by_ds_sup: dict[str, list[float]] = defaultdict(list)
    by_ds_ins: dict[str, list[float]] = defaultdict(list)
    all_sup: list[float] = []
    all_ins: list[float] = []
    for r, s in zip(rows, scores, strict=True):
        ds = r.get("dataset", "unknown")
        if int(r["label"]) == 1:
            by_ds_sup[ds].append(s)
            all_sup.append(s)
        else:
            by_ds_ins[ds].append(s)
            all_ins.append(s)

    per_dataset = {
        ds: _metrics(by_ds_ins.get(ds, []), by_ds_sup.get(ds, []))
        for ds in sorted(set(by_ds_sup) | set(by_ds_ins))
    }
    return {
        "benchmark": "contradiction_aggrefact",
        "model": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        "device": _device_label(device),
        "n_total": len(rows),
        "granularity": granularity,
        "top_k": top_k if granularity == "passage" else None,
        "signal": "P(contradiction); label 1=supported (false-halt), 0=unsupported (recall lower bound)",
        "overall": _metrics(all_ins, all_sup),
        "per_dataset": per_dataset,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("max_samples", nargs="?", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="CUDA device index for GPU inference; -1 (default) runs on CPU.",
    )
    parser.add_argument(
        "--granularity",
        choices=("document", "passage"),
        default="document",
        help="Grounding granularity: whole truncated document, or top-k "
        "relevant passages scored individually (production-faithful).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Passages scored per claim when --granularity passage.",
    )
    args = parser.parse_args()
    result = run(
        args.max_samples,
        batch_size=args.batch_size,
        device=args.device,
        granularity=args.granularity,
        top_k=args.top_k,
    )
    save_results(result, f"contradiction_aggrefact_{args.granularity}.json")
    o = result["overall"]
    print(
        f"\nAggreFact contradiction signal "
        f"(n={result['n_total']}, {result['granularity']}, {result['device']}):"
    )
    print(f"  AUC: {o['auc_contra_vs_inconsistent']}")
    for thr, m in o["thresholds"].items():
        print(
            f"  thr={thr}: false-halt(supported)={m['false_halt_rate']:.1%} "
            f"| recall(unsupported)={m['recall']:.1%}"
        )


if __name__ == "__main__":
    main()
