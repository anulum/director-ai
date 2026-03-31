# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — AggreFact Evaluation

"""Evaluate a model on full LLM-AggreFact (29K) from local JSONL.

Supports batched inference and configurable max token length for
comparing models at different context windows.

Usage::

    # Baseline (FactCG-DeBERTa)
    python tools/eval_aggrefact.py baseline

    # Custom model at 512 tokens
    python tools/eval_aggrefact.py modernbert_512 models/modernbert-factcg \\
        --max-length 512 --batch-size 16

    # Long-context evaluation
    python tools/eval_aggrefact.py modernbert_4096 models/modernbert-factcg \\
        --max-length 4096 --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TEMPLATE = (
    "{text_a}\n\nChoose your answer: based on the paragraph above "
    'can we conclude that "{text_b}"?\n\nOPTIONS:\n- Yes\n- No\n'
    "I think the answer is "
)

DATA_SEARCH_PATHS = [
    "data/aggrefact_test.jsonl",
    "benchmarks/aggrefact_test.jsonl",
]


def _find_data_file() -> str:
    for p in DATA_SEARCH_PATHS:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"aggrefact_test.jsonl not found in: {DATA_SEARCH_PATHS}")


def evaluate(
    tag: str,
    model_path: str,
    max_length: int = 512,
    batch_size: int = 32,
) -> dict:
    print(f"Evaluating: {tag} (model_path={model_path!r}, max_length={max_length})")

    if tag == "baseline" or not model_path:
        mn = "yaxili96/FactCG-DeBERTa-v3-Large"
        tokenizer = AutoTokenizer.from_pretrained(mn)
        model = AutoModelForSequenceClassification.from_pretrained(mn).cuda().eval()
    else:
        is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))
        if is_lora:
            from peft import PeftModel

            base = AutoModelForSequenceClassification.from_pretrained(
                "yaxili96/FactCG-DeBERTa-v3-Large"
            )
            model = PeftModel.from_pretrained(base, model_path)
            model = model.merge_and_unload()
            tokenizer = AutoTokenizer.from_pretrained(
                "yaxili96/FactCG-DeBERTa-v3-Large"
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = model.cuda().eval()

    data_file = _find_data_file()
    print(f"Loading AggreFact from {data_file}...")
    rows = []
    with open(data_file) as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"Loaded {len(rows)} samples")

    texts = []
    labels = []
    ds_names = []
    for row in rows:
        doc = row.get("doc", "")
        claim = row.get("claim", "")
        label = row.get("label")
        ds_name = row.get("dataset", "unknown")
        if label is None or not doc or not claim:
            continue
        texts.append(TEMPLATE.format(text_a=doc, text_b=claim))
        labels.append(int(label))
        ds_names.append(ds_name)

    t0 = time.perf_counter()
    all_scores: list[float] = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to("cuda")
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        all_scores.extend(probs[:, 1].cpu().numpy().tolist())
        done = min(i + batch_size, len(texts))
        if done % 5000 < batch_size:
            elapsed = time.perf_counter() - t0
            rate = done / elapsed
            eta = (len(texts) - done) / rate / 60
            print(f"  {done}/{len(texts)} ({rate:.0f}/s, ETA {eta:.0f}m)", flush=True)

    by_dataset: dict[str, list[tuple[int, float]]] = {}
    for score, label, ds_name in zip(all_scores, labels, ds_names, strict=True):
        by_dataset.setdefault(ds_name, []).append((label, score))

    elapsed = time.perf_counter() - t0
    best_thresh, best_avg = 0.5, 0.0
    for t_int in range(10, 91):
        t = t_int / 100.0
        accs = [
            balanced_accuracy_score(
                [p[0] for p in v], [1 if p[1] >= t else 0 for p in v]
            )
            for v in by_dataset.values()
        ]
        avg = float(np.mean(accs))
        if avg > best_avg:
            best_avg, best_thresh = avg, t

    per_ds = {}
    for ds_name in sorted(by_dataset):
        v = by_dataset[ds_name]
        ba = balanced_accuracy_score(
            [p[0] for p in v], [1 if p[1] >= best_thresh else 0 for p in v]
        )
        per_ds[ds_name] = {"ba": round(float(ba), 4), "n": len(v)}

    result = {
        "model": tag,
        "macro_ba": round(float(best_avg), 4),
        "threshold": best_thresh,
        "max_length": max_length,
        "batch_size": batch_size,
        "elapsed": round(elapsed, 1),
        "per_dataset": per_ds,
    }
    os.makedirs("results", exist_ok=True)
    out_path = f"results/eval_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(
        f"{tag}: {best_avg * 100:.2f}% BA (t={best_thresh}) in {elapsed / 60:.1f}m -> {out_path}"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on LLM-AggreFact")
    parser.add_argument("tag", help="Evaluation tag/name")
    parser.add_argument("model_path", nargs="?", default="", help="Path to model")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    evaluate(args.tag, args.model_path, args.max_length, args.batch_size)
