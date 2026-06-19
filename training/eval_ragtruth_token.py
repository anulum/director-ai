# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — example-level evaluation of the token RAGTruth detector

"""Score the trained token detector at the EXAMPLE level, the LettuceDetect metric.

The trainer optimises a per-token loss with heavy positive weighting (hallucinated
tokens are ~5% of response tokens), which over-predicts at the default 0.5 cut.
The competitive number is example-level: a response is flagged hallucinated when
it contains a hallucinated span. This script runs the model once over the full
2700-row test split, caches each response's per-token hallucination probabilities,
then sweeps the decision rule offline:

  flag(response) = (number of response tokens with P(hallucinated) >= p) >= k

and reports precision / recall / F1 / balanced accuracy at the best (p, k), so the
result is directly comparable to LettuceDetect's example-level F1 of 79.22.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - cache-only diagnostics mode
    torch = None  # type: ignore[assignment]

try:
    from datasets import load_dataset
except ModuleNotFoundError:  # pragma: no cover - cache-only diagnostics mode
    load_dataset = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForTokenClassification, AutoTokenizer
except ModuleNotFoundError:  # pragma: no cover - cache-only diagnostics mode
    AutoModelForTokenClassification = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

try:
    from training.train_ragtruth_token import parse_spans  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - Jarvis flat-file upload mode
    from train_ragtruth_token import parse_spans  # type: ignore[no-redef]  # noqa: E402

MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    "/media/anulum/GOTM/_scratch/director_ragtruth/ragtruth-token-modernbert",
)
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "1024"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
CACHE = os.environ.get(
    "CACHE", "/media/anulum/GOTM/_scratch/director_ragtruth/token_eval_probs.json"
)
RESULT = os.environ.get("RESULT", CACHE.replace(".json", "_result.json"))
TOP_FALSE_POSITIVES = int(os.environ.get("TOP_FALSE_POSITIVES", "25"))
HARD_NEGATIVES = os.environ.get(
    "HARD_NEGATIVES", RESULT.replace(".json", "_hard_negatives.jsonl")
)
DATASET = os.environ.get("DATASET", "wandb/RAGTruth-processed")
DATASET_SPLIT = os.environ.get("DATASET_SPLIT", "test")
DATASET_MAX_ROWS = int(os.environ.get("DATASET_MAX_ROWS", "0"))
DATASET_ROW_OFFSET = int(os.environ.get("DATASET_ROW_OFFSET", "0"))
DATASET_ROW_STRIDE = int(os.environ.get("DATASET_ROW_STRIDE", "1"))
BASE_TOKENIZER = os.environ.get("BASE_TOKENIZER", "answerdotai/ModernBERT-base")


def _as_text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _task_type(ex: dict) -> str:
    for key in ("task_type", "source", "dataset", "split_source"):
        value = ex.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _metadata(
    ex: dict,
    *,
    row_index: int,
    resp_token_count: int | None = None,
    context_token_count: int | None = None,
) -> dict:
    context = _as_text(ex.get("context"))
    response = _as_text(ex.get("output"))
    spans = parse_spans(ex.get("hallucination_labels"))
    metadata = {
        "row_index": row_index,
        "task_type": _task_type(ex),
        "context_chars": len(context),
        "response_chars": len(response),
        "hallucination_span_count": len(spans),
    }
    metadata["response_tokens"] = (
        int(resp_token_count) if resp_token_count is not None else len(response.split())
    )
    metadata["context_tokens"] = (
        int(context_token_count)
        if context_token_count is not None
        else len(context.split())
    )
    return metadata


def row_label(ex: dict) -> int:
    return 1 if parse_spans(ex.get("hallucination_labels")) else 0


DEVICE = os.environ.get(
    "DEVICE",
    "cuda" if torch is not None and torch.cuda.is_available() else "cpu",
)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_indices(row_count: int) -> list[int]:
    """Return deterministic dataset indices for full or subset scoring."""
    if DATASET_ROW_STRIDE < 1:
        raise ValueError("DATASET_ROW_STRIDE must be >= 1")
    if DATASET_ROW_OFFSET < 0:
        raise ValueError("DATASET_ROW_OFFSET must be >= 0")
    indices = list(range(DATASET_ROW_OFFSET, row_count, DATASET_ROW_STRIDE))
    if DATASET_MAX_ROWS > 0:
        indices = indices[:DATASET_MAX_ROWS]
    return indices


def collect_probs() -> list[dict]:
    """Run the model over a dataset split; return per-example token probabilities."""
    if (
        torch is None
        or AutoTokenizer is None
        or AutoModelForTokenClassification is None
    ):
        raise ModuleNotFoundError(
            "torch and transformers are required for model inference; install the "
            "finetune extra or use an existing probability cache"
        )
    if load_dataset is None:
        raise ModuleNotFoundError(
            "datasets is required for model inference; install the finetune extra"
        )
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    except ValueError as exc:
        if "Tokenizer class" not in str(exc):
            raise
        print(
            f"WARNING: local tokenizer config failed; using {BASE_TOKENIZER}",
            file=sys.stderr,
        )
        tok = AutoTokenizer.from_pretrained(BASE_TOKENIZER)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).to(DEVICE).eval()
    ds = load_dataset(DATASET, split=DATASET_SPLIT)
    selected_indices = _selected_indices(len(ds))
    out = []
    start = time.perf_counter()
    for offset in range(0, len(selected_indices), BATCH_SIZE):
        batch_indices = selected_indices[offset : offset + BATCH_SIZE]
        batch = ds.select(batch_indices)
        contexts = [ex["context"] or "" for ex in batch]
        outputs = [ex["output"] or "" for ex in batch]
        labels = [row_label(ex) for ex in batch]
        batch_enc = tok(
            contexts,
            outputs,
            truncation="only_first",
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        enc_dev = {k: v.to(DEVICE) for k, v in batch_enc.items()}
        with torch.no_grad():
            logits = model(**enc_dev).logits
        probs = torch.softmax(logits, dim=-1)[:, :, 1].cpu().numpy()
        for local_idx, label in enumerate(labels):
            seq_ids = batch_enc.sequence_ids(local_idx)
            resp_probs = [
                float(probs[local_idx, token_idx])
                for token_idx, seq_id in enumerate(seq_ids)
                if seq_id == 1
            ]
            context_tokens = sum(1 for seq_id in seq_ids if seq_id == 0)
            row_index = batch_indices[local_idx]
            meta = _metadata(
                batch[local_idx],
                row_index=row_index,
                resp_token_count=len(resp_probs),
                context_token_count=context_tokens,
            )
            out.append({"label": label, "resp_probs": resp_probs, **meta})
        done = min(offset + BATCH_SIZE, len(selected_indices))
        if done % 400 == 0 or done == len(selected_indices):
            elapsed = time.perf_counter() - start
            print(
                f"  scored {done}/{len(selected_indices)} selected rows "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
    Path(CACHE).parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "w") as fh:
        json.dump(out, fh)
    return out


def _enrich_cached_records(records: list[dict]) -> list[dict]:
    """Add dataset metadata to older probability caches without rerunning the model."""
    if not records:
        return records
    if all("task_type" in r and "context_chars" in r for r in records):
        return records
    if load_dataset is None:
        return records
    try:
        ds = load_dataset(DATASET, split=DATASET_SPLIT)
    except Exception as exc:  # pragma: no cover - defensive for offline cache reads
        print(
            f"WARNING: could not enrich cached records from RAGTruth: {exc}",
            file=sys.stderr,
        )
        return records
    enriched = []
    for fallback_index, record in enumerate(records):
        row_index = int(record.get("row_index", fallback_index))
        merged = dict(record)
        if row_index < len(ds):
            meta = _metadata(
                ds[row_index],
                row_index=row_index,
                resp_token_count=len(record.get("resp_probs", [])),
            )
            for key, value in meta.items():
                merged.setdefault(key, value)
        enriched.append(merged)
    return enriched


def _load_test_rows() -> list[dict] | None:
    if load_dataset is None:
        return None
    try:
        return list(load_dataset(DATASET, split=DATASET_SPLIT))
    except Exception as exc:  # pragma: no cover - defensive for offline cache reads
        print(
            f"WARNING: could not load RAGTruth rows for hard negatives: {exc}",
            file=sys.stderr,
        )
        return None


def _confusion_metrics(labels: np.ndarray, flagged: np.ndarray) -> dict:
    tp = int((flagged & (labels == 1)).sum())
    fp = int((flagged & (labels == 0)).sum())
    fn = int((~flagged & (labels == 1)).sum())
    tn = int((~flagged & (labels == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return {
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "balanced_accuracy": (tpr + tnr) / 2,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _bucket(value: int | float | None, bounds: tuple[int, ...]) -> str:
    if value is None:
        return "unknown"
    numeric = int(value)
    lower = 0
    for upper in bounds:
        if numeric <= upper:
            return f"{lower}-{upper}"
        lower = upper + 1
    return f">{bounds[-1]}"


def _group_metrics(
    records: list[dict], labels: np.ndarray, flagged: np.ndarray, group_key
) -> list[dict]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        groups[str(group_key(record))].append(idx)
    rows = []
    for name, indices in sorted(groups.items()):
        idxs = np.array(indices)
        item = _confusion_metrics(labels[idxs], flagged[idxs])
        item.update(
            {
                "group": name,
                "n": int(len(indices)),
                "n_hallucinated": int(labels[idxs].sum()),
                "n_grounded": int((labels[idxs] == 0).sum()),
            }
        )
        rows.append(item)
    return rows


def _diagnostics(records: list[dict], labels: np.ndarray, best: dict) -> dict:
    flagged = np.array(
        [
            sum(1 for x in r["resp_probs"] if x >= best["p"]) >= best["k"]
            for r in records
        ]
    )
    false_positives = []
    for record, label, is_flagged in zip(records, labels, flagged, strict=False):
        if label != 0 or not is_flagged:
            continue
        probs = record.get("resp_probs", [])
        false_positives.append(
            {
                "row_index": int(record.get("row_index", -1)),
                "task_type": record.get("task_type", "unknown"),
                "response_tokens": int(record.get("response_tokens", len(probs))),
                "context_tokens": record.get("context_tokens"),
                "context_chars": int(record.get("context_chars", 0)),
                "response_chars": int(record.get("response_chars", 0)),
                "max_token_probability": max(probs) if probs else 0.0,
                "tokens_at_threshold": sum(1 for x in probs if x >= best["p"]),
            }
        )
    false_positives.sort(
        key=lambda item: (item["tokens_at_threshold"], item["max_token_probability"]),
        reverse=True,
    )
    return {
        "decision_rule": {"p": best["p"], "k": best["k"]},
        "by_task_type": _group_metrics(
            records, labels, flagged, lambda r: r.get("task_type", "unknown")
        ),
        "by_response_token_bucket": _group_metrics(
            records,
            labels,
            flagged,
            lambda r: _bucket(
                r.get("response_tokens", len(r.get("resp_probs", []))),
                (32, 64, 128, 256, 512),
            ),
        ),
        "by_context_token_bucket": _group_metrics(
            records,
            labels,
            flagged,
            lambda r: _bucket(r.get("context_tokens"), (128, 256, 512, 768, 1024)),
        ),
        "by_context_char_bucket": _group_metrics(
            records,
            labels,
            flagged,
            lambda r: _bucket(r.get("context_chars"), (1000, 2000, 4000, 8000, 16000)),
        ),
        "by_hallucination_span_count": _group_metrics(
            records,
            labels,
            flagged,
            lambda r: _bucket(r.get("hallucination_span_count"), (0, 1, 2, 3, 5)),
        ),
        "top_false_positives": false_positives[:TOP_FALSE_POSITIVES],
    }


def _false_positive_records(
    records: list[dict], labels: np.ndarray, best: dict
) -> list[dict]:
    flagged = np.array(
        [
            sum(1 for x in r["resp_probs"] if x >= best["p"]) >= best["k"]
            for r in records
        ]
    )
    false_positives = []
    for record, label, is_flagged in zip(records, labels, flagged, strict=False):
        if label != 0 or not is_flagged:
            continue
        probs = record.get("resp_probs", [])
        sorted_probs = sorted(probs, reverse=True)
        false_positives.append(
            {
                "row_index": int(record.get("row_index", -1)),
                "task_type": record.get("task_type", "unknown"),
                "response_tokens": int(record.get("response_tokens", len(probs))),
                "context_tokens": record.get("context_tokens"),
                "context_chars": int(record.get("context_chars", 0)),
                "response_chars": int(record.get("response_chars", 0)),
                "max_token_probability": sorted_probs[0] if sorted_probs else 0.0,
                "mean_top5_token_probability": (
                    float(np.mean(sorted_probs[:5])) if sorted_probs else 0.0
                ),
                "tokens_at_threshold": sum(1 for x in probs if x >= best["p"]),
            }
        )
    false_positives.sort(
        key=lambda item: (
            item["tokens_at_threshold"],
            item["mean_top5_token_probability"],
            item["max_token_probability"],
        ),
        reverse=True,
    )
    return false_positives


def _write_hard_negatives(
    path: str, records: list[dict], labels: np.ndarray, best: dict
) -> int:
    if not path or path == "0":
        return 0
    false_positives = _false_positive_records(records, labels, best)
    if not false_positives:
        Path(path).write_text("")
        return 0

    rows = _load_test_rows()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for item in false_positives:
            row = (
                rows[item["row_index"]]
                if rows is not None and item["row_index"] >= 0
                else {}
            )
            payload = {
                **item,
                "label": 0,
                "dataset": DATASET,
                "source_split": DATASET_SPLIT,
                "candidate_weight": min(5.0, 1.0 + item["tokens_at_threshold"] / 20.0),
                "decision_rule": {"p": best["p"], "k": best["k"]},
                "context": _as_text(row.get("context")) if row else "",
                "query": _as_text(row.get("query")) if row else "",
                "output": _as_text(row.get("output")) if row else "",
                "hallucination_labels": row.get("hallucination_labels")
                if row
                else None,
            }
            fh.write(json.dumps(payload, sort_keys=True) + "\n")
    return len(false_positives)


def evaluate(records: list[dict]) -> dict:
    records = _enrich_cached_records(records)
    labels = np.array([r["label"] for r in records])
    n_pos, n_neg = int(labels.sum()), int((labels == 0).sum())
    best = None
    grid = []
    for p in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        for k in (1, 2, 3, 5):
            flagged = np.array(
                [sum(1 for x in r["resp_probs"] if x >= p) >= k for r in records]
            )
            item = _confusion_metrics(labels, flagged)
            item.update({"p": p, "k": k})
            grid.append(item)
            if best is None or item["f1"] > best["f1"]:
                best = item
    result = {
        "model_dir": MODEL_DIR,
        "model_sha256": _sha256(Path(MODEL_DIR) / "model.safetensors"),
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "device": DEVICE,
        "dataset": DATASET,
        "dataset_split": DATASET_SPLIT,
        "dataset_row_selection": {
            "max_rows": DATASET_MAX_ROWS,
            "row_offset": DATASET_ROW_OFFSET,
            "row_stride": DATASET_ROW_STRIDE,
        },
        "n": len(labels),
        "n_hallucinated": n_pos,
        "n_grounded": n_neg,
        "best": best,
        "grid": grid,
        "diagnostics": _diagnostics(records, labels, best),
        "hard_negatives": {
            "path": HARD_NEGATIVES
            if HARD_NEGATIVES and HARD_NEGATIVES != "0"
            else None,
            "count": 0,
        },
        "reference": {
            "current_best_example_f1": 0.7628985507246377,
            "lettucedetect_example_f1": 0.7922,
        },
    }
    result["hard_negatives"]["count"] = _write_hard_negatives(
        HARD_NEGATIVES, records, labels, best
    )
    print(
        f"\n=== Example-level RAGTruth (n={len(labels)}, {n_pos} hall / {n_neg} grounded) ==="
    )
    print(f"  BEST F1        : {best['f1']:.4f}")
    print(f"  precision      : {best['precision']:.4f}")
    print(f"  recall         : {best['recall']:.4f}")
    print(f"  balanced acc   : {best['balanced_accuracy']:.4f}")
    print(f"  false-pos rate : {best['fpr']:.4f}")
    print(f"  @ token-prob>={best['p']}  min-tokens={best['k']}")
    print("  (LettuceDetect example-level F1 = 79.22)")
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    return result


if __name__ == "__main__":
    if os.path.exists(CACHE) and "--recompute" not in sys.argv:
        print(f"Loading cached probs from {CACHE}")
        with open(CACHE) as fh:
            recs = json.load(fh)
    else:
        recs = collect_probs()
    evaluate(recs)
