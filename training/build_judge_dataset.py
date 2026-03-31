# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Build Binary Judge Dataset
"""
Build a binary (approve/reject) dataset for the local judge classifier.

Takes the existing 3-class NLI dataset (training/data/), remaps labels
to binary, runs FactCG NLI scoring to get divergence scores, filters to
borderline zone (0.2-0.8), and saves as training/data_judge/.

The judge model learns to make approve/reject decisions on cases where
the NLI scorer is uncertain (borderline divergence). The NLI divergence
is prepended to the input text so the judge can leverage it as a feature.

Usage::

    python training/build_judge_dataset.py
    python training/build_judge_dataset.py --subsample 0  # all samples (no limit)
    python training/build_judge_dataset.py --subsample 0 --borderline-keep 0  # keep all borderline
    python training/build_judge_dataset.py --use-onnx  # faster NLI scoring
    python training/build_judge_dataset.py --num-gpus 3 --gpu-offset 1  # multi-GPU
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "data_judge"

# Binary labels
LABEL_APPROVE = 0  # entailment → factual
LABEL_REJECT = 1  # neutral/contradiction → hallucinated

# Borderline zone boundaries (NLI divergence)
BORDERLINE_LOW = 0.2
BORDERLINE_HIGH = 0.8


def remap_labels(dataset: Dataset) -> Dataset:
    """Remap 3-class NLI labels to binary: 0→0 (approve), 1→1 (reject), 2→1 (reject)."""

    def _remap(example):
        example["label"] = LABEL_APPROVE if example["label"] == 0 else LABEL_REJECT
        return example

    return dataset.map(_remap, desc="Remapping labels to binary")


def stratified_subsample(dataset: Dataset, n: int, seed: int = 42) -> Dataset:
    """Stratified subsample maintaining label balance."""
    rng = np.random.default_rng(seed)
    labels = np.array(dataset["label"])
    indices = []
    for lab in [LABEL_APPROVE, LABEL_REJECT]:
        mask = np.where(labels == lab)[0]
        k = min(len(mask), n * (mask.shape[0] / len(labels)))
        chosen = rng.choice(mask, size=int(k), replace=False)
        indices.extend(chosen.tolist())
    rng.shuffle(indices)
    return dataset.select(indices[:n])


def score_with_nli(
    dataset: Dataset, use_onnx: bool = False, batch_size: int = 16
) -> Dataset:
    """Run FactCG NLI on each sample, add 'nli_divergence' column."""
    if use_onnx:
        return _score_onnx(dataset, batch_size)
    return _score_pytorch(dataset)


def _score_pytorch(dataset: Dataset) -> Dataset:
    """Score via PyTorch NLI model (sequential)."""
    from director_ai.core.nli import NLIScorer

    scorer = NLIScorer(use_model=True, backend="deberta")
    divergences = []
    total = len(dataset)
    t0 = time.monotonic()

    for i, row in enumerate(dataset):
        d = scorer.score(row["premise"], row["hypothesis"])
        divergences.append(round(d, 4))
        if (i + 1) % 1000 == 0:
            elapsed = time.monotonic() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate
            logger.info("Scored %d/%d (%.1f/s, ETA %.0fs)", i + 1, total, rate, eta)

    return dataset.add_column("nli_divergence", divergences)


def _score_onnx(dataset: Dataset, batch_size: int = 16) -> Dataset:
    """Score via ONNX NLI model (batched)."""
    from director_ai.core.nli import NLIScorer

    scorer = NLIScorer(use_model=True, backend="onnx")
    premises = dataset["premise"]
    hypotheses = dataset["hypothesis"]
    divergences = []
    total = len(dataset)
    t0 = time.monotonic()

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        pairs = list(zip(premises[start:end], hypotheses[start:end], strict=True))
        batch_scores = scorer.score_batch(pairs)
        divergences.extend([round(s, 4) for s in batch_scores])
        if (start + batch_size) % 5000 < batch_size:
            elapsed = time.monotonic() - t0
            done = start + batch_size
            rate = done / elapsed
            eta = (total - done) / rate if rate > 0 else 0
            logger.info(
                "Scored %d/%d (%.1f/s, ETA %.0fs)", min(done, total), total, rate, eta
            )

    return dataset.add_column("nli_divergence", divergences)


def _score_gpu_shard(args_tuple):
    """Score a shard on a specific GPU (used by multi-GPU scoring)."""
    shard_path, gpu_id, shard_id = args_tuple
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from director_ai.core.nli import NLIScorer

    shard = load_from_disk(shard_path)
    scorer = NLIScorer(use_model=True, backend="deberta")
    divergences = []
    total = len(shard)
    t0 = time.monotonic()

    for i, row in enumerate(shard):
        d = scorer.score(row["premise"], row["hypothesis"])
        divergences.append(round(d, 4))
        if (i + 1) % 1000 == 0:
            elapsed = time.monotonic() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate
            logger.info(
                "GPU %d shard %d: %d/%d (%.1f/s, ETA %.0fs)",
                gpu_id,
                shard_id,
                i + 1,
                total,
                rate,
                eta,
            )

    result_ds = shard.add_column("nli_divergence", divergences)
    out_path = f"{shard_path}_scored"
    result_ds.save_to_disk(out_path)
    return out_path


def score_with_nli_multigpu(
    dataset: Dataset,
    num_gpus: int = 1,
    gpu_offset: int = 0,
) -> Dataset:
    """Score using multiple GPUs in parallel via process pool."""
    if num_gpus <= 1:
        return _score_pytorch(dataset)

    # Use NTFS disk for temp shards — root partition too small
    tmpdir = Path(os.environ.get("TMPDIR", "/tmp")) / f"judge_shards_{os.getpid()}"
    tmpdir.mkdir(parents=True, exist_ok=True)
    shards = []
    shard_size = len(dataset) // num_gpus

    for i in range(num_gpus):
        start = i * shard_size
        end = len(dataset) if i == num_gpus - 1 else (i + 1) * shard_size
        shard = dataset.select(range(start, end))
        shard_path = str(tmpdir / f"shard_{i}")
        shard.save_to_disk(shard_path)
        shards.append((shard_path, gpu_offset + i, i))

    logger.info(
        "Scoring %d samples across %d GPUs (GPU %d-%d)",
        len(dataset),
        num_gpus,
        gpu_offset,
        gpu_offset + num_gpus - 1,
    )

    with ProcessPoolExecutor(max_workers=num_gpus) as pool:
        result_paths = list(pool.map(_score_gpu_shard, shards))

    scored_shards = [load_from_disk(p) for p in result_paths]
    return concatenate_datasets(scored_shards)


def format_judge_input(example):
    """Format input text with NLI divergence prepended."""
    example["text"] = (
        f"NLI divergence: {example['nli_divergence']:.2f}\n"
        f"Context: {example['premise'][:400]}\n"
        f"Response: {example['hypothesis'][:400]}"
    )
    return example


def filter_and_balance(
    dataset: Dataset,
    borderline_keep: int = 25000,
    confident_keep: int = 10000,
    seed: int = 42,
) -> Dataset:
    """Keep borderline samples (0.2-0.8) plus some confident ones."""
    rng = np.random.default_rng(seed)
    divs = np.array(dataset["nli_divergence"])

    borderline_mask = (divs >= BORDERLINE_LOW) & (divs <= BORDERLINE_HIGH)
    confident_mask = ~borderline_mask

    borderline_idx = np.where(borderline_mask)[0]
    confident_idx = np.where(confident_mask)[0]

    logger.info(
        "Borderline: %d, Confident: %d", len(borderline_idx), len(confident_idx)
    )

    if borderline_keep > 0 and len(borderline_idx) > borderline_keep:
        borderline_idx = rng.choice(borderline_idx, size=borderline_keep, replace=False)
    if confident_keep > 0 and len(confident_idx) > confident_keep:
        confident_idx = rng.choice(confident_idx, size=confident_keep, replace=False)

    indices = np.concatenate([borderline_idx, confident_idx])
    rng.shuffle(indices)
    return dataset.select(indices.tolist())


def main():
    parser = argparse.ArgumentParser(description="Build binary judge dataset")
    parser.add_argument(
        "--subsample",
        type=int,
        default=50000,
        help="Subsample size (0 = use all samples)",
    )
    parser.add_argument(
        "--borderline-keep",
        type=int,
        default=25000,
        help="Max borderline samples to keep (0 = keep all)",
    )
    parser.add_argument(
        "--confident-keep",
        type=int,
        default=10000,
        help="Max confident samples to keep (0 = keep all)",
    )
    parser.add_argument("--use-onnx", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for parallel NLI scoring",
    )
    parser.add_argument(
        "--gpu-offset",
        type=int,
        default=0,
        help="First GPU index to use (e.g. 1 to skip GPU 0)",
    )
    args = parser.parse_args()

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"{DATA_DIR} not found — run data_pipeline.py first")
    logger.info("Loading existing 3-class dataset from %s", DATA_DIR)
    ds = load_from_disk(str(DATA_DIR))
    train_ds = ds["train"]
    logger.info("Loaded %d training samples", len(train_ds))

    logger.info("Remapping to binary labels (approve/reject)")
    train_ds = remap_labels(train_ds)
    labels = np.array(train_ds["label"])
    logger.info(
        "Binary distribution: approve=%d, reject=%d",
        int((labels == 0).sum()),
        int((labels == 1).sum()),
    )

    if args.subsample > 0:
        logger.info("Stratified subsample → %d", args.subsample)
        sub = stratified_subsample(train_ds, args.subsample, seed=args.seed)
    else:
        logger.info("Using ALL %d samples (no subsample limit)", len(train_ds))
        sub = train_ds

    logger.info("Running NLI scoring on %d samples...", len(sub))
    t0 = time.monotonic()
    if args.num_gpus > 1 and not args.use_onnx:
        sub = score_with_nli_multigpu(
            sub,
            num_gpus=args.num_gpus,
            gpu_offset=args.gpu_offset,
        )
    else:
        sub = score_with_nli(sub, use_onnx=args.use_onnx, batch_size=args.batch_size)
    logger.info("NLI scoring done in %.1fs", time.monotonic() - t0)

    logger.info("Filtering to borderline + confident samples")
    filtered = filter_and_balance(
        sub,
        borderline_keep=args.borderline_keep,
        confident_keep=args.confident_keep,
        seed=args.seed,
    )

    logger.info("Formatting judge input text")
    filtered = filtered.map(format_judge_input, desc="Formatting inputs")

    # Train/eval split
    split = filtered.train_test_split(test_size=args.eval_ratio, seed=args.seed)
    final = DatasetDict({"train": split["train"], "eval": split["test"]})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final.save_to_disk(str(OUTPUT_DIR))

    train_labels = np.array(final["train"]["label"])
    eval_labels = np.array(final["eval"]["label"])
    stats = {
        "total": len(filtered),
        "train": len(final["train"]),
        "eval": len(final["eval"]),
        "train_approve": int((train_labels == 0).sum()),
        "train_reject": int((train_labels == 1).sum()),
        "eval_approve": int((eval_labels == 0).sum()),
        "eval_reject": int((eval_labels == 1).sum()),
    }
    import json

    (OUTPUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    logger.info("Saved judge dataset to %s: %s", OUTPUT_DIR, stats)


if __name__ == "__main__":
    main()
