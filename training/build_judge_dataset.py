# SPDX-License-Identifier: Apache-2.0
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
import json
import logging
import os
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, cast

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

Example = dict[str, Any]
ShardArgs = tuple[str, int, int]


def remap_labels(dataset: Dataset) -> Dataset:
    """Map 3-class NLI labels to the binary judge label scheme.

    Parameters
    ----------
    dataset:
        Dataset containing a numeric ``label`` column with NLI labels
        ``0`` (entailment), ``1`` (neutral), and ``2`` (contradiction).

    Returns
    -------
    Dataset
        Dataset with ``label`` rewritten to ``0`` for approve and ``1`` for
        reject.
    """

    def _remap(example: Example) -> Example:
        example["label"] = LABEL_APPROVE if example["label"] == 0 else LABEL_REJECT
        return example

    return dataset.map(_remap, desc="Remapping labels to binary")


def stratified_subsample(dataset: Dataset, n: int, seed: int = 42) -> Dataset:
    """Select up to ``n`` examples while preserving label proportions.

    Parameters
    ----------
    dataset:
        Binary-labelled dataset with a numeric ``label`` column.
    n:
        Maximum number of rows to retain.
    seed:
        Seed for deterministic index sampling.

    Returns
    -------
    Dataset
        Subsampled dataset in shuffled order.
    """
    rng = np.random.default_rng(seed)
    labels = np.array(dataset["label"])
    if len(labels) == 0:
        return dataset.select([])

    indices: list[int] = []
    for lab in [LABEL_APPROVE, LABEL_REJECT]:
        mask = np.where(labels == lab)[0]
        target = int(n * (mask.shape[0] / len(labels)))
        k = min(len(mask), target)
        if k > 0:
            chosen = rng.choice(mask, size=k, replace=False)
            indices.extend(int(index) for index in chosen.tolist())
    rng.shuffle(indices)
    return dataset.select(indices[:n])


def score_with_nli(
    dataset: Dataset, use_onnx: bool = False, batch_size: int = 16
) -> Dataset:
    """Score samples with the configured NLI backend.

    Parameters
    ----------
    dataset:
        Dataset containing ``premise`` and ``hypothesis`` columns.
    use_onnx:
        Route scoring through the ONNX backend when true; otherwise use the
        PyTorch backend.
    batch_size:
        Batch size for ONNX scoring.

    Returns
    -------
    Dataset
        Dataset with a numeric ``nli_divergence`` column.
    """
    if use_onnx:
        return _score_onnx(dataset, batch_size)
    return _score_pytorch(dataset)


def _score_pytorch(dataset: Dataset) -> Dataset:
    """Score via PyTorch NLI model (sequential)."""
    from director_ai.core.scoring.nli import NLIScorer

    scorer = NLIScorer(use_model=True, backend="deberta")
    divergences: list[float] = []
    total = len(dataset)
    t0 = time.monotonic()

    for i, row in enumerate(dataset):
        sample = cast(Mapping[str, object], row)
        d = scorer.score(str(sample["premise"]), str(sample["hypothesis"]))
        divergences.append(round(d, 4))
        if (i + 1) % 1000 == 0:
            elapsed = time.monotonic() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate
            logger.info("Scored %d/%d (%.1f/s, ETA %.0fs)", i + 1, total, rate, eta)

    return dataset.add_column("nli_divergence", divergences)


def _score_onnx(dataset: Dataset, batch_size: int = 16) -> Dataset:
    """Score via ONNX NLI model (batched)."""
    from director_ai.core.scoring.nli import NLIScorer

    scorer = NLIScorer(use_model=True, backend="onnx")
    premises = [str(value) for value in dataset["premise"]]
    hypotheses = [str(value) for value in dataset["hypothesis"]]
    divergences: list[float] = []
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


def _score_gpu_shard(args_tuple: ShardArgs) -> str:
    """Score a shard on a specific GPU (used by multi-GPU scoring)."""
    shard_path, gpu_id, shard_id = args_tuple
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from director_ai.core.scoring.nli import NLIScorer

    shard = load_from_disk(shard_path)
    scorer = NLIScorer(use_model=True, backend="deberta")
    divergences: list[float] = []
    total = len(shard)
    t0 = time.monotonic()

    for i, row in enumerate(shard):
        sample = cast(Mapping[str, object], row)
        d = scorer.score(str(sample["premise"]), str(sample["hypothesis"]))
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
    """Score samples across multiple GPU-bound worker processes.

    Parameters
    ----------
    dataset:
        Dataset containing ``premise`` and ``hypothesis`` columns.
    num_gpus:
        Number of worker processes and visible GPU IDs to use.
    gpu_offset:
        First physical GPU index assigned to worker zero.

    Returns
    -------
    Dataset
        Dataset with scored shards concatenated in shard order.
    """
    if num_gpus <= 1:
        return _score_pytorch(dataset)

    # Use caller-controlled scratch storage on the working disk when available.
    tmpdir = Path(os.environ.get("TMPDIR", "/tmp")) / f"judge_shards_{os.getpid()}"
    tmpdir.mkdir(parents=True, exist_ok=True)
    shards: list[ShardArgs] = []
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


def format_judge_input(example: Example) -> Example:
    """Create the text field consumed by the binary judge trainer.

    Parameters
    ----------
    example:
        Row mapping with ``nli_divergence``, ``premise``, and ``hypothesis``.

    Returns
    -------
    dict[str, Any]
        The same mapping with ``text`` set to a three-line judge input.
    """
    divergence = float(example["nli_divergence"])
    premise = str(example["premise"])
    hypothesis = str(example["hypothesis"])
    example["text"] = (
        f"NLI divergence: {divergence:.2f}\n"
        f"Context: {premise[:400]}\n"
        f"Response: {hypothesis[:400]}"
    )
    return example


def apply_precomputed_divergence(dataset: Dataset, column: str) -> Dataset:
    """Use an existing divergence column instead of running model inference.

    Parameters
    ----------
    dataset:
        Dataset containing the premise, hypothesis, label, and divergence
        columns.
    column:
        Name of the numeric column to expose as ``nli_divergence``.

    Returns
    -------
    Dataset
        Dataset with a normalised ``nli_divergence`` column.

    Raises
    ------
    ValueError
        If the requested column is missing or contains non-numeric values.
    """
    column_names = list(dataset.column_names or [])
    if column not in column_names:
        available = ", ".join(column_names) if column_names else "<none>"
        raise ValueError(
            f"precomputed divergence column {column!r} is missing; "
            f"available columns: {available}"
        )

    try:
        divergences = [round(float(value), 4) for value in dataset[column]]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"precomputed divergence column {column!r} must contain numeric values"
        ) from exc

    if "nli_divergence" in column_names:
        dataset = dataset.remove_columns(["nli_divergence"])
    return dataset.add_column("nli_divergence", divergences)


def filter_and_balance(
    dataset: Dataset,
    borderline_keep: int = 25000,
    confident_keep: int = 10000,
    seed: int = 42,
) -> Dataset:
    """Keep borderline samples plus a bounded confident calibration set.

    Parameters
    ----------
    dataset:
        Dataset containing a numeric ``nli_divergence`` column.
    borderline_keep:
        Maximum rows from the inclusive ``0.2`` to ``0.8`` divergence band.
        ``0`` keeps every borderline row.
    confident_keep:
        Maximum rows outside the borderline band. ``0`` keeps every confident
        row.
    seed:
        Seed for deterministic zone sampling and final shuffling.

    Returns
    -------
    Dataset
        Filtered dataset containing the retained borderline and confident rows.
    """
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


def build_judge_dataset(
    *,
    input_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
    subsample: int = 50000,
    borderline_keep: int = 25000,
    confident_keep: int = 10000,
    use_onnx: bool = False,
    batch_size: int = 16,
    eval_ratio: float = 0.1,
    seed: int = 42,
    num_gpus: int = 1,
    gpu_offset: int = 0,
    precomputed_divergence_column: str | None = None,
) -> DatasetDict:
    """Build, split, and persist the binary judge training dataset.

    Parameters
    ----------
    input_dir:
        On-disk Hugging Face ``DatasetDict`` containing a ``train`` split.
    output_dir:
        Directory where the judge ``DatasetDict`` and ``stats.json`` are saved.
    subsample:
        Maximum number of samples to score before filtering. ``0`` keeps all
        input samples.
    borderline_keep:
        Maximum borderline samples after scoring. ``0`` keeps all borderline
        samples.
    confident_keep:
        Maximum confident samples after scoring. ``0`` keeps all confident
        samples.
    use_onnx:
        Use the ONNX NLI backend for scoring.
    batch_size:
        ONNX batch size.
    eval_ratio:
        Fraction assigned to the eval split.
    seed:
        Random seed for sampling and splitting.
    num_gpus:
        Number of GPUs for PyTorch multi-process scoring.
    gpu_offset:
        First GPU index used by multi-process scoring.
    precomputed_divergence_column:
        Optional numeric column to use as ``nli_divergence`` instead of running
        NLI model inference.

    Returns
    -------
    DatasetDict
        Persisted ``train`` and ``eval`` splits.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"{input_dir} not found — run data_pipeline.py first")
    logger.info("Loading existing 3-class dataset from %s", input_dir)
    loaded = load_from_disk(str(input_dir))
    if not isinstance(loaded, DatasetDict):
        raise TypeError(f"{input_dir} must contain a Hugging Face DatasetDict")
    if "train" not in loaded:
        raise ValueError(f"{input_dir} must contain a 'train' split")

    train_ds = loaded["train"]
    logger.info("Loaded %d training samples", len(train_ds))

    logger.info("Remapping to binary labels (approve/reject)")
    train_ds = remap_labels(train_ds)
    labels = np.array(train_ds["label"])
    logger.info(
        "Binary distribution: approve=%d, reject=%d",
        int((labels == 0).sum()),
        int((labels == 1).sum()),
    )

    if subsample > 0:
        logger.info("Stratified subsample → %d", subsample)
        sub = stratified_subsample(train_ds, subsample, seed=seed)
    else:
        logger.info("Using ALL %d samples (no subsample limit)", len(train_ds))
        sub = train_ds

    if precomputed_divergence_column is None:
        logger.info("Running NLI scoring on %d samples...", len(sub))
        t0 = time.monotonic()
        if num_gpus > 1 and not use_onnx:
            sub = score_with_nli_multigpu(
                sub,
                num_gpus=num_gpus,
                gpu_offset=gpu_offset,
            )
        else:
            sub = score_with_nli(sub, use_onnx=use_onnx, batch_size=batch_size)
        logger.info("NLI scoring done in %.1fs", time.monotonic() - t0)
    else:
        logger.info(
            "Using precomputed NLI divergence column %s",
            precomputed_divergence_column,
        )
        sub = apply_precomputed_divergence(sub, precomputed_divergence_column)

    logger.info("Filtering to borderline + confident samples")
    filtered = filter_and_balance(
        sub,
        borderline_keep=borderline_keep,
        confident_keep=confident_keep,
        seed=seed,
    )

    logger.info("Formatting judge input text")
    filtered = filtered.map(format_judge_input, desc="Formatting inputs")

    split = filtered.train_test_split(test_size=eval_ratio, seed=seed)
    final = DatasetDict({"train": split["train"], "eval": split["test"]})

    output_dir.mkdir(parents=True, exist_ok=True)
    final.save_to_disk(str(output_dir))

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

    (output_dir / "stats.json").write_text(
        json.dumps(stats, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Saved judge dataset to %s: %s", output_dir, stats)
    return final


def main(argv: Sequence[str] | None = None) -> None:
    """Parse CLI arguments and build the judge dataset."""
    parser = argparse.ArgumentParser(description="Build binary judge dataset")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DATA_DIR,
        help="Input DatasetDict directory (default: training/data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output DatasetDict directory (default: training/data_judge)",
    )
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
        "--precomputed-divergence-column",
        default=None,
        help=(
            "Use this existing numeric column as nli_divergence instead of "
            "running NLI model inference"
        ),
    )
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
    args = parser.parse_args(argv)

    build_judge_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        subsample=args.subsample,
        borderline_keep=args.borderline_keep,
        confident_keep=args.confident_keep,
        use_onnx=args.use_onnx,
        batch_size=args.batch_size,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        num_gpus=args.num_gpus,
        gpu_offset=args.gpu_offset,
        precomputed_divergence_column=args.precomputed_divergence_column,
    )


if __name__ == "__main__":
    main()
