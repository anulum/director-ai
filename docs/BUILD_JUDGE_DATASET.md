# Director-AI Judge Dataset Builder

> **Module**: `training/build_judge_dataset.py` | **Version**: 3.12.0 | **License**: GNU AGPL v3
>
> © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
> © Code 2020–2026 Miroslav Šotek. All rights reserved.

---

## Overview

The judge dataset builder transforms the 3-class NLI training dataset
(produced by `data_pipeline.py`) into a binary classification dataset
optimised for the local DeBERTa-v3-base judge model. The judge model learns
to make approve/reject decisions on borderline cases where the NLI scorer's
divergence falls in the uncertain zone (0.2–0.8).

The pipeline has five stages:

1. **Load** the 3-class dataset from `training/data/`
2. **Remap** labels to binary (approve/reject)
3. **Subsample** with stratified balance (optional)
4. **Score** every sample with the FactCG NLI model to obtain divergence values
5. **Filter** to keep borderline samples plus a controlled number of confident ones
6. **Format** input text with the NLI divergence prepended
7. **Split** into train/eval and save to `training/data_judge/`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  build_judge_dataset.py                         │
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌───────────────┐   │
│  │ training/data │────▶│ remap_labels │────▶│  subsample    │   │
│  │ (3-class NLI) │     │ (binary)     │     │ (stratified)  │   │
│  └──────────────┘     └──────────────┘     └───────┬───────┘   │
│                                                     │           │
│                       ┌─────────────────────────────▼────────┐  │
│                       │         NLI Scoring                  │  │
│                       │                                      │  │
│                       │  ┌─────────┐  ┌─────────────────┐    │  │
│                       │  │ Single  │  │   Multi-GPU      │   │  │
│                       │  │  GPU    │  │ ProcessPool      │   │  │
│                       │  │ PyTorch │  │  ┌───┐ ┌───┐    │   │  │
│                       │  │         │  │  │GPU│ │GPU│ .. │   │  │
│                       │  │         │  │  │ 1 │ │ 2 │    │   │  │
│                       │  └─────────┘  │  └───┘ └───┘    │   │  │
│                       │     or        └─────────────────┘    │  │
│                       │  ┌─────────┐                         │  │
│                       │  │  ONNX   │                         │  │
│                       │  │ batched │                         │  │
│                       │  └─────────┘                         │  │
│                       └─────────────────────────┬────────────┘  │
│                                                 │               │
│                       ┌─────────────────────────▼────────────┐  │
│                       │      filter_and_balance              │  │
│                       │  borderline (0.2-0.8): keep N        │  │
│                       │  confident (<0.2 or >0.8): keep M    │  │
│                       └─────────────────────────┬────────────┘  │
│                       ┌─────────────────────────▼────────────┐  │
│                       │      format_judge_input              │  │
│                       │  "NLI divergence: 0.45\n..."         │  │
│                       └─────────────────────────┬────────────┘  │
│                       ┌─────────────────────────▼────────────┐  │
│                       │  Train/eval split → data_judge/      │  │
│                       └──────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Binary Label Scheme

The 3-class NLI labels are collapsed to binary:

| NLI Label      | ID | Judge Label | ID | Rationale                        |
|----------------|----|-------------|----|----------------------------------|
| Entailment     | 0  | Approve     | 0  | Factually supported → pass       |
| Neutral        | 1  | Reject      | 1  | Insufficient evidence → halt     |
| Contradiction  | 2  | Reject      | 1  | Factual conflict → halt          |

The conservative mapping (neutral → reject) reflects Director-AI's
safety-first design: when the NLI model cannot determine entailment, the
guardrail should err on the side of rejection.

---

## Core Functions

### `remap_labels(dataset: Dataset) -> Dataset`

Maps 3-class labels to binary using `dataset.map()`. The mapping function
at line 58 checks `example["label"] == 0` for approve, everything else is
reject. Returns a new dataset with the `label` column overwritten.

### `stratified_subsample(dataset: Dataset, n: int, seed: int = 42) -> Dataset`

Reduces the dataset to `n` samples while maintaining the original
approve/reject ratio. Uses `numpy.random.default_rng(seed)` for
reproducibility. The algorithm:

1. Compute per-label proportion: `mask.shape[0] / len(labels)`
2. Sample `k = min(count, n × proportion)` indices per label
3. Concatenate and shuffle indices
4. Select first `n` indices

When `--subsample 0` is passed, this step is skipped entirely (line 309),
using all available samples.

### `score_with_nli(dataset, use_onnx, batch_size) -> Dataset`

Dispatcher that routes to `_score_pytorch()` or `_score_onnx()` based on
the `--use-onnx` flag. Returns the dataset with an added `nli_divergence`
column (float, 4 decimal places).

### `_score_pytorch(dataset: Dataset) -> Dataset`

Sequential scoring using `NLIScorer(backend="deberta")`. Processes one
sample at a time. Logs progress every 1,000 samples with rate (samples/s)
and ETA (seconds).

**Performance**: ~25 samples/s on a single RX 6600 XT (8 GB VRAM) with
FP32. ~45 samples/s with FP16. For 446K samples at 25/s: ~5 hours.

### `_score_onnx(dataset, batch_size) -> Dataset`

Batched scoring using `NLIScorer(backend="onnx")`. Processes `batch_size`
samples per forward pass (default 16). Uses `zip(strict=True)` for
premise/hypothesis pairing safety.

**Performance**: ~80 samples/s on CPU, ~200 samples/s on GPU (ONNX Runtime
with CUDA EP). Faster than PyTorch for inference-only workloads.

### `_score_gpu_shard(args_tuple) -> str`

Per-GPU worker function for multi-GPU scoring. Receives a tuple of
`(shard_path, gpu_id, shard_id)`. Sets `CUDA_VISIBLE_DEVICES` before
importing any CUDA-dependent module to ensure process-level GPU isolation.

The function:
1. Sets `os.environ["CUDA_VISIBLE_DEVICES"]` to the assigned GPU ID
2. Imports `NLIScorer` (triggers CUDA initialisation on the assigned device)
3. Loads the shard from disk
4. Scores sequentially within the shard
5. Saves the scored shard to `{shard_path}_scored`
6. Returns the output path

### `score_with_nli_multigpu(dataset, num_gpus, gpu_offset) -> Dataset`

Orchestrates parallel NLI scoring across multiple GPUs using
`concurrent.futures.ProcessPoolExecutor`. The algorithm:

1. Create a temporary directory on NTFS (`TMPDIR` env var) to avoid
   filling the root partition
2. Split the dataset into `num_gpus` shards of equal size (last shard
   gets remainder)
3. Save each shard to disk (required for cross-process transfer)
4. Launch `num_gpus` worker processes via ProcessPoolExecutor
5. Each worker calls `_score_gpu_shard()` with its assigned GPU
6. Collect scored shards and concatenate via `concatenate_datasets()`

**GPU isolation**: each worker process sets `CUDA_VISIBLE_DEVICES` before
importing PyTorch, ensuring that each process sees only its assigned GPU
as device 0. This avoids CUDA context conflicts.

**TMPDIR**: shards are stored under `Path(os.environ.get("TMPDIR", "/tmp"))`
to support systems where the root partition is too small for temporary data
(e.g. 93 GB root disk with most storage on NTFS).

**Performance**: near-linear scaling. 3 GPUs (RX 6600 XT) at ~25/s each
= ~75/s total. 446K samples in ~1.5 hours instead of ~5 hours.

### `filter_and_balance(dataset, borderline_keep, confident_keep, seed) -> Dataset`

Separates samples into two zones based on NLI divergence:

| Zone       | Divergence range | Purpose                              |
|------------|------------------|--------------------------------------|
| Borderline | 0.2 – 0.8       | Cases where the judge adds value     |
| Confident  | < 0.2 or > 0.8  | Easy cases for calibration anchoring |

The function subsamples each zone independently:

- `borderline_keep` (default 25,000): max borderline samples to retain.
  Set to 0 to keep all borderline samples.
- `confident_keep` (default 10,000): max confident samples to retain.
  Set to 0 to keep all confident samples.

Indices are shuffled with `numpy.random.default_rng(seed)` before selection.

### `format_judge_input(example) -> example`

Formats the input text that the judge model will see during training and
inference. The format prepends the NLI divergence score as a structured
feature:

```
NLI divergence: 0.45
Context: The Earth orbits the Sun at an average distance...
Response: The Earth revolves around the Sun at about 150...
```

Both context and response are truncated to 400 characters to fit within
the judge model's 384-token input limit.

---

## CLI Usage

### Default (50K subsample, 25K borderline, 10K confident)

```bash
python training/build_judge_dataset.py
```

### Full dataset (no limits)

```bash
python training/build_judge_dataset.py --subsample 0 --borderline-keep 0 --confident-keep 0
```

### Multi-GPU scoring (GPUs 1-3, skip GPU 0)

```bash
python training/build_judge_dataset.py --subsample 0 --borderline-keep 0 \
    --num-gpus 3 --gpu-offset 1
```

### ONNX scoring (faster on CPU)

```bash
python training/build_judge_dataset.py --use-onnx --batch-size 32
```

### CLI Flags

| Flag               | Default | Description                                     |
|--------------------|---------|-------------------------------------------------|
| `--subsample`      | 50000   | Max samples before NLI scoring (0 = all)        |
| `--borderline-keep`| 25000   | Max borderline samples after scoring (0 = all)  |
| `--confident-keep` | 10000   | Max confident samples after scoring (0 = all)   |
| `--use-onnx`       | off     | Use ONNX backend for scoring                    |
| `--batch-size`     | 16      | Batch size for ONNX scoring                     |
| `--eval-ratio`     | 0.1     | Eval split proportion                           |
| `--seed`           | 42      | Random seed for reproducibility                 |
| `--num-gpus`       | 1       | Number of GPUs for parallel scoring              |
| `--gpu-offset`     | 0       | First GPU index (skip lower-indexed GPUs)        |

---

## Output Format

### Directory structure

```
training/data_judge/
├── train/
│   ├── data-00000-of-00001.arrow
│   └── state.json
├── eval/
│   ├── data-00000-of-00001.arrow
│   └── state.json
├── dataset_dict.json
└── stats.json
```

### Schema

| Field            | Type   | Description                                  |
|------------------|--------|----------------------------------------------|
| `premise`        | string | Source context text                           |
| `hypothesis`     | string | Response to verify                           |
| `label`          | int    | 0 = approve, 1 = reject                     |
| `source`         | string | Provenance (e.g. `fever`, `halueval_qa`)     |
| `nli_divergence` | float  | FactCG NLI divergence score (0.0–1.0)        |
| `text`           | string | Formatted judge input (divergence + context)  |

### Statistics file (`stats.json`)

```json
{
  "total": 446067,
  "train": 401460,
  "eval": 44607,
  "train_approve": 200730,
  "train_reject": 200730,
  "eval_approve": 22304,
  "eval_reject": 22303
}
```

---

## Borderline Zone Strategy

The judge model exists because the NLI scorer alone produces uncertain
results in the divergence range 0.2–0.8. In this zone:

- **0.2–0.4**: NLI leans towards entailment but is not confident.
  Many genuine paraphrases and stylistic variations land here.
- **0.4–0.6**: Maximum uncertainty. The NLI model cannot reliably
  distinguish entailment from contradiction.
- **0.6–0.8**: NLI leans towards contradiction but may be wrong.
  Complex multi-sentence claims often produce false positives here.

Outside the borderline zone:

- **< 0.2**: the NLI model is confident the response entails the context.
  The judge model should learn to approve these quickly.
- **> 0.8**: the NLI model is confident the response contradicts the context.
  The judge model should learn to reject these quickly.

Including a controlled number of confident samples (default 10K) provides
calibration anchors — the judge learns what "easy approve" and "easy reject"
look like, preventing drift in its decision boundary.

---

## Multi-GPU Architecture

The multi-GPU scoring uses process-level parallelism, not thread-level,
because:

1. **GIL**: Python's GIL prevents true parallel execution in threads.
   PyTorch releases the GIL during CUDA kernels, but tokenisation and
   pre/post-processing hold it.
2. **CUDA contexts**: each GPU requires its own CUDA context. Sharing
   contexts across threads is fragile and leads to synchronisation bugs.
3. **Memory isolation**: each process has its own memory space, preventing
   accidental cross-contamination of model weights or intermediate tensors.

The trade-off is higher startup cost (each process loads the NLI model
independently, ~2 GB VRAM per GPU) and the need to serialise/deserialise
dataset shards to disk. For large datasets (>100K samples), the
parallelism benefit far outweighs this overhead.

### TMPDIR Requirement

The multi-GPU pipeline creates temporary dataset shards on disk. On systems
where the root partition is small (e.g. 93 GB with most storage on a
separate NTFS volume), the `TMPDIR` environment variable must point to
a partition with sufficient space:

```bash
export TMPDIR=/media/anulum/724AA8E84AA8AA75/linux_data/tmp
python training/build_judge_dataset.py --num-gpus 3 --gpu-offset 1
```

Without this, the default `/tmp` on the root partition may fill up during
scoring, causing SIGKILL (OOM killer) or ENOSPC errors.

---

## Downstream Consumer

The judge dataset is consumed by `training/train_judge.py`, which fine-tunes
a DeBERTa-v3-base model (184M parameters) as a binary classifier. The
trained model is saved to `training/output/deberta-v3-base-judge/` and
loaded by `CoherenceScorer(scorer_backend="hybrid")` at inference time.

---

## Dependencies

| Package     | Version | Purpose                                   |
|-------------|---------|-------------------------------------------|
| `datasets`  | ≥2.14   | Dataset loading, Arrow serialisation      |
| `numpy`     | ≥1.24   | Random sampling, array operations         |
| `torch`     | ≥2.0    | NLI model inference (PyTorch backend)     |
| `transformers` | ≥4.30 | NLI model loading                       |
| `onnxruntime` | ≥1.15 | ONNX inference (optional, --use-onnx)   |

---

## Error Handling

- **Missing input dataset**: raises `FileNotFoundError` with guidance to
  run `data_pipeline.py` first (line 289).
- **GPU out of memory**: individual GPU shards will fail with CUDA OOM.
  Reduce shard size by increasing `--num-gpus` or reducing `--subsample`.
- **TMPDIR full**: SIGKILL or ENOSPC. Set `TMPDIR` to a larger partition.

---

## Testing

Covered by `tests/test_build_judge_dataset.py` (38 tests):

- `remap_labels()` correctness (0→0, 1→1, 2→1)
- `stratified_subsample()` balance preservation
- `filter_and_balance()` zone separation
- `format_judge_input()` text formatting
- Edge cases: empty dataset, single-class dataset, subsample > dataset size

Run:

```bash
pytest tests/test_build_judge_dataset.py -v
```

---

## File Reference

| Item                     | Path                                  |
|--------------------------|---------------------------------------|
| Builder module           | `training/build_judge_dataset.py`     |
| Input dataset            | `training/data/`                      |
| Output dataset           | `training/data_judge/`                |
| Statistics               | `training/data_judge/stats.json`      |
| Tests                    | `tests/test_build_judge_dataset.py`   |
| Upstream: data pipeline  | `training/data_pipeline.py`           |
| Downstream: judge trainer | `training/train_judge.py`            |
| Downstream: scorer       | `src/director_ai/core/scoring/scorer.py` |
