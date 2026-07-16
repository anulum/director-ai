# Director-AI Training Data Pipeline

> **Module**: `training/data_pipeline.py` | **Version**: 3.18.0 | **License**: Apache-2.0
>
> © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
> © Code 2020–2026 Miroslav Šotek. All rights reserved.

---

## Overview

The training data pipeline builds a unified NLI (Natural Language Inference) dataset
from seven heterogeneous sources, normalising all samples to a consistent
`(premise, hypothesis, label)` schema with three-class labels:

| Label ID | Name            | Semantics                              |
|----------|-----------------|----------------------------------------|
| 0        | Entailment      | Hypothesis is factually supported      |
| 1        | Neutral         | Insufficient information to determine  |
| 2        | Contradiction   | Hypothesis contradicts the premise     |

This dataset serves as the foundation for training the FactCG NLI model
(`microsoft/FactCG-DeBERTa-v3-Large`) and, downstream, for building the
binary judge dataset used by the local DeBERTa-v3-base classifier.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   data_pipeline.py                          │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌──────────┐  │
│  │ HaluEval │  │  FEVER   │  │ VitaminC  │  │ ANLI R3  │  │
│  │  ~60K    │  │  ~203K   │  │ ~100K cap │  │  ~100K   │  │
│  └────┬─────┘  └────┬─────┘  └─────┬─────┘  └────┬─────┘  │
│       │              │              │              │        │
│  ┌────┴──────┐  ┌────┴──────┐  ┌────┴──────┐               │
│  │ RAGTruth  │  │  SummaC   │  │ AggreFact │  (optional)   │
│  │  variable │  │  variable │  │   ~29K    │               │
│  └────┬──────┘  └────┬──────┘  └────┬──────┘               │
│       │              │              │                       │
│       └──────────────┴──────────────┘                       │
│                      │                                      │
│              ┌───────▼────────┐                             │
│              │  Unify labels  │                             │
│              │  (0, 1, 2)     │                             │
│              └───────┬────────┘                             │
│              ┌───────▼────────┐                             │
│              │   ClassLabel   │                             │
│              │   cast         │                             │
│              └───────┬────────┘                             │
│              ┌───────▼────────┐                             │
│              │ Stratified     │                             │
│              │ 90/10 split    │                             │
│              └───────┬────────┘                             │
│              ┌───────▼────────┐                             │
│              │ Save to disk   │                             │
│              │ training/data/ │                             │
│              └────────────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Sources

Remote HuggingFace sources are loaded with explicit `revision=` arguments so
rebuilds do not silently drift when upstream repositories change. The pipeline
pins every public source to an immutable commit SHA. `mteb/summac` currently
requires authenticated repository metadata, so its default revision remains
`main`; set `DIRECTOR_AI_SUMMAC_REVISION` to an approved commit SHA in
production environments that use SummaC.

| Source | Revision |
|--------|----------|
| `pminervini/HaluEval` | `12a856119f03975a94509091e8cada3e6be6ead7` |
| `pietrolesci/nli_fever` | `1eddac63112eee1fdf1966e0bca27a5ff248c772` |
| `tals/vitaminc` | `be6febb761b0b2807687e61e0b5282e459df2fa0` |
| `anli` | `8e4813d81f46d313dac7892e1c28076917cfcdf9` |
| `wandb/RAGTruth-processed` | `eb4f4b9d1b68eb7092d3e1a61c0cd82d9808737b` |
| `mteb/summac` | `main` by default; override with `DIRECTOR_AI_SUMMAC_REVISION` |
| `lytang/LLM-AggreFact` | `981dfd0bd8e58e7238a9ab92b2e6ea44bce918e4` |

### 1. HaluEval (~60,000 samples)

**Dataset**: `pminervini/HaluEval` on HuggingFace Hub.

HaluEval provides paired correct and hallucinated outputs for three LLM task
types. Each task contributes both entailment (correct) and contradiction
(hallucinated) samples. No neutral samples are produced.

| Task           | Premise field       | Correct field      | Hallucinated field         |
|----------------|---------------------|--------------------|----------------------------|
| QA             | `knowledge`/`question` | `right_answer`  | `hallucinated_answer`      |
| Dialogue       | `dialogue_history`/`knowledge` | `right_response` | `hallucinated_response` |
| Summarisation  | `document`          | `right_summary`    | `hallucinated_summary`     |

**Label mapping**: correct → 0 (entailment), hallucinated → 2 (contradiction).

**Loader**: `_load_halueval()`.

### 2. FEVER (~203,000 samples)

**Dataset**: `pietrolesci/nli_fever` on HuggingFace Hub.

FEVER (Fact Extraction and VERification) is a large-scale fact verification
benchmark. Claims are paired with Wikipedia evidence sentences. The loader
handles both integer and string label formats for robustness against upstream
schema changes.

**Label mapping**: string labels (`entailment`, `neutral`, `contradiction`)
are mapped via a lookup dictionary. Integer labels (0, 1, 2) are passed
through directly.

**Loader**: `_load_fever()`.

### 3. VitaminC (~100,000 samples, capped)

**Dataset**: `tals/vitaminc` on HuggingFace Hub.

VitaminC is a contrastive fact verification dataset where evidence sentences
are minimally edited to flip the label. This adversarial construction makes
it particularly valuable for training NLI models to detect subtle factual
changes.

**Raw size**: ~370,000 samples. Without capping, VitaminC would dominate the
training distribution at ~50.6% of all samples, introducing a bias towards
its specific adversarial style. The cap at 100,000 (constant `VITAMINC_CAP`)
reduces its share to ~30%, preserving dataset diversity.

**Label mapping**: `SUPPORTS` → 0, `NOT ENOUGH INFO` → 1, `REFUTES` → 2.

**Capping**: deterministic local `random.Random(42)` sampling.

**Loader**: `_load_vitaminc()`.

### 4. ANLI Round 3 (~100,000 samples)

**Dataset**: `anli` (split `train_r3`) on HuggingFace Hub.

ANLI (Adversarial NLI) Round 3 is the hardest split of the adversarial NLI
benchmark. Human annotators crafted hypotheses specifically designed to fool
modern NLI models. Including this split forces the trained model
to handle adversarial edge cases that simpler datasets would not cover.

**Label mapping**: labels are already integers (0, 1, 2) matching the
target schema.

**Loader**: `_load_anli_r3()`.

### 5. RAGTruth (variable, optional)

**Dataset**: `wandb/RAGTruth-processed` on HuggingFace Hub.

RAGTruth provides hallucination annotations for RAG (Retrieval-Augmented
Generation) system outputs. Each sample has a context (retrieved passages)
and a model response, with fine-grained hallucination labels categorised as
`evident_conflict` or `baseless_info`.

**Label mapping**: binary — if either `evident_conflict > 0` or
`baseless_info > 0`, the sample is labelled contradiction (2); otherwise
entailment (0). No neutral samples.

**Text truncation**: both premise and hypothesis are capped at 2,000
characters to prevent outlier-length RAG contexts from dominating batch
construction during training.

**Activation**: `--include-ragtruth` CLI flag or `include_ragtruth=True`.

**Loader**: `_load_ragtruth()`.

### 6. SummaC (variable, optional)

**Dataset**: `mteb/summac` on HuggingFace Hub.

SummaC evaluates summarisation consistency — whether a summary is factually
consistent with its source document. Binary labels: 1 = consistent
(entailment), 0 = inconsistent (contradiction).

**Resilience**: the SummaC dataset has been intermittently unavailable on
HuggingFace Hub. The build path wraps optional SummaC loading in `try/except`
to prevent pipeline failure if the dataset is temporarily offline.

**Text truncation**: premise capped at 2,000 characters.

**Activation**: `--include-summac` CLI flag or `include_summac=True`.

**Loader**: `_load_summac()`.

### 7. LLM-AggreFact (~29,000, optional, gated)

**Dataset**: `lytang/LLM-AggreFact` on HuggingFace Hub.

AggreFact aggregates 11 sub-datasets covering summarisation, RAG, and
grounding tasks. This is a gated dataset — access requires a HuggingFace
token with explicit approval from the dataset maintainers.

**Sub-datasets**: the `dataset` field in each row identifies the source
sub-dataset. The `source` field in the output is set to
`aggrefact_{dataset_name}` for provenance tracking.

**Label mapping**: 1 → entailment (supported), 0 → contradiction (not
supported). No neutral samples.

**Authentication**: requires `HF_TOKEN` environment variable. If not set,
the loader logs a warning and returns an empty list.

**Text truncation**: premise capped at 2,000 characters.

**Activation**: `--include-aggrefact` CLI flag or `include_aggrefact=True`.

**Loader**: `_load_aggrefact()`.

---

## Output Format

### Schema

Each sample in the output dataset has four fields:

| Field        | Type   | Description                                      |
|--------------|--------|--------------------------------------------------|
| `premise`    | string | Source text (evidence, context, document)         |
| `hypothesis` | string | Claim, response, or summary to verify             |
| `label`      | ClassLabel | 0=entailment, 1=neutral, 2=contradiction      |
| `source`     | string | Provenance tag (e.g. `halueval_qa`, `fever`)     |

### Directory structure

```
training/data/
├── train/
│   ├── data-00000-of-00001.arrow
│   └── state.json
├── eval/
│   ├── data-00000-of-00001.arrow
│   └── state.json
├── dataset_dict.json
└── stats.json
```

### Statistics file (`stats.json`)

Written alongside the dataset. Contains:

```json
{
  "total": 495631,
  "train": 446068,
  "eval": 49563,
  "label_distribution": {"0": 198432, "1": 102891, "2": 194308},
  "source_distribution": {
    "halueval_qa": 20000,
    "halueval_dialogue": 20000,
    "halueval_summarization": 20000,
    "fever": 203465,
    "vitaminc": 100000,
    "anli_r3": 100459,
    "ragtruth": 2478,
    "aggrefact_sotopia": 1200,
    ...
  }
}
```

---

## CLI Usage

### Basic (4 core sources)

```bash
python training/data_pipeline.py
```

Loads HaluEval + FEVER + VitaminC + ANLI R3. Output: ~463K samples.

### Include RAG-specific sources

```bash
python training/data_pipeline.py --include-ragtruth --include-summac
```

### Include all sources (requires HF_TOKEN for AggreFact)

```bash
export HF_TOKEN="hf_..."
python training/data_pipeline.py --all
```

### Build from local source packs

For air-gapped regression builds or release evidence that must not download
from Hugging Face, provide one or more JSON Lines source packs:

```bash
python training/data_pipeline.py \
  --local-source-jsonl evidence/local_nli.jsonl \
  --output-dir training/data_local
```

Each JSONL row must use the production schema:

```json
{"premise":"Evidence text","hypothesis":"Claim text","label":0,"source":"local_contract"}
```

When `--local-source-jsonl` is present, the pipeline does not load remote
sources. It still uses the same `ClassLabel` cast, stratified split, on-disk
`DatasetDict` writer, and `stats.json` writer as the remote pipeline.

### Flags

| Flag                  | Default | Description                              |
|-----------------------|---------|------------------------------------------|
| `--include-ragtruth`  | off     | Include RAGTruth dataset                 |
| `--include-summac`    | off     | Include SummaC dataset                   |
| `--include-aggrefact` | off     | Include LLM-AggreFact (needs HF_TOKEN)   |
| `--local-source-jsonl` | none   | Local JSONL pack; repeat to combine packs |
| `--output-dir`        | `training/data` | Output DatasetDict + `stats.json` directory |
| `--all`               | off     | Enable all optional sources              |

---

## API Usage

```python
from pathlib import Path

from training.data_pipeline import build_dataset

# Core sources only
dataset = build_dataset()

# All sources
dataset = build_dataset(
    include_ragtruth=True,
    include_summac=True,
    include_aggrefact=True,
)

# Local deterministic source pack, no remote downloads
dataset = build_dataset(
    local_source_jsonl=[Path("evidence/local_nli.jsonl")],
    output_dir=Path("training/data_local"),
)

print(dataset)
# DatasetDict({
#     train: Dataset({features: ['premise', 'hypothesis', 'label', 'source'], num_rows: 446068})
#     eval:  Dataset({features: ['premise', 'hypothesis', 'label', 'source'], num_rows: 49563})
# })
```

---

## Dataset Split Strategy

The pipeline uses a **stratified 90/10 train/eval split** (line 381) with
`seed=42` for reproducibility. Stratification is by the `label` column,
ensuring that the label distribution in the eval set mirrors the training set.

This is achieved via HuggingFace's `train_test_split()` with
`stratify_by_column="label"`, which requires the column to be cast to
`ClassLabel` type first (line 376).

---

## Label Distribution Rationale

The combined dataset has an intentional imbalance:

- **Entailment (~40%)**: correct outputs the model should approve
- **Neutral (~21%)**: ambiguous cases requiring careful scoring
- **Contradiction (~39%)**: hallucinated outputs the model should reject

The near-balance between entailment and contradiction reflects Director-AI's
operational requirement: the guardrail must be equally capable of approving
correct outputs and rejecting hallucinated ones. The smaller neutral share
reflects the relative rarity of genuinely ambiguous claims in deployment.

---

## VitaminC Capping Rationale

Without the 100K cap, VitaminC contributes ~370K samples — 50.6% of the
combined dataset. This creates several problems:

1. **Distribution bias**: the trained model over-fits to VitaminC's specific
   adversarial editing style (minimal word substitutions to flip labels).
2. **Source diversity loss**: other sources' signal is diluted.
3. **Evaluation leakage risk**: VitaminC's eval split would dominate metrics,
   masking weaknesses on other domains.

The 100K cap (constant `VITAMINC_CAP`) reduces VitaminC's share to ~30%,
balancing its adversarial value against dataset diversity. The cap uses
a local `random.Random(42)` sampler for reproducibility.

---

## Downstream Consumers

1. **NLI model fine-tuning**: the 3-class dataset directly trains the FactCG
   DeBERTa-v3-Large model used by `NLIScorer`.
2. **Judge dataset construction**: `build_judge_dataset.py` loads this dataset,
   remaps to binary labels, runs NLI scoring to find borderline cases, and
   produces the judge training set.
3. **Benchmark evaluation**: `benchmarks/e2e_eval.py` uses HaluEval data
   (the same upstream source) for end-to-end guardrail evaluation.

---

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `HF_TOKEN` | For AggreFact only | HuggingFace token for gated dataset |
| `DIRECTOR_AI_SUMMAC_REVISION` | For production SummaC builds | Immutable SummaC commit SHA override when SummaC metadata is available |

---

## Dependencies

| Package      | Version | Purpose                            |
|--------------|---------|------------------------------------|
| `datasets`   | ≥2.14   | HuggingFace dataset loading/saving |
| `numpy`      | ≥1.24   | (indirect, via datasets)           |

---

## Error Handling

- **Missing HF_TOKEN**: AggreFact loader returns empty list with a warning.
  Pipeline continues without AggreFact data.
- **SummaC unavailable**: wrapped in `try/except`. Pipeline continues without
  SummaC data.
- **Empty premise/hypothesis**: skipped silently in all loaders (guard
  clauses at the start of each row-processing loop).
- **Unknown label format**: rows with labels that cannot be mapped to
  integers 0-2 are skipped via `continue`.
- **Local JSONL validation**: local source rows fail closed when `premise`,
  `hypothesis`, or `source` is missing or blank, when `label` is not exactly
  0, 1, or 2, or when the file is empty or malformed. Diagnostics include the
  source filename and line number.
- **Split validation**: the pipeline rejects empty datasets, missing labels,
  or label classes with fewer than two examples before writing any output.

---

## Performance

Dataset construction is I/O-bound (downloading from HuggingFace Hub) on
first run. Subsequent runs use the HuggingFace cache.

| Operation          | Typical duration | Bottleneck           |
|--------------------|------------------|----------------------|
| First run (all)    | 15-30 min        | Network download     |
| Cached run (all)   | 2-5 min          | Arrow serialisation  |
| Core sources only  | 1-3 min (cached) | Arrow serialisation  |
| Local JSONL pack    | seconds-minutes  | Arrow serialisation  |

Output dataset size on disk: ~2.1 GB (all sources), ~1.8 GB (core only).

---

## Testing

Covered by `tests/test_data_pipeline.py` and
`tests/test_data_pipeline_real_surface.py`:

- All seven loader functions tested with mocked `datasets.load_dataset`
- Label mapping correctness for each source
- VitaminC capping behaviour
- `build_dataset()` integration with optional source flags
- Pinned HuggingFace revisions for remote source loaders
- Empty/malformed row handling
- Statistics file generation and schema
- `ClassLabel` casting verification
- Real CLI build from a local JSONL source pack
- Fail-closed local-row validation with filename and line-number diagnostics

Run:

```bash
pytest tests/test_data_pipeline.py tests/test_data_pipeline_real_surface.py -v
```

---

## File Reference

| Item                     | Path                            |
|--------------------------|---------------------------------|
| Pipeline module          | `training/data_pipeline.py`     |
| Output dataset           | `training/data/`                |
| Statistics               | `training/data/stats.json`      |
| Tests                    | `tests/test_data_pipeline.py`, `tests/test_data_pipeline_real_surface.py` |
| Downstream: judge builder | `training/build_judge_dataset.py` |
| Downstream: NLI training | `training/train_nli.py`         |
