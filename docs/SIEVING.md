# Sieving: Denoising-Robust Fine-Tuning for NLI

## Concept

Sieving corrupts a fraction of input tokens during NLI fine-tuning and
forces the model to still predict correct entailment/contradiction labels.
This produces representations grounded in semantic structure rather than
surface-level token overlap.

Named after the metallurgical process of separating signal from noise
through progressively finer filters.

## Motivation

Inspired by diffusion-based LM training (Mercury, arXiv:2506.17298) which
replaces autoregressive loss with a denoising diffusion loss, forcing
models to reconstruct from corrupted inputs. While Mercury targets
generative decoding speed, we adapt the core insight — denoising makes
representations more robust — for discriminative NLI classification.

### Problem

Standard NLI fine-tuning produces models that over-rely on surface features:
- High lexical overlap → predict entailment (fails on PAWS)
- Presence of negation words → predict contradiction (crude heuristic)
- Dataset-specific artifacts → memorized shortcuts

### Solution

By training on corrupted inputs (10-15% of tokens masked or randomized),
the model cannot rely on any single token's presence. It must learn
distributed, semantic-level features that survive partial information loss.

## Implementation

### SievingCollator

Wraps HuggingFace's `DataCollatorWithPadding`. After padding, corrupts
a configurable fraction of non-special tokens:

| Selection | Action | Probability |
|-----------|--------|-------------|
| Selected (noise_ratio) | Replace with [MASK] | 80% |
| Selected (noise_ratio) | Replace with random token | 10% |
| Selected (noise_ratio) | Keep unchanged | 10% |
| Not selected | Keep unchanged | — |

Special tokens ([CLS], [SEP], [PAD]) are never corrupted.
Noise is applied only during training (disabled during evaluation).

### SievingTrainer

Subclass of HuggingFace Trainer that automatically disables noise
in the collator during `evaluate()` calls.

### Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `noise_ratio` | 0.10 | 0.05–0.20 | Fraction of tokens corrupted per sample |
| `mask_prob` | 0.80 | 0.60–0.90 | Fraction of corrupted tokens replaced with [MASK] |

## Usage

```python
from sieving import SievingCollator, SievingTrainer

collator = SievingCollator(tokenizer, noise_ratio=0.10)
trainer = SievingTrainer(
    model=model, args=training_args,
    train_dataset=train_ds, eval_dataset=val_ds,
    data_collator=collator, compute_metrics=compute_metrics,
)
trainer.train()
```

## GPU Memory Constraints

DeBERTa-v3-Large (304M params) with max_length=512 is tight on 24GB GPUs:

| GPU VRAM | Max train batch_size | Recommended config |
|----------|---------------------|--------------------|
| 24GB (RTX 6000) | 4 | `bs=4, grad_accum=4, eval_bs=8` |
| 48GB (L40S/A6000) | 16 | `bs=16, eval_bs=32` |

Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for better memory fragmentation handling.

## A/B Test Protocol

`run_sieving_demo.py` runs side-by-side comparison on ClimateFEVER:

1. **Standard**: Fine-tune with noise_ratio=0.0 (baseline)
2. **Sieving**: Fine-tune with noise_ratio=0.10

ClimateFEVER is flattened to (evidence, claim) pairs (~6,873 total).
Evidence labels are **integers**: 0=SUPPORTS→1, 2=REFUTES→0, 1=NOT_ENOUGH_INFO→skip.

Both are evaluated on:
- **Clean test set**: original ClimateFEVER test split
- **Noisy test set**: same data with 5% character-level typos injected

Metrics:
- `clean_bal_acc`: balanced accuracy on clean data
- `noisy_bal_acc`: balanced accuracy on typo-corrupted data
- `robustness_gap`: clean - noisy (lower = more robust)

## Expected Outcomes

Based on analogous techniques (R-Drop, token cutoff, manifold mixup):
- Clean accuracy: neutral to slight improvement (+0 to +2%)
- Noisy accuracy: measurable improvement (+2 to +5%)
- Robustness gap: significant reduction (30-50% smaller)

The technique is most valuable for domains with noisy input:
customer support, social media, OCR-extracted text, ASR transcripts.

## Files

| File | Purpose |
|------|---------|
| `tools/sieving.py` | SievingCollator + SievingTrainer (standalone, uploadable to GPU) |
| `tools/run_sieving_demo.py` | A/B comparison on ClimateFEVER |
| `docs/SIEVING.md` | This document |

## References

- Mercury (arXiv:2506.17298): diffusion loss for LMs, parallel denoising
- BERT (Devlin et al. 2019): original masked language modeling
- R-Drop (Wu et al. 2021): regularization via output distribution consistency
- Token Cutoff (Shen et al. 2020): data augmentation by token deletion
