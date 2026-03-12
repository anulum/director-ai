"""Sieving: denoising-robust fine-tuning for NLI models.

Inspired by diffusion-based LM training (Mercury, arXiv:2506.17298),
Sieving corrupts a fraction of input tokens during training and forces
the model to still predict correct NLI labels from noisy input. This
produces representations that rely on semantic structure rather than
surface token matching — critical for adversarial (ANLI, PAWS) and
noisy-text (customer support, social media) domains.

Corruption strategy (BERT MLM ratios):
  - 80% of selected tokens → [MASK]
  - 10% → random vocabulary token
  - 10% → unchanged (model can't rely on [MASK] as signal)

Special tokens ([CLS], [SEP], [PAD]) are never corrupted.

Usage:
    from sieving import SievingCollator, SievingTrainer

    collator = SievingCollator(tokenizer, noise_ratio=0.10)
    trainer = SievingTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, compute_metrics=compute_metrics,
    )
    trainer.train()
"""

from __future__ import annotations

import torch
from transformers import DataCollatorWithPadding, Trainer


class SievingCollator:
    """Data collator that adds token-level noise during training.

    Wraps DataCollatorWithPadding: pads first, then corrupts a random
    fraction of non-special input tokens.
    """

    def __init__(self, tokenizer, noise_ratio: float = 0.10, mask_prob: float = 0.80):
        self.base = DataCollatorWithPadding(tokenizer, padding=True)
        self.noise_ratio = noise_ratio
        self.mask_prob = mask_prob
        self.mask_id = tokenizer.mask_token_id
        self.vocab_size = tokenizer.vocab_size
        self.special_ids = set(tokenizer.all_special_ids)
        self.training = True

    def __call__(self, features):
        batch = self.base(features)
        if self.training and self.noise_ratio > 0:
            batch["input_ids"] = self._corrupt(
                batch["input_ids"], batch["attention_mask"]
            )
        return batch

    def _corrupt(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        out = input_ids.clone()
        # Corruptible: not padding, not special tokens
        corruptible = attention_mask.bool()
        for sid in self.special_ids:
            corruptible = corruptible & (input_ids != sid)

        probs = torch.zeros_like(input_ids, dtype=torch.float)
        probs[corruptible] = self.noise_ratio
        selected = torch.bernoulli(probs).bool()

        rand = torch.rand_like(probs)
        mask_pos = selected & (rand < self.mask_prob)
        random_pos = (
            selected & ~mask_pos & (rand < self.mask_prob + 0.5 * (1 - self.mask_prob))
        )

        out[mask_pos] = self.mask_id
        random_tokens = torch.randint(0, self.vocab_size, out.shape, device=out.device)
        out[random_pos] = random_tokens[random_pos]
        return out


class SievingTrainer(Trainer):
    """Trainer that disables noise during evaluation."""

    def evaluate(self, *args, **kwargs):
        if hasattr(self.data_collator, "training"):
            self.data_collator.training = False
        result = super().evaluate(*args, **kwargs)
        if hasattr(self.data_collator, "training"):
            self.data_collator.training = True
        return result
