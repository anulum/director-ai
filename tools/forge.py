"""Forge: stacked NLI training techniques for accuracy superiority.

Four independently toggleable techniques that compose:

  1. R-Drop  — KL penalty between two dropout-masked forward passes.
               Forces output consistency. (Wu et al. 2021, arXiv:2106.14448)
  2. FGM    — Fast Gradient Method adversarial perturbation on word
               embeddings. (Miyato et al. 2017, arXiv:1605.07725)
  3. Focal  — Down-weight easy examples with (1-p_t)^γ weighting.
               (Lin et al. 2017, arXiv:1708.02002)
  4. Sieving — Token corruption during training (via SievingCollator).
               Handled by the data collator, not this Trainer.

ForgeTrainer extends HuggingFace Trainer. When all config values are 0,
it behaves identically to the base Trainer.

Usage:
    from forge import ForgeTrainer, ForgeConfig
    from sieving import SievingCollator

    cfg = ForgeConfig(rdrop_alpha=4.0, fgm_epsilon=1.0)
    collator = SievingCollator(tokenizer, noise_ratio=0.10)
    trainer = ForgeTrainer(
        model=model, args=args, forge_config=cfg,
        train_dataset=ds, data_collator=collator, ...
    )
    trainer.train()

Post-training:
    from forge import model_soup
    merged = model_soup([path1, path2, ...])  # weight-space average
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from transformers import Trainer


@dataclass
class ForgeConfig:
    """Training technique toggles. Set any value to 0.0 to disable it."""

    rdrop_alpha: float = 0.0
    """R-Drop KL penalty weight. Wu et al. recommend 4.0 for NLI."""

    fgm_epsilon: float = 0.0
    """FGM perturbation magnitude. Typical range: 0.5–1.5."""

    focal_gamma: float = 0.0
    """Focal loss gamma. 0 = standard CE. Lin et al. default: 2.0."""

    fgm_emb_name: str = "word_embeddings"
    """Name fragment to identify the embedding layer for FGM."""

    def active_techniques(self) -> list[str]:
        """Return names of enabled techniques (for logging)."""
        out = []
        if self.rdrop_alpha > 0:
            out.append(f"R-Drop(α={self.rdrop_alpha})")
        if self.fgm_epsilon > 0:
            out.append(f"FGM(ε={self.fgm_epsilon})")
        if self.focal_gamma > 0:
            out.append(f"Focal(γ={self.focal_gamma})")
        return out


# ── Loss functions ────────────────────────────────────────────────


def focal_loss(
    logits: torch.Tensor, labels: torch.Tensor, gamma: float
) -> torch.Tensor:
    """Focal loss: -(1 - p_t)^γ · log(p_t).  Lin et al. 2017, Eq. 5."""
    ce = F.cross_entropy(logits, labels, reduction="none")
    p_t = torch.exp(-ce)
    return ((1 - p_t) ** gamma * ce).mean()


def symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    """Symmetric KL divergence: 0.5 · (KL(a‖b) + KL(b‖a))."""
    p = F.softmax(logits_a, dim=-1)
    q = F.softmax(logits_b, dim=-1)
    log_p = F.log_softmax(logits_a, dim=-1)
    log_q = F.log_softmax(logits_b, dim=-1)
    return 0.5 * (
        F.kl_div(log_p, q, reduction="batchmean")
        + F.kl_div(log_q, p, reduction="batchmean")
    )


# ── Trainer ───────────────────────────────────────────────────────


class ForgeTrainer(Trainer):
    """HuggingFace Trainer with R-Drop, FGM, and Focal Loss stacked."""

    def __init__(self, *args, forge_config: ForgeConfig | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.forge = forge_config or ForgeConfig()
        self._fgm_backup: dict[str, torch.Tensor] = {}
        active = self.forge.active_techniques()
        if active:
            print(f"[Forge] Active: {', '.join(active)}")

    # ── Loss computation ──────────────────────────────────────────

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        fwd = {k: v for k, v in inputs.items() if k != "labels"}

        if self.forge.rdrop_alpha > 0 and model.training:
            out1 = model(**fwd)
            out2 = model(**fwd)

            loss1 = self._base_loss(out1.logits, labels)
            loss2 = self._base_loss(out2.logits, labels)
            ce = 0.5 * (loss1 + loss2)

            kl = symmetric_kl(out1.logits, out2.logits)
            loss = ce + self.forge.rdrop_alpha * kl
            return (loss, out1) if return_outputs else loss

        out = model(**fwd)
        loss = self._base_loss(out.logits, labels)
        return (loss, out) if return_outputs else loss

    def _base_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.forge.focal_gamma > 0:
            return focal_loss(logits, labels, self.forge.focal_gamma)
        return F.cross_entropy(logits, labels)

    # ── FGM adversarial training ──────────────────────────────────

    def training_step(self, model, inputs, num_items_in_batch=None, **kwargs):
        if self.forge.fgm_epsilon <= 0:
            return super().training_step(
                model, inputs, num_items_in_batch=num_items_in_batch, **kwargs
            )

        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        self.accelerator.backward(loss)

        # Adversarial pass: perturb embeddings → forward+backward → restore
        self._fgm_attack(model)
        with self.compute_loss_context_manager():
            loss_adv = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss_adv = loss_adv.mean()
        self.accelerator.backward(loss_adv)
        self._fgm_restore(model)

        return loss.detach()

    def _fgm_attack(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if self.forge.fgm_emb_name not in name:
                continue
            if param.grad is None:
                continue
            self._fgm_backup[name] = param.data.clone()
            norm = torch.norm(param.grad)
            if norm > 0:
                param.data.add_(self.forge.fgm_epsilon * param.grad / norm)

    def _fgm_restore(self, model):
        for name, param in model.named_parameters():
            if name in self._fgm_backup:
                param.data = self._fgm_backup[name]
        self._fgm_backup.clear()

    # ── Evaluation: disable sieving noise ─────────────────────────

    def evaluate(self, *args, **kwargs):
        if hasattr(self.data_collator, "training"):
            self.data_collator.training = False
        result = super().evaluate(*args, **kwargs)
        if hasattr(self.data_collator, "training"):
            self.data_collator.training = True
        return result


# ── Post-training: Model Soup ─────────────────────────────────────


def model_soup(
    checkpoint_dirs: list[str],
    output_dir: str | None = None,
) -> str:
    """Weight-space average of multiple fine-tuned checkpoints.

    All checkpoints must share the same architecture (e.g. all
    DeBERTa-v3-Large with 2 output labels). Wortsman et al. 2022, ICML.

    Returns the output directory path.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if len(checkpoint_dirs) < 2:
        raise ValueError("model_soup requires at least 2 checkpoints")

    print(f"[Forge] Model Soup: averaging {len(checkpoint_dirs)} checkpoints")

    # Load first model as the accumulator
    merged = AutoModelForSequenceClassification.from_pretrained(checkpoint_dirs[0])
    state = merged.state_dict()
    n = len(checkpoint_dirs)

    # Accumulate weights from remaining checkpoints
    for ckpt_dir in checkpoint_dirs[1:]:
        m = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
        for key, val in m.state_dict().items():
            state[key] = state[key] + val
        del m

    # Average
    for key in state:
        state[key] = state[key] / n

    merged.load_state_dict(state)

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(checkpoint_dirs[0]), "model-soup")
    os.makedirs(output_dir, exist_ok=True)
    merged.save_pretrained(output_dir)

    # Copy tokenizer from first checkpoint
    try:
        tok = AutoTokenizer.from_pretrained(checkpoint_dirs[0])
        tok.save_pretrained(output_dir)
    except Exception:
        pass

    print(f"[Forge] Model Soup saved to {output_dir}")
    return output_dir
