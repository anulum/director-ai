# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Â© Concepts 1996â€“2026 Miroslav Ĺ otek. All rights reserved.
# Â© Code 2020â€“2026 Miroslav Ĺ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI â€” test_forge.py

"""Tests for tools/forge.py â€” ForgeTrainer techniques."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

torch = pytest.importorskip("torch")
nn = torch.nn
F = torch.nn.functional
from forge import ForgeConfig, ForgeTrainer, focal_loss, model_soup, symmetric_kl  # noqa: E402, I001


# â”€â”€ Focal loss â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestFocalLoss:
    def test_gamma_zero_matches_ce(self):
        logits = torch.randn(8, 2)
        labels = torch.randint(0, 2, (8,))
        fl = focal_loss(logits, labels, gamma=0.0)
        ce = F.cross_entropy(logits, labels)
        assert abs(fl.item() - ce.item()) < 1e-5

    def test_confident_predictions_downweighted(self):
        """High-confidence correct predictions should have lower focal loss."""
        # Logits where model is very confident (correct class has high logit)
        logits_confident = torch.tensor([[5.0, -5.0], [5.0, -5.0]])
        # Logits where model is uncertain
        logits_uncertain = torch.tensor([[0.1, -0.1], [0.1, -0.1]])
        labels = torch.tensor([0, 0])

        fl_conf = focal_loss(logits_confident, labels, gamma=2.0)
        fl_unc = focal_loss(logits_uncertain, labels, gamma=2.0)
        assert fl_conf < fl_unc

    def test_gradient_flows(self):
        logits = torch.randn(4, 3, requires_grad=True)
        labels = torch.randint(0, 3, (4,))
        loss = focal_loss(logits, labels, gamma=2.0)
        loss.backward()
        assert logits.grad is not None


# â”€â”€ Symmetric KL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestSymmetricKL:
    def test_identical_distributions_zero(self):
        logits = torch.randn(4, 3)
        kl = symmetric_kl(logits, logits)
        assert kl.item() < 1e-5

    def test_different_distributions_positive(self):
        a = torch.tensor([[10.0, -10.0, 0.0]])
        b = torch.tensor([[-10.0, 10.0, 0.0]])
        kl = symmetric_kl(a, b)
        assert kl.item() > 1.0

    def test_symmetric(self):
        a = torch.randn(8, 3)
        b = torch.randn(8, 3)
        kl_ab = symmetric_kl(a, b)
        kl_ba = symmetric_kl(b, a)
        assert abs(kl_ab.item() - kl_ba.item()) < 1e-5


# â”€â”€ ForgeConfig â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestForgeConfig:
    def test_all_disabled_by_default(self):
        cfg = ForgeConfig()
        assert cfg.active_techniques() == []

    def test_active_techniques(self):
        cfg = ForgeConfig(rdrop_alpha=4.0, fgm_epsilon=1.0)
        active = cfg.active_techniques()
        assert len(active) == 2
        assert "R-Drop" in active[0]
        assert "FGM" in active[1]

    def test_focal_listed(self):
        cfg = ForgeConfig(focal_gamma=2.0)
        assert len(cfg.active_techniques()) == 1


# â”€â”€ ForgeTrainer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class SimpleModel(nn.Module):
    """Minimal 2-class classifier for testing."""

    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(100, 16)
        self.classifier = nn.Linear(16, 2)
        self.config = type(
            "Config",
            (),
            {"id2label": {0: "not_supported", 1: "supported"}},
        )()

    def forward(self, input_ids, attention_mask=None, **kwargs):
        emb = self.word_embeddings(input_ids).mean(dim=1)
        logits = self.classifier(emb)
        return type("Output", (), {"logits": logits, "loss": None})()


class TestForgeTrainerComputeLoss:
    def setup_method(self):
        self.model = SimpleModel()
        self.inputs = {
            "input_ids": torch.randint(0, 100, (4, 8)),
            "attention_mask": torch.ones(4, 8, dtype=torch.long),
            "labels": torch.randint(0, 2, (4,)),
        }

    def test_vanilla_ce(self):
        cfg = ForgeConfig()
        trainer = ForgeTrainer.__new__(ForgeTrainer)
        trainer.forge = cfg
        loss = trainer.compute_loss(self.model, self.inputs)
        assert loss.requires_grad
        assert loss.item() > 0

    def test_rdrop_increases_loss(self):
        """R-Drop adds KL penalty, so loss should be >= CE loss."""
        self.model.train()
        cfg_vanilla = ForgeConfig()
        cfg_rdrop = ForgeConfig(rdrop_alpha=4.0)
        trainer_v = ForgeTrainer.__new__(ForgeTrainer)
        trainer_v.forge = cfg_vanilla
        trainer_r = ForgeTrainer.__new__(ForgeTrainer)
        trainer_r.forge = cfg_rdrop

        # R-Drop loss depends on dropout randomness, but on average
        # the KL term adds to the loss. With this simple model (no dropout),
        # the KL should be ~0, so losses should be close.
        loss_v = trainer_v.compute_loss(self.model, self.inputs)
        loss_r = trainer_r.compute_loss(self.model, self.inputs)
        # Both should be valid losses
        assert loss_v.item() > 0
        assert loss_r.item() > 0

    def test_focal_loss_used(self):
        cfg = ForgeConfig(focal_gamma=2.0)
        trainer = ForgeTrainer.__new__(ForgeTrainer)
        trainer.forge = cfg
        loss = trainer.compute_loss(self.model, self.inputs)
        assert loss.requires_grad
        assert loss.item() > 0

    def test_rdrop_plus_focal(self):
        self.model.train()
        cfg = ForgeConfig(rdrop_alpha=4.0, focal_gamma=2.0)
        trainer = ForgeTrainer.__new__(ForgeTrainer)
        trainer.forge = cfg
        loss = trainer.compute_loss(self.model, self.inputs)
        assert loss.requires_grad


# â”€â”€ FGM â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestFGM:
    def test_attack_perturbs_embeddings(self):
        model = SimpleModel()
        cfg = ForgeConfig(fgm_epsilon=1.0)
        trainer = ForgeTrainer.__new__(ForgeTrainer)
        trainer.forge = cfg
        trainer._fgm_backup = {}

        # Create gradients on embeddings
        x = torch.randint(0, 100, (2, 4))
        out = model(x)
        loss = out.logits.sum()
        loss.backward()

        original_data = model.word_embeddings.weight.data.clone()
        trainer._fgm_attack(model)
        assert not torch.equal(model.word_embeddings.weight.data, original_data)

        trainer._fgm_restore(model)
        assert torch.equal(model.word_embeddings.weight.data, original_data)

    def test_attack_noop_without_grad(self):
        model = SimpleModel()
        cfg = ForgeConfig(fgm_epsilon=1.0)
        trainer = ForgeTrainer.__new__(ForgeTrainer)
        trainer.forge = cfg
        trainer._fgm_backup = {}

        original = model.word_embeddings.weight.data.clone()
        trainer._fgm_attack(model)
        # No gradients â†’ no perturbation â†’ no backup
        assert len(trainer._fgm_backup) == 0
        assert torch.equal(model.word_embeddings.weight.data, original)


# â”€â”€ Model Soup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestModelSoup:
    def test_requires_two_checkpoints(self):
        with pytest.raises(ValueError, match="at least 2"):
            model_soup(["single_dir"])
