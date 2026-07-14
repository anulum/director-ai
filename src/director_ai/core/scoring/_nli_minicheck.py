# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — MiniCheck Backend (NLIScorer mixin)

"""MiniCheck backend for the NLI scorer.

:class:`MiniCheckBackendMixin` owns the ``minicheck`` package integration
of :class:`~director_ai.core.scoring.nli.NLIScorer`: lazy backend loading
(including the manual DeBERTa reconstruction used when ``device_map="auto"``
fails on ROCm/older torch), single-pair and batched scoring, and the
heuristic fallback on any backend failure. The mixin owns no ``__init__``
— the composing scorer initialises every attribute declared on the class
body, and the ``TYPE_CHECKING`` stub documents the fallback the scorer
provides.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ._nli_provisioning import _resolve_revision

__all__ = ["MiniCheckBackendMixin"]

logger = logging.getLogger("DirectorAI.NLI")


class MiniCheckBackendMixin:
    """MiniCheck-backend surface of :class:`NLIScorer`.

    All state is initialised by the composing scorer's ``__init__``; the
    annotations below declare that shared contract for static analysis
    without creating attributes.
    """

    # MiniCheck library variant name → pinned HuggingFace checkpoint. The variant
    # string is what the ``minicheck`` package's ``MiniCheck(model_name=...)``
    # accepts; the checkpoint is used for the immutable revision pin and the
    # manual DeBERTa fallback loader. Ordered fast/small → slow/accurate.
    _MINICHECK_CKPTS = {
        "deberta-v3-large": "lytang/MiniCheck-DeBERTa-v3-Large",
        "flan-t5-large": "lytang/MiniCheck-Flan-T5-Large",
        "Bespoke-MiniCheck-7B": "bespokelabs/Bespoke-MiniCheck-7B",
    }

    # Shared state initialised by the composing scorer.
    use_model: bool
    _minicheck: Any
    _minicheck_loaded: bool
    _minicheck_variant: str
    _cache_dir: str | None

    if TYPE_CHECKING:
        # Fallback provided by the composing scorer.

        @classmethod
        def _heuristic_score(cls, premise: str, hypothesis: str) -> float: ...

    def _ensure_minicheck(self) -> bool:
        """Load the MiniCheck backend if available."""
        if self._minicheck_loaded:
            return self._minicheck is not None
        self._minicheck_loaded = True
        try:  # pragma: no cover — requires minicheck package with model
            try:
                from minicheck import MiniCheck
            except ImportError:
                from minicheck.minicheck import MiniCheck

            variant = self._minicheck_variant
            try:
                self._minicheck = MiniCheck(
                    model_name=variant,
                    cache_dir=self._cache_dir,
                )
            except (RuntimeError, ValueError):
                if variant != "deberta-v3-large":
                    # The manual reconstruction below is DeBERTa-specific
                    # (sequence-classification head); larger variants such as
                    # Bespoke-MiniCheck-7B are causal LMs the package must load.
                    raise
                # device_map="auto" fails on ROCm/older torch — load manually
                logger.info("MiniCheck device_map=auto failed, loading manually")
                self._minicheck = MiniCheck.__new__(MiniCheck)
                from minicheck.inference import Inferencer

                inf = Inferencer.__new__(Inferencer)
                inf.model_name = variant
                inf.max_model_len = 2048
                inf.batch_size = 16

                import torch
                from transformers import (
                    AutoConfig,
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )

                ckpt = self._MINICHECK_CKPTS[variant]
                mc_rev = _resolve_revision(ckpt)
                config = AutoConfig.from_pretrained(
                    ckpt,
                    num_labels=2,
                    finetuning_task="text-classification",
                    cache_dir=self._cache_dir,
                    revision=mc_rev,
                )
                config.problem_type = "single_label_classification"
                inf.tokenizer = AutoTokenizer.from_pretrained(
                    ckpt,
                    use_fast=True,
                    cache_dir=self._cache_dir,
                    revision=mc_rev,
                )
                inf.model = AutoModelForSequenceClassification.from_pretrained(
                    ckpt,
                    config=config,
                    cache_dir=self._cache_dir,
                    revision=mc_rev,
                )
                from .._device import select_torch_device

                device = select_torch_device()
                inf.model.to(device).eval()
                inf.softmax = torch.nn.Softmax(dim=-1)
                if self._minicheck is None:
                    raise RuntimeError("MiniCheck wrapper not initialised") from None
                self._minicheck.model = inf

            logger.info("MiniCheck backend loaded.")
            return True
        except ImportError:
            logger.warning("minicheck package not installed — pip install minicheck")
            return False
        except (
            RuntimeError,
            OSError,
            ValueError,
            AttributeError,
        ) as e:
            logger.warning(
                "MiniCheck init failed: %s — using heuristic fallback",
                e,
            )
            self._minicheck = None
            return False

    def _minicheck_score(self, premise: str, hypothesis: str) -> float:
        """Score one pair through MiniCheck or fall back heuristically."""
        if not getattr(self, "use_model", True) and not self._minicheck_loaded:
            return self._heuristic_score(premise, hypothesis)
        if not self._ensure_minicheck() or self._minicheck is None:
            return self._heuristic_score(premise, hypothesis)
        try:
            result = self._minicheck.score(docs=[premise], claims=[hypothesis])
        except (
            RuntimeError,
            OSError,
            ValueError,
            AttributeError,
            NotImplementedError,
        ) as e:
            logger.warning("MiniCheck score failed: %s; using heuristic fallback", e)
            self._minicheck = None
            return self._heuristic_score(premise, hypothesis)
        # MiniCheck returns (pred_labels, max_probs, sentences, prob_arrays)
        if isinstance(result, tuple):
            _, max_probs, *_ = result
            return float(1.0 - max_probs[0])
        return float(1.0 - result[0])

    def _minicheck_score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score pairs through MiniCheck or fall back heuristically."""
        if not getattr(self, "use_model", True) and not self._minicheck_loaded:
            return [self._heuristic_score(p, h) for p, h in pairs]
        if not self._ensure_minicheck() or self._minicheck is None:
            return [self._heuristic_score(p, h) for p, h in pairs]
        docs = [p for p, _ in pairs]
        claims = [h for _, h in pairs]
        try:
            result = self._minicheck.score(docs=docs, claims=claims)
        except (
            RuntimeError,
            OSError,
            ValueError,
            AttributeError,
            NotImplementedError,
        ) as e:
            logger.warning(
                "MiniCheck batch score failed: %s; using heuristic fallback", e
            )
            self._minicheck = None
            return [self._heuristic_score(p, h) for p, h in pairs]
        if isinstance(result, tuple):
            _, max_probs, *_ = result
            preds = max_probs
        else:
            preds = result
        return [float(1.0 - s) for s in preds]
