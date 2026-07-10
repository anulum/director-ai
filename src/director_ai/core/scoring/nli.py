# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Real NLI Backend (DeBERTa)

"""NLI-based logical divergence scorer with batched inference and ONNX.

Default model: FactCG-DeBERTa-v3-Large (75.8% balanced accuracy
on AggreFact). Alternative: MiniCheck-DeBERTa-L (72.6%),
install with ``pip install director-ai[minicheck]``.

Backends: ``deberta`` (PyTorch), ``onnx`` (ONNX Runtime),
``minicheck``. Batch inference groups multiple chunks into a
single forward pass (3-5x latency reduction on chunked inputs).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .backends import ScorerBackend

from ..metrics import metrics
from ._nli_claims import ClaimCoverageMixin
from ._nli_export import (
    OnnxDynamicBatcher,
    _load_onnx_session,
    _validate_onnx_scheduler_config,
    export_onnx,
    export_tensorrt,
)
from ._nli_numeric import (
    _count_below_threshold as _count_below_threshold,
)
from ._nli_numeric import (
    _mean_float as _mean_float,
)
from ._nli_numeric import (
    _probs_to_confidence as _probs_to_confidence,
)
from ._nli_numeric import (
    _probs_to_divergence,
    _resolve_label_indices,
    _softmax_np,
)
from ._nli_numeric import (
    _sum_float_list as _sum_float_list,
)
from ._nli_numeric import (
    _weighted_sum_float as _weighted_sum_float,
)
from ._nli_provisioning import (
    _DEFAULT_MODEL,
    nli_available,
)
from ._nli_provisioning import (
    _DEFAULT_MODEL_REVISION as _DEFAULT_MODEL_REVISION,
)
from ._nli_provisioning import (
    _RECOMMENDED_MODEL as _RECOMMENDED_MODEL,
)
from ._nli_provisioning import (
    MODEL_REGISTRY as MODEL_REGISTRY,
)
from ._nli_provisioning import (
    _load_nli_model as _load_nli_model,
)
from ._nli_provisioning import (
    _resolve_revision as _resolve_revision,
)
from ._nli_provisioning import (
    clear_model_cache as clear_model_cache,
)

__all__ = [
    "NLIScorer",
    "OnnxDynamicBatcher",
    "export_onnx",
    "export_tensorrt",
    "nli_available",
]

# GPU amortization: ~$0.01/1K tokens for local DeBERTa inference
_DEFAULT_COST_PER_TOKEN = 1e-5

logger = logging.getLogger("DirectorAI.NLI")

# FactCG instruction template (NAACL 2025, derenlei/FactCG)
_FACTCG_TEMPLATE = (
    "{text_a}\n\nChoose your answer: based on the paragraph above "
    'can we conclude that "{text_b}"?\n\nOPTIONS:\n- Yes\n- No\n'
    "I think the answer is "
)

# Heuristic divergence defaults (shared with scorer.py)
_DIVERGENCE_NEUTRAL = 0.5
_DIVERGENCE_ALIGNED = 0.1
_DIVERGENCE_CONTRADICTED = 0.9


# ── Scorer ───────────────────────────────────────────────────────


class NLIScorer(ClaimCoverageMixin):
    """NLI-based logical divergence scorer.

    Parameters
    ----------
    use_model : bool — attempt to load model on first score().
    max_length : int — max token length for NLI input.
    model_name : str | None — HuggingFace model ID or local path.
    backend : str | ScorerBackend — "deberta", "onnx", "minicheck",
        "lite", or a ScorerBackend instance.
    quantize_8bit : bool — 8-bit quantization (requires bitsandbytes).
    device : str | None — torch device ("cpu", "cuda", "cuda:0").
    torch_dtype : str | None — "float16", "bfloat16", or "float32".
    onnx_path : str | None — directory with exported ONNX model.

    """

    _BACKENDS = ("deberta", "minicheck", "onnx", "lite")

    # MiniCheck library variant name → pinned HuggingFace checkpoint. The variant
    # string is what the ``minicheck`` package's ``MiniCheck(model_name=...)``
    # accepts; the checkpoint is used for the immutable revision pin and the
    # manual DeBERTa fallback loader. Ordered fast/small → slow/accurate.
    _MINICHECK_CKPTS = {
        "deberta-v3-large": "lytang/MiniCheck-DeBERTa-v3-Large",
        "flan-t5-large": "lytang/MiniCheck-Flan-T5-Large",
        "Bespoke-MiniCheck-7B": "bespokelabs/Bespoke-MiniCheck-7B",
    }

    def __init__(
        self,
        use_model: bool = True,
        max_length: int = 512,
        model_name: str | None = None,
        backend: str | ScorerBackend = "deberta",
        quantize_8bit: bool = False,
        device: str | None = None,
        torch_dtype: str | None = None,
        onnx_path: str | None = None,
        onnx_batch_size: int = 16,
        onnx_flush_timeout_ms: float = 10.0,
        cost_per_token: float = _DEFAULT_COST_PER_TOKEN,
        lora_adapter_path: str | None = None,
        revision: str | None = None,
        minicheck_variant: str = "deberta-v3-large",
    ) -> None:
        # Accept ScorerBackend instance directly
        self._custom_backend = None
        if not isinstance(backend, str):
            from .backends import ScorerBackend

            if isinstance(backend, ScorerBackend):
                self._custom_backend = backend
                backend = "__custom__"
            else:
                raise TypeError(
                    f"backend must be str or ScorerBackend, got {type(backend)!r}",
                )

        if backend != "__custom__" and backend not in self._BACKENDS:
            raise ValueError(
                f"backend must be one of {self._BACKENDS}, got {backend!r}",
            )
        if backend == "onnx":
            _validate_onnx_scheduler_config(
                onnx_batch_size,
                onnx_flush_timeout_ms,
            )
        self.use_model = use_model
        self.max_length = max_length
        self.backend = backend
        self._model_name = model_name or _DEFAULT_MODEL
        self._quantize_8bit = quantize_8bit
        self._device = device
        self._torch_dtype = torch_dtype
        self._onnx_path = onnx_path
        self._onnx_batch_size = onnx_batch_size
        self._onnx_flush_timeout_ms = onnx_flush_timeout_ms
        self._tokenizer = None
        self._model = None
        self._onnx_session = None
        self._model_loaded = False
        self._minicheck = None
        self._minicheck_loaded = False
        self._cache_dir = os.environ.get("HF_HOME")
        self._lite_scorer: Any = None
        self._onnx_batcher: OnnxDynamicBatcher | None = None
        self._last_token_count: int = 0
        self._cost_per_token: float = cost_per_token
        # Label indices resolved from model.config.id2label after loading.
        # None = not yet resolved; tuple = (contradiction_idx, neutral_idx)
        self._label_indices: tuple[int, int] | None = None
        self._lora_adapter_path = lora_adapter_path
        self._revision = revision
        if minicheck_variant not in self._MINICHECK_CKPTS:
            raise ValueError(
                f"minicheck_variant must be one of "
                f"{tuple(self._MINICHECK_CKPTS)}, got {minicheck_variant!r}",
            )
        self._minicheck_variant = minicheck_variant

    @property
    def _backend_ready(self) -> bool:
        """Return whether the selected scoring backend is ready for inference."""
        if self._custom_backend is not None:
            return True
        if self.backend == "lite":
            return True
        if self.backend == "onnx":
            return self._onnx_session is not None
        return self._model is not None

    def _ensure_model(self) -> bool:
        """Load model if not yet loaded. Returns True if ready."""
        if self._model_loaded:
            return self._backend_ready
        if self._custom_backend is not None:
            # A custom ScorerBackend supplies its own inference; never load an
            # NLI checkpoint for it. score()/score_batch() delegate directly.
            self._model_loaded = True
            return True
        if not self.use_model:
            self._model_loaded = True
            return False

        if self.backend == "onnx":
            if not self._onnx_path:
                logger.warning(
                    "onnx backend requires onnx_path — falling back to heuristic",
                )
                self._model_loaded = True
                return False
            # Path-traversal safety is handled inside _load_onnx_session, which
            # canonicalises the directory and enforces _is_relative_to on the
            # resolved model file; a bad path falls back to the heuristic.
            self._tokenizer, self._onnx_session = _load_onnx_session(
                self._onnx_path,
                device=self._device,
            )
            self._onnx_batcher = OnnxDynamicBatcher(
                onnx_scorer_fn=self._onnx_score_batch,
                max_batch=self._onnx_batch_size,
                flush_timeout_ms=self._onnx_flush_timeout_ms,
                session=self._onnx_session,
            )
        else:
            self._tokenizer, self._model = _load_nli_model(
                self._model_name,
                quantize_8bit=self._quantize_8bit,
                device=self._device,
                torch_dtype=self._torch_dtype,
                revision=self._revision,
            )
            if self._model is not None:
                self._label_indices = _resolve_label_indices(self._model)
                if self._lora_adapter_path:
                    self._load_lora_adapter(self._lora_adapter_path)
        self._model_loaded = True
        return self._backend_ready

    def _load_lora_adapter(self, adapter_path: str) -> None:
        """Merge a PEFT/LoRA adapter into the loaded base model."""
        try:
            from peft import PeftModel

            logger.info("Loading LoRA adapter: %s", adapter_path)
            if self._model is None:
                raise RuntimeError("Cannot load LoRA adapter before base NLI model")
            peft_model = PeftModel.from_pretrained(self._model, adapter_path)
            merged = peft_model.merge_and_unload()
            merged.eval()
            self._model = merged
            logger.info("LoRA adapter merged successfully")
        except ImportError:
            logger.warning("peft not installed — cannot load LoRA adapter")
        except (OSError, ValueError) as e:
            logger.warning("Failed to load LoRA adapter: %s", e)

    @property
    def model_available(self) -> bool:
        """Return whether model-backed scoring is available after lazy loading."""
        return self._ensure_model()

    @property
    def last_token_count(self) -> int:
        """Return the number of tokens processed since the last reset."""
        return self._last_token_count

    @property
    def last_estimated_cost(self) -> float:
        """Return estimated inference cost for the accumulated token count."""
        return self._last_token_count * self._cost_per_token

    def reset_token_counter(self) -> None:
        """Reset accumulated token accounting for this scorer instance."""
        self._last_token_count = 0

    def score(self, premise: str, hypothesis: str) -> float:
        """Compute logical divergence between premise and hypothesis.

        Returns float in [0, 1]: 0 = entailment, 1 = contradiction.
        """
        if self._custom_backend is not None:
            return self._custom_backend.score(premise, hypothesis)
        if self.backend == "lite":
            return self._lite_score(premise, hypothesis)
        if self.backend == "minicheck":
            return self._minicheck_score(premise, hypothesis)
        if not self._ensure_model():
            return self._heuristic_score(premise, hypothesis)
        if self.backend == "onnx":
            return self._onnx_score_batch([(premise, hypothesis)])[0]
        return self._model_score(premise, hypothesis)

    async def ascore(self, premise: str, hypothesis: str) -> float:
        """Async score() — runs inference in a thread pool."""
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.score, premise, hypothesis)

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score multiple (premise, hypothesis) pairs.

        Uses a single batched forward pass when a model backend
        is available (3-5x faster than sequential scoring).
        """
        if not pairs:
            return []
        if self._custom_backend is not None:
            return self._custom_backend.score_batch(pairs)
        if self.backend == "lite":
            return self._lite_score_batch(pairs)
        if self.backend == "minicheck":
            return self._minicheck_score_batch(pairs)
        if not self._ensure_model():
            return [self._heuristic_score(p, h) for p, h in pairs]
        if self.backend == "onnx":
            return self._onnx_score_batch(pairs)
        return self._model_score_batch(pairs)

    async def ascore_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Async batch scoring — runs in a thread pool."""
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.score_batch, pairs)

    # ── MiniCheck backend ────────────────────────────────────────

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

    # ── PyTorch backend ──────────────────────────────────────────

    @property
    def _is_factcg(self) -> bool:
        """Return whether the configured model expects the FactCG prompt template."""
        return "factcg" in self._model_name.lower()

    def _model_score(self, premise: str, hypothesis: str) -> float:
        """Single-pair PyTorch inference.

        Handles 2-class (supported/not-supported) and 3-class
        (entailment/neutral/contradiction) models. FactCG uses an
        instruction template; standard NLI uses two-segment input.
        """
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("NLI model not loaded")

        import torch

        device = next(self._model.parameters()).device

        with metrics.timer("nli_inference_seconds"):
            if self._is_factcg:
                text = _FACTCG_TEMPLATE.format(text_a=premise, text_b=hypothesis)
                inputs = self._tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                )
            else:
                inputs = self._tokenizer(
                    premise,
                    hypothesis,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                )

            self._last_token_count += inputs["input_ids"].numel()
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self._model(**inputs).logits

            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        if len(probs) == 2:
            return float(1.0 - probs[1])
        ci, ni = self._label_indices or (2, 1)
        return float(probs[ci]) + float(probs[ni]) * 0.5

    def _model_score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Batched PyTorch inference — single forward pass."""
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("NLI model not loaded")

        import torch

        device = next(self._model.parameters()).device

        with metrics.timer("nli_batch_inference_seconds"):
            if self._is_factcg:
                texts = [_FACTCG_TEMPLATE.format(text_a=p, text_b=h) for p, h in pairs]
                inputs = self._tokenizer(
                    texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                )
            else:
                premises = [p for p, _ in pairs]
                hypotheses = [h for _, h in pairs]
                inputs = self._tokenizer(
                    premises,
                    hypotheses,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                )

            self._last_token_count += inputs["input_ids"].numel()
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self._model(**inputs).logits

            probs = torch.softmax(logits, dim=1).cpu().numpy()

        return _probs_to_divergence(probs, self._label_indices)

    # ── ONNX backend ─────────────────────────────────────────────

    def _onnx_score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Batched ONNX Runtime inference."""
        if self._tokenizer is None or self._onnx_session is None:
            raise RuntimeError("ONNX session not loaded")

        with metrics.timer("nli_onnx_batch_seconds"):
            if self._is_factcg:
                texts = [_FACTCG_TEMPLATE.format(text_a=p, text_b=h) for p, h in pairs]
                inputs = self._tokenizer(
                    texts,
                    return_tensors="np",
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                )
            else:
                premises = [p for p, _ in pairs]
                hypotheses = [h for _, h in pairs]
                inputs = self._tokenizer(
                    premises,
                    hypotheses,
                    return_tensors="np",
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                )

            self._last_token_count += inputs["input_ids"].size
            # Feed only inputs the ONNX graph expects, cast to int64
            expected = {i.name for i in self._onnx_session.get_inputs()}
            feed = {
                k: v.astype(np.int64) if v.dtype != np.int64 else v
                for k, v in inputs.items()
                if k in expected
            }
            logits = self._onnx_session.run(None, feed)[0]

        return _probs_to_divergence(_softmax_np(logits), self._label_indices)

    def score_batch_with_confidence(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[tuple[float, float]]:
        """Score pairs and return (divergence, confidence) tuples.

        Confidence is 1 - entropy of the softmax distribution, normalised
        to [0, 1]. High confidence = model is certain about its prediction.
        """
        if not pairs:
            return []
        if self._custom_backend is not None or self.backend == "lite":
            scores = self.score_batch(pairs)
            return [(s, 1.0) for s in scores]
        if self.backend == "minicheck":
            scores = self.score_batch(pairs)
            return [(s, 1.0) for s in scores]
        if not self._ensure_model():
            scores = [self._heuristic_score(p, h) for p, h in pairs]
            return [(s, 0.5) for s in scores]

        if self.backend == "onnx":
            return self._onnx_score_batch_with_confidence(pairs)
        return self._model_score_batch_with_confidence(pairs)

    def _model_score_batch_with_confidence(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[tuple[float, float]]:
        """Batched PyTorch inference returning (divergence, confidence)."""
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("NLI model not loaded")

        import torch

        device = next(self._model.parameters()).device

        with metrics.timer("nli_batch_inference_seconds"):
            if self._is_factcg:
                texts = [_FACTCG_TEMPLATE.format(text_a=p, text_b=h) for p, h in pairs]
                inputs = self._tokenizer(
                    texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                )
            else:
                premises = [p for p, _ in pairs]
                hypotheses = [h for _, h in pairs]
                inputs = self._tokenizer(
                    premises,
                    hypotheses,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                )

            self._last_token_count += inputs["input_ids"].numel()
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        divergences = _probs_to_divergence(probs, self._label_indices)
        confidences = _probs_to_confidence(probs)
        return list(zip(divergences, confidences, strict=True))

    def _onnx_score_batch_with_confidence(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[tuple[float, float]]:
        """Batched ONNX inference returning (divergence, confidence)."""
        if self._tokenizer is None or self._onnx_session is None:
            raise RuntimeError("ONNX session not loaded")

        with metrics.timer("nli_onnx_batch_seconds"):
            if self._is_factcg:
                texts = [_FACTCG_TEMPLATE.format(text_a=p, text_b=h) for p, h in pairs]
                inputs = self._tokenizer(
                    texts,
                    return_tensors="np",
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                )
            else:
                premises = [p for p, _ in pairs]
                hypotheses = [h for _, h in pairs]
                inputs = self._tokenizer(
                    premises,
                    hypotheses,
                    return_tensors="np",
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                )

            self._last_token_count += inputs["input_ids"].size
            expected = {i.name for i in self._onnx_session.get_inputs()}
            feed = {
                k: v.astype(np.int64) if v.dtype != np.int64 else v
                for k, v in inputs.items()
                if k in expected
            }
            logits = self._onnx_session.run(None, feed)[0]

        sm = _softmax_np(logits)
        divergences = _probs_to_divergence(sm, self._label_indices)
        confidences = _probs_to_confidence(sm)
        return list(zip(divergences, confidences, strict=True))

    # ── Lite backend ─────────────────────────────────────────────

    def _ensure_lite(self) -> None:
        """Initialise the lightweight lexical scorer backend."""
        if self._lite_scorer is None:
            from .lite_scorer import LiteScorer

            self._lite_scorer = LiteScorer()

    def _lite_score(self, premise: str, hypothesis: str) -> float:
        """Score one pair with the lightweight lexical backend."""
        self._ensure_lite()
        return float(self._lite_scorer.score(premise, hypothesis))

    def _lite_score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score pairs with the lightweight lexical backend."""
        self._ensure_lite()
        return [float(v) for v in self._lite_scorer.score_batch(pairs)]

    # ── Heuristic fallback ───────────────────────────────────────

    _NEGATION_WORDS = frozenset(
        {
            "not",
            "no",
            "never",
            "neither",
            "nobody",
            "nothing",
            "nowhere",
            "nor",
            "cannot",
            "can't",
            "isn't",
            "aren't",
            "wasn't",
            "weren't",
            "won't",
            "wouldn't",
            "shouldn't",
            "couldn't",
            "doesn't",
            "don't",
            "didn't",
            "hasn't",
            "haven't",
            "hadn't",
            "without",
            "false",
        },
    )

    @classmethod
    def _heuristic_score(cls, premise: str, hypothesis: str) -> float:
        """Deterministic heuristic fallback (no model needed)."""
        h_lower = hypothesis.lower()
        if "consistent with reality" in h_lower:
            return _DIVERGENCE_ALIGNED
        if "opposite is true" in h_lower:
            return _DIVERGENCE_CONTRADICTED
        if "depends on your perspective" in h_lower:
            return _DIVERGENCE_NEUTRAL
        p_words = set(premise.lower().split())
        h_words = set(hypothesis.lower().split())
        if not p_words:
            return _DIVERGENCE_NEUTRAL
        overlap = len(p_words & h_words) / max(len(p_words), 1)
        raw = _DIVERGENCE_NEUTRAL - overlap * 0.3
        # Negation asymmetry: if one side has negation and the other
        # doesn't, high overlap likely means semantic contradiction.
        p_neg = bool(p_words & cls._NEGATION_WORDS)
        h_neg = bool(h_words & cls._NEGATION_WORDS)
        if p_neg != h_neg and overlap > 0.3:
            raw = max(raw, 0.7)
        return float(np.clip(raw, _DIVERGENCE_ALIGNED, _DIVERGENCE_CONTRADICTED))
