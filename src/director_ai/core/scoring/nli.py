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
import threading
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .backends import ScorerBackend
    from .claim_decomposition import AtomicClaimDecomposer

from ._nli_claims import ClaimCoverageMixin
from ._nli_export import (
    OnnxDynamicBatcher,
    _load_onnx_session,
    _validate_onnx_scheduler_config,
    export_onnx,
    export_tensorrt,
)
from ._nli_minicheck import MiniCheckBackendMixin
from ._nli_model_inference import ModelInferenceMixin
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
    _probs_to_divergence as _probs_to_divergence,
)
from ._nli_numeric import (
    _resolve_label_indices,
)
from ._nli_numeric import (
    _softmax_np as _softmax_np,
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

# Heuristic divergence defaults (shared with scorer.py)
_DIVERGENCE_NEUTRAL = 0.5
_DIVERGENCE_ALIGNED = 0.1
_DIVERGENCE_CONTRADICTED = 0.9


# ── Scorer ───────────────────────────────────────────────────────


class NLIScorer(ClaimCoverageMixin, MiniCheckBackendMixin, ModelInferenceMixin):
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
        claim_decomposer: AtomicClaimDecomposer | None = None,
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
        self._tokenizer_lock = threading.Lock()
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
        self._claim_decomposer = claim_decomposer
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
