# SPDX-License-Identifier: AGPL-3.0-or-later
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
import re
from functools import lru_cache

import numpy as np

from ..metrics import metrics

try:
    from backfire_kernel import (
        rust_probs_to_confidence,
        rust_probs_to_divergence,
        rust_softmax,
    )

    _RUST_NLI = True
except ImportError:
    _RUST_NLI = False
from ._nli_export import (
    OnnxDynamicBatcher,  # noqa: F401 — re-export
    _load_onnx_session,
    export_onnx,  # noqa: F401 — re-export
    export_tensorrt,  # noqa: F401 — re-export
)

__all__ = [
    "NLIScorer",
    "export_onnx",
    "export_tensorrt",
    "nli_available",
]

# GPU amortization: ~$0.01/1K tokens for local DeBERTa inference
_DEFAULT_COST_PER_TOKEN = 1e-5

logger = logging.getLogger("DirectorAI.NLI")

_DEFAULT_MODEL = "yaxili96/FactCG-DeBERTa-v3-Large"
_DEFAULT_MODEL_REVISION = "0430e3509dbd28d2dff7a117c0eae25359ff3e80"
_RECOMMENDED_MODEL = "lytang/MiniCheck-DeBERTa-L"

#: Pinned revisions for all known HuggingFace Hub models used in
#: production. Prevents supply-chain drift when upstream pushes a
#: new commit.  Update SHAs deliberately after verifying the new
#: revision against the AggreFact benchmark.
MODEL_REGISTRY: dict[str, str] = {
    _DEFAULT_MODEL: _DEFAULT_MODEL_REVISION,
    "lytang/MiniCheck-DeBERTa-v3-Large": "2f2d01a54fa022a7ffadb76260e1ea8bc88c82bb",
}


def _resolve_revision(model_name: str, revision: str | None = None) -> str | None:
    """Return an explicit revision SHA for *model_name*, or *None* if unknown.

    If *revision* is already set by the caller, it takes precedence.
    """
    if revision is not None:
        return revision
    return MODEL_REGISTRY.get(model_name)


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


# â"€â"€ Helpers â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€


def _softmax_np(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax for 2D numpy array.

    Uses Rust accelerator for large batches when available.
    """
    if _RUST_NLI and x.size >= 100:
        flat = x.flatten().tolist()
        cols = x.shape[1]
        result = rust_softmax(flat, cols)
        return np.array(result, dtype=np.float64).reshape(x.shape)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    s: np.ndarray = e / e.sum(axis=1, keepdims=True)
    return s


def _resolve_label_indices(model) -> tuple[int, int]:
    """Read model.config.id2label to find contradiction and neutral indices.

    Returns (contradiction_idx, neutral_idx). Falls back to (2, 1) if
    id2label is missing or labels are unrecognisable.
    """
    id2label = getattr(getattr(model, "config", None), "id2label", None)
    if not id2label:
        return (2, 1)
    contra_idx = 2
    neutral_idx = 1
    for idx, label in id2label.items():
        normed = str(label).lower().strip()
        if normed in ("contradiction", "contradict"):
            contra_idx = int(idx)
        elif normed == "neutral":
            neutral_idx = int(idx)
    return (contra_idx, neutral_idx)


def _probs_to_divergence(
    probs: np.ndarray,
    label_indices: tuple[int, int] | None = None,
) -> list[float]:
    """Convert softmax rows to divergence scores.

    2-class: divergence = 1 - P(supported).
    3-class: divergence = P(contradiction) + 0.5 * P(neutral).
    ``label_indices`` is (contradiction_idx, neutral_idx) from
    ``_resolve_label_indices``; defaults to (2, 1).

    Uses Rust accelerator for large batches when available.
    """
    ncols = probs.shape[1]
    ci, ni = label_indices or (2, 1)
    if _RUST_NLI and probs.shape[0] >= 10:
        flat = probs.flatten().tolist()
        return [float(v) for v in rust_probs_to_divergence(flat, ncols, ci, ni)]
    if ncols == 2:
        return [float(1.0 - row[1]) for row in probs]
    return [float(row[ci]) + float(row[ni]) * 0.5 for row in probs]


def _probs_to_confidence(probs: np.ndarray) -> list[float]:
    """Convert softmax rows to confidence scores.

    Confidence = 1 - H(p)/log(K) where H is entropy and K is num classes.
    Returns values in [0, 1]: 1 = maximally confident (one-hot),
    0 = maximally uncertain (uniform).

    Uses Rust accelerator for large batches when available.
    """
    ncols = probs.shape[1]
    if _RUST_NLI and probs.shape[0] >= 10:
        flat = probs.flatten().tolist()
        return [float(v) for v in rust_probs_to_confidence(flat, ncols)]
    log_k = float(np.log(ncols)) if ncols > 1 else 1.0
    result: list[float] = []
    for row in probs:
        clipped = np.clip(row, 1e-10, 1.0)
        entropy = -float(np.sum(clipped * np.log(clipped)))
        result.append(max(0.0, 1.0 - entropy / log_k))
    return result


# â"€â"€ Model Loaders â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€


@lru_cache(maxsize=4)
def _load_nli_model(
    model_name: str = _DEFAULT_MODEL,
    quantize_8bit: bool = False,
    device: str | None = None,
    torch_dtype: str | None = None,
    revision: str | None = None,
):
    """Lazily load an NLI model + tokenizer (cached by model_name).

    Call ``clear_model_cache()`` to release GPU memory held by cached models.
    """
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        if device is None:
            from .._device import select_torch_device

            device = select_torch_device()
            if device.startswith("cuda"):
                logger.info("auto-selected GPU device %s", device)

        rev = _resolve_revision(model_name, revision)
        logger.info(
            "Loading NLI model: %s (device=%s, revision=%s)",
            model_name,
            device,
            rev[:12] if rev else "latest",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False, revision=rev
        )

        load_kwargs: dict = {}
        if torch_dtype:
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            load_kwargs["torch_dtype"] = dtype_map.get(torch_dtype, torch.float32)

        if quantize_8bit:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                load_kwargs["device_map"] = "auto"
                logger.info("Loading with 8-bit quantization")
            except ImportError:
                logger.warning(
                    "bitsandbytes not installed — loading without quantization",
                )

        load_kwargs.setdefault("low_cpu_mem_usage", False)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            revision=rev,
            **load_kwargs,
        )

        if device and "device_map" not in load_kwargs:
            model = model.to(device)

        model.eval()
        logger.info("NLI model loaded successfully.")
        return tokenizer, model
    except (ImportError, RuntimeError, OSError, ValueError) as e:
        logger.warning("NLI model unavailable: %s — using heuristic fallback", e)
        return None, None


def clear_model_cache() -> None:
    """Evict all cached NLI models to free GPU memory."""
    _load_nli_model.cache_clear()
    _load_onnx_session.cache_clear()
    logger.info("NLI model cache cleared")


def nli_available() -> bool:
    """Check whether torch + transformers are importable."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


# ── Scorer â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€


class NLIScorer:
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
        backend: str = "deberta",
        quantize_8bit: bool = False,
        device: str | None = None,
        torch_dtype: str | None = None,
        onnx_path: str | None = None,
        onnx_batch_size: int = 16,
        onnx_flush_timeout_ms: float = 10.0,
        cost_per_token: float = _DEFAULT_COST_PER_TOKEN,
        lora_adapter_path: str | None = None,
        revision: str | None = None,
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
        self._lite_scorer = None
        self._onnx_batcher: OnnxDynamicBatcher | None = None
        self._last_token_count: int = 0
        self._cost_per_token: float = cost_per_token
        # Label indices resolved from model.config.id2label after loading.
        # None = not yet resolved; tuple = (contradiction_idx, neutral_idx)
        self._label_indices: tuple[int, int] | None = None
        self._lora_adapter_path = lora_adapter_path
        self._revision = revision

    @property
    def _backend_ready(self) -> bool:
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
            assert self._model is not None  # guaranteed by caller
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
        return self._ensure_model()

    @property
    def last_token_count(self) -> int:
        return self._last_token_count

    @property
    def last_estimated_cost(self) -> float:
        return self._last_token_count * self._cost_per_token

    def reset_token_counter(self) -> None:
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

    # â"€â"€ MiniCheck backend â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

    def _ensure_minicheck(self) -> bool:
        if self._minicheck_loaded:
            return self._minicheck is not None
        self._minicheck_loaded = True
        try:  # pragma: no cover — requires minicheck package with model
            try:
                from minicheck import MiniCheck
            except ImportError:
                from minicheck.minicheck import MiniCheck

            try:
                self._minicheck = MiniCheck(
                    model_name="deberta-v3-large",
                    cache_dir=self._cache_dir,
                )
            except (RuntimeError, ValueError):
                # device_map="auto" fails on ROCm/older torch — load manually
                logger.info("MiniCheck device_map=auto failed, loading manually")
                self._minicheck = MiniCheck.__new__(MiniCheck)
                from minicheck.inference import Inferencer

                inf = Inferencer.__new__(Inferencer)
                inf.model_name = "deberta-v3-large"
                inf.max_model_len = 2048
                inf.batch_size = 16

                import torch
                from transformers import (
                    AutoConfig,
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )

                ckpt = "lytang/MiniCheck-DeBERTa-v3-Large"
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
                assert self._minicheck is not None  # narrowing for mypy
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
        if not getattr(self, "use_model", True) and not self._minicheck_loaded:
            return self._heuristic_score(premise, hypothesis)
        if not self._ensure_minicheck() or self._minicheck is None:
            return self._heuristic_score(premise, hypothesis)
        result = self._minicheck.score(docs=[premise], claims=[hypothesis])
        # MiniCheck returns (pred_labels, max_probs, sentences, prob_arrays)
        if isinstance(result, tuple):
            _, max_probs, *_ = result
            return float(1.0 - max_probs[0])
        return float(1.0 - result[0])

    def _minicheck_score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not getattr(self, "use_model", True) and not self._minicheck_loaded:
            return [self._heuristic_score(p, h) for p, h in pairs]
        if not self._ensure_minicheck() or self._minicheck is None:
            return [self._heuristic_score(p, h) for p, h in pairs]
        docs = [p for p, _ in pairs]
        claims = [h for _, h in pairs]
        result = self._minicheck.score(docs=docs, claims=claims)
        if isinstance(result, tuple):
            _, max_probs, *_ = result
            preds = max_probs
        else:
            preds = result
        return [float(1.0 - s) for s in preds]

    # â"€â"€ PyTorch backend â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

    @property
    def _is_factcg(self) -> bool:
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

    # â"€â"€ ONNX backend â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

    # â"€â"€ Chunked scoring â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split on sentence-ending punctuation + whitespace.

        Avoids splitting after abbreviations (e.g. U.S., Dr., etc.)
        and decimal numbers (e.g. 2.3%).
        """
        abbrev_re = re.compile(
            r"(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Inc|Ltd|Corp|vs|etc|e\.g|i\.e|U\.S|U\.K)\.\s+",
            re.IGNORECASE,
        )
        # Protect abbreviations with placeholder
        protected = abbrev_re.sub(
            lambda m: m.group().replace(". ", ".<NOSPLIT>"),
            text.strip(),
        )
        # Protect decimals: digit.digit
        protected = re.sub(r"(\d)\.(\d)", r"\1.<NOSPLIT>\2", protected)
        parts = re.split(r"(?<=[.!?])\s+", protected)
        return [s.replace("<NOSPLIT>", " ").strip() for s in parts if s.strip()]

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token count (~4 chars/token for English)."""
        return len(text) // 4 + 1

    def _build_chunks(
        self,
        sentences: list[str],
        budget: int,
        overlap_ratio: float = 0.0,
    ) -> list[str]:
        """Group sentences into chunks within *budget* tokens.

        When ``overlap_ratio > 0``, uses sliding-window overlap: after
        filling a chunk, the next chunk starts from the sentence at
        ``(1 - overlap_ratio) * chunk_length`` position. With the
        default ``overlap_ratio=0``, uses 1-sentence overlap (legacy).
        """
        if overlap_ratio > 0:
            return self._build_chunks_overlap(sentences, budget, overlap_ratio)

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = self._estimate_tokens(sent)
            if current and current_tokens + sent_tokens > budget:
                chunks.append(" ".join(current))
                current = [current[-1]]
                current_tokens = self._estimate_tokens(current[0])
            current.append(sent)
            current_tokens += sent_tokens

        if current:
            chunks.append(" ".join(current))
        return chunks or [" ".join(sentences)]

    def _build_chunks_overlap(
        self,
        sentences: list[str],
        budget: int,
        overlap_ratio: float,
    ) -> list[str]:
        """Sliding-window chunking with configurable token overlap."""
        chunks: list[str] = []
        i = 0
        while i < len(sentences):
            current: list[str] = []
            current_tokens = 0
            j = i
            while j < len(sentences):
                st = self._estimate_tokens(sentences[j])
                if current and current_tokens + st > budget:
                    break
                current.append(sentences[j])
                current_tokens += st
                j += 1

            if not current:
                current.append(sentences[i])
                j = i + 1

            chunks.append(" ".join(current))
            # Stride: advance by (1 - overlap_ratio) * num sentences in chunk
            stride = max(1, int(len(current) * (1.0 - overlap_ratio)))
            i += stride

        return chunks or [" ".join(sentences)]

    def _score_chunked_with_counts(
        self,
        premise: str,
        hypothesis: str,
        outer_agg: str = "max",
        inner_agg: str = "max",
        premise_ratio: float = 0.4,
        overlap_ratio: float = 0.0,
    ) -> tuple[float, list[float], int, int]:
        """Bidirectional chunked scoring with chunk counts.

        Returns (agg_score, per_hyp_scores, prem_count, hyp_count).

        inner_agg controls how premise-chunk scores are combined per
        hypothesis chunk: "max" (default, conservative — worst evidence
        wins), "min" (best evidence wins — use for summarization), or
        "mean".

        outer_agg controls how hypothesis-chunk scores are combined:
        "max" (default) or "mean".

        premise_ratio controls the token budget split between premise and
        hypothesis.  Default 0.4 (40% premise, 60% hypothesis) suits QA;
        use 0.85 for summarization where source documents are long and
        summary sentences are short.
        """
        hyp_budget = int(self.max_length * (1 - premise_ratio))
        prem_budget = int(self.max_length * premise_ratio)

        hyp_fits = self._estimate_tokens(hypothesis) <= hyp_budget
        prem_fits = self._estimate_tokens(premise) <= prem_budget

        if hyp_fits and prem_fits:
            s = self.score(premise, hypothesis)
            metrics.observe("nli_premise_chunks", 1)
            metrics.observe("nli_hypothesis_chunks", 1)
            return s, [s], 1, 1

        hyp_sents = self._split_sentences(hypothesis)
        hyp_chunks = (
            self._build_chunks(hyp_sents, hyp_budget, overlap_ratio)
            if not hyp_fits and len(hyp_sents) > 1
            else [hypothesis]
        )

        prem_sents = self._split_sentences(premise)
        prem_chunks = (
            self._build_chunks(prem_sents, prem_budget, overlap_ratio)
            if not prem_fits and len(prem_sents) > 1
            else [premise]
        )

        pairs = [(pc, hc) for pc in prem_chunks for hc in hyp_chunks]
        all_scores = self.score_batch(pairs)

        n_prem = len(prem_chunks)
        n_hyp = len(hyp_chunks)
        per_hyp: list[float] = []
        for h_idx in range(n_hyp):
            scores_h = [all_scores[p * n_hyp + h_idx] for p in range(n_prem)]
            if inner_agg == "min":
                per_hyp.append(min(scores_h))
            elif inner_agg == "mean":
                per_hyp.append(sum(scores_h) / len(scores_h))
            else:
                per_hyp.append(max(scores_h))

        if outer_agg == "max":
            agg = max(per_hyp)
        elif outer_agg == "trimmed_mean":
            # Drop top 25% of per-hypothesis scores (most divergent) before averaging.
            # More robust to outlier sentences that NLI can't match.
            sorted_scores = sorted(per_hyp)
            keep = max(1, int(len(sorted_scores) * 0.75))
            agg = sum(sorted_scores[:keep]) / keep
        else:
            agg = sum(per_hyp) / len(per_hyp)

        metrics.observe("nli_premise_chunks", n_prem)
        metrics.observe("nli_hypothesis_chunks", n_hyp)
        return agg, per_hyp, n_prem, n_hyp

    def score_chunked(
        self,
        premise: str,
        hypothesis: str,
        outer_agg: str = "max",
        inner_agg: str = "max",
        premise_ratio: float = 0.4,
        overlap_ratio: float = 0.0,
    ) -> tuple[float, list[float]]:
        """Bidirectional chunked scoring for long premises and hypotheses.

        Returns (aggregated_score, per_hypothesis_chunk_scores).
        """
        agg, per_hyp, _, _ = self._score_chunked_with_counts(
            premise,
            hypothesis,
            outer_agg=outer_agg,
            inner_agg=inner_agg,
            premise_ratio=premise_ratio,
            overlap_ratio=overlap_ratio,
        )
        return agg, per_hyp

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

    def score_chunked_confidence_weighted(
        self,
        premise: str,
        hypothesis: str,
        inner_agg: str = "max",
        premise_ratio: float = 0.4,
        overlap_ratio: float = 0.0,
    ) -> tuple[float, list[float]]:
        """Chunked scoring with confidence-weighted outer aggregation.

        Instead of max/mean over hypothesis chunks, weights each chunk's
        divergence by the model's confidence (1 - normalised entropy).
        Uncertain chunks contribute less to the aggregate.
        """
        hyp_budget = int(self.max_length * (1 - premise_ratio))
        prem_budget = int(self.max_length * premise_ratio)

        hyp_fits = self._estimate_tokens(hypothesis) <= hyp_budget
        prem_fits = self._estimate_tokens(premise) <= prem_budget

        if hyp_fits and prem_fits:
            result = self.score_batch_with_confidence([(premise, hypothesis)])
            s, _ = result[0]
            return s, [s]

        hyp_sents = self._split_sentences(hypothesis)
        hyp_chunks = (
            self._build_chunks(hyp_sents, hyp_budget, overlap_ratio)
            if not hyp_fits and len(hyp_sents) > 1
            else [hypothesis]
        )

        prem_sents = self._split_sentences(premise)
        prem_chunks = (
            self._build_chunks(prem_sents, prem_budget, overlap_ratio)
            if not prem_fits and len(prem_sents) > 1
            else [premise]
        )

        pairs = [(pc, hc) for pc in prem_chunks for hc in hyp_chunks]
        results_with_conf = self.score_batch_with_confidence(pairs)

        n_prem = len(prem_chunks)
        n_hyp = len(hyp_chunks)
        per_hyp: list[float] = []
        per_hyp_conf: list[float] = []

        for h_idx in range(n_hyp):
            scores_h = [results_with_conf[p * n_hyp + h_idx] for p in range(n_prem)]
            divs = [s[0] for s in scores_h]
            if inner_agg == "min":
                per_hyp.append(min(divs))
            elif inner_agg == "mean":
                per_hyp.append(sum(divs) / len(divs))
            else:
                per_hyp.append(max(divs))
            avg_conf = sum(s[1] for s in scores_h) / len(scores_h)
            per_hyp_conf.append(avg_conf)

        # Confidence-weighted mean
        total_weight = sum(per_hyp_conf)
        if total_weight > 1e-9:
            agg = (
                sum(d * c for d, c in zip(per_hyp, per_hyp_conf, strict=True))
                / total_weight
            )
        else:
            agg = sum(per_hyp) / len(per_hyp)

        return agg, per_hyp

    # â"€â"€ Claim decomposition â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

    def decompose_claims(self, text: str) -> list[str]:
        """Split text into individual claim sentences."""
        return self._split_sentences(text)

    def score_decomposed(
        self,
        premise: str,
        hypothesis: str,
    ) -> tuple[float, list[float]]:
        """Score each claim in hypothesis independently against premise.

        Returns (max_score, per_claim_scores).
        """
        claims = self.decompose_claims(hypothesis)
        if not claims:
            return self.score(premise, hypothesis), [self.score(premise, hypothesis)]

        if len(claims) == 1:
            s = self.score(premise, claims[0])
            return s, [s]

        pairs = [(premise, c) for c in claims]
        scores = self.score_batch(pairs)
        return max(scores), scores

    def score_claim_coverage(
        self,
        source: str,
        summary: str,
        support_threshold: float = 0.6,
    ) -> tuple[float, list[float], list[str]]:
        """Decompose summary into claims and compute coverage against source.

        A claim is "supported" when its NLI divergence < support_threshold.
        Coverage = supported_claims / total_claims.

        For long sources, each claim is scored with chunked NLI so that
        at least one source chunk can provide evidence.

        Returns (coverage, per_claim_divergences, claims).
        """
        claims = self.decompose_claims(summary)
        if not claims:
            s = self.score(source, summary)
            return float(s < support_threshold), [s], [summary]

        # Score each claim against the full source via chunked NLI.
        # inner_agg="min" picks the best-matching source chunk per claim.
        divs: list[float] = []
        for claim in claims:
            div, _ = self.score_chunked(
                source,
                claim,
                inner_agg="min",
                outer_agg="mean",
                premise_ratio=0.85,
            )
            divs.append(div)

        supported = sum(1 for d in divs if d < support_threshold)
        coverage = supported / len(claims)
        return coverage, divs, claims

    def score_claim_coverage_with_attribution(
        self,
        source: str,
        summary: str,
        support_threshold: float = 0.6,
    ) -> tuple[float, list[float], list[str], list]:
        """Like score_claim_coverage but also returns sentence-level attributions.

        For each claim, finds the source sentence with lowest divergence
        (best evidence match). Returns list of ClaimAttribution objects.
        """
        from ..types import ClaimAttribution

        claims = self.decompose_claims(summary)
        source_sents = self._split_sentences(source)

        if not claims:
            s = self.score(source, summary)
            attr = [
                ClaimAttribution(
                    claim=summary,
                    claim_index=0,
                    source_sentence=source_sents[0] if source_sents else source,
                    source_index=0,
                    divergence=s,
                    supported=s < support_threshold,
                ),
            ]
            return float(s < support_threshold), [s], [summary], attr

        if not source_sents:
            source_sents = [source]

        max_attribution_pairs = 10_000
        n_pairs = len(claims) * len(source_sents)
        if n_pairs > max_attribution_pairs:
            raise ValueError(
                f"Attribution would create {n_pairs} pairs "
                f"({len(claims)} claims Ă— {len(source_sents)} source sentences), "
                f"exceeding limit of {max_attribution_pairs}",
            )

        pairs = [(src_s, claim) for claim in claims for src_s in source_sents]
        all_divs = self.score_batch(pairs)

        n_src = len(source_sents)
        per_claim_divs: list[float] = []
        attributions: list[ClaimAttribution] = []

        for c_idx, claim in enumerate(claims):
            # Slice this claim's scores across all source sentences
            claim_scores = all_divs[c_idx * n_src : (c_idx + 1) * n_src]
            best_idx = int(np.argmin(claim_scores))
            best_div = claim_scores[best_idx]
            per_claim_divs.append(best_div)

            attributions.append(
                ClaimAttribution(
                    claim=claim,
                    claim_index=c_idx,
                    source_sentence=source_sents[best_idx],
                    source_index=best_idx,
                    divergence=best_div,
                    supported=best_div < support_threshold,
                ),
            )

        supported = sum(1 for d in per_claim_divs if d < support_threshold)
        coverage = supported / len(claims)
        return coverage, per_claim_divs, claims, attributions

    # â"€â"€ Lite backend â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

    def _ensure_lite(self):
        if self._lite_scorer is None:
            from .lite_scorer import LiteScorer

            self._lite_scorer = LiteScorer()  # type: ignore[assignment]

    def _lite_score(self, premise: str, hypothesis: str) -> float:
        self._ensure_lite()
        return self._lite_scorer.score(premise, hypothesis)  # type: ignore[attr-defined, no-any-return]

    def _lite_score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        self._ensure_lite()
        return self._lite_scorer.score_batch(pairs)  # type: ignore[attr-defined, no-any-return]

    # â"€â"€ Heuristic fallback â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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
