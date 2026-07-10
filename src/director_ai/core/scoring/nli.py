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

import hashlib
import logging
import os
import re
import warnings
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from ..types import ClaimAttribution
    from .backends import ScorerBackend

from ..mandatory import mandatory_execution
from ..metrics import metrics
from ..model_revisions import (
    DEFAULT_NLI_MODEL,
    DEFAULT_NLI_MODEL_REVISION,
    MODEL_REVISION_REGISTRY,
    resolve_model_revision,
)
from . import _nli_accel
from ._nli_chunking import ChunkingMixin
from ._nli_export import (
    OnnxDynamicBatcher,
    _load_onnx_session,
    _validate_onnx_scheduler_config,
    export_onnx,
    export_tensorrt,
)
from ._nli_numeric import (
    _count_below_threshold,
    _probs_to_divergence,
    _resolve_label_indices,
    _softmax_np,
)
from ._nli_numeric import (
    _mean_float as _mean_float,
)
from ._nli_numeric import (
    _probs_to_confidence as _probs_to_confidence,
)
from ._nli_numeric import (
    _sum_float_list as _sum_float_list,
)
from ._nli_numeric import (
    _weighted_sum_float as _weighted_sum_float,
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

_DEFAULT_MODEL = DEFAULT_NLI_MODEL
_DEFAULT_MODEL_REVISION = DEFAULT_NLI_MODEL_REVISION
_RECOMMENDED_MODEL = "lytang/MiniCheck-DeBERTa-L"

#: Pinned revisions for all known HuggingFace Hub models used in
#: production. Prevents supply-chain drift when upstream pushes a
#: new commit.  Update SHAs deliberately after verifying the new
#: revision against the AggreFact benchmark.
MODEL_REGISTRY: dict[str, str] = MODEL_REVISION_REGISTRY

_SKIP_ARTIFACT_FILENAMES = {
    "optimizer.pt",
    "rng_state.pth",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
}


def _resolve_revision(model_name: str, revision: str | None = None) -> str | None:
    """Return an immutable revision SHA for *model_name*.

    If *revision* is already set by the caller, it takes precedence.
    """
    return resolve_model_revision(model_name, revision)


def _split_gs_uri(uri: str) -> tuple[str, str]:
    """Split a ``gs://bucket/prefix`` URI into bucket and prefix."""
    if not uri.startswith("gs://"):
        raise ValueError(f"not a GCS URI: {uri!r}")
    bucket_and_prefix = uri[5:]
    bucket, sep, prefix = bucket_and_prefix.partition("/")
    if not bucket or not sep or not prefix.strip("/"):
        raise ValueError(f"GCS URI must include bucket and prefix: {uri!r}")
    return bucket, prefix.strip("/")


def _safe_cache_name(uri: str) -> str:
    """Return a filesystem-safe deterministic cache directory name."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", uri[5:] if uri.startswith("gs://") else uri)
    slug = slug.strip("-")[:80] or "model"
    digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:16]
    return f"{slug}-{digest}"


def _should_skip_artifact(rel_path: str) -> bool:
    """Return whether a managed model artifact file should be ignored."""
    parts = Path(rel_path).parts
    if any(part.startswith("checkpoint-") for part in parts):
        return True
    return Path(rel_path).name in _SKIP_ARTIFACT_FILENAMES


def _download_gcs_model_artifact(uri: str) -> str:
    """Download a managed scorer artefact to a local Transformers cache."""
    bucket_name, prefix = _split_gs_uri(uri)
    cache_root = Path(
        os.environ.get("DIRECTOR_MODEL_CACHE_DIR")
        or os.environ.get("HF_HOME")
        or "~/.cache/huggingface",
    ).expanduser()
    target_dir = cache_root / "director-ai-scorers" / _safe_cache_name(uri)
    marker = target_dir / ".director-ai-complete"
    if marker.exists():
        return str(target_dir)

    try:
        import google.cloud.storage as storage  # type: ignore[import-untyped]  # google-cloud-storage lacks py.typed metadata in this venv.
    except ImportError as exc:
        raise RuntimeError(
            "loading managed scorer artefacts requires google-cloud-storage; "
            "install director-ai[managed-training]",
        ) from exc

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    downloaded = 0
    target_dir.mkdir(parents=True, exist_ok=True)
    for blob in client.list_blobs(bucket, prefix=f"{prefix}/"):
        rel_path = blob.name[len(prefix) :].lstrip("/")
        if not rel_path or _should_skip_artifact(rel_path):
            continue
        out_path = target_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(out_path))
        downloaded += 1

    if downloaded == 0:
        raise FileNotFoundError(f"no model artefact files found at {uri}")
    marker.write_text(f"{uri}\n", encoding="utf-8")
    return str(target_dir)


def _resolve_model_source(model_name: str) -> str:
    """Return a local model source path, downloading GCS artifacts when needed."""
    if model_name.startswith("gs://"):
        return _download_gcs_model_artifact(model_name)
    return model_name


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


# ── Model Loaders ───────────────────────────────────────────────


@lru_cache(maxsize=4)
def _load_nli_model(
    model_name: str = _DEFAULT_MODEL,
    quantize_8bit: bool = False,
    device: str | None = None,
    torch_dtype: str | None = None,
    revision: str | None = None,
) -> tuple[Any, Any]:
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

        model_source = _resolve_model_source(model_name)
        rev = (
            None
            if model_source != model_name
            else _resolve_revision(model_name, revision)
        )
        logger.info(
            "Loading NLI model: %s (device=%s, revision=%s)",
            model_name,
            device,
            rev[:12] if rev else "latest",
        )
        # Third-party transformers DeBERTa import path currently emits
        # torch.jit.script deprecation warnings. Suppress only that exact
        # warning while bootstrapping the model.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"`torch\.jit\.script` is deprecated\..*",
                category=DeprecationWarning,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_source, use_fast=False, revision=rev
            )

            load_kwargs: dict[str, Any] = {}
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

                    bits_and_bytes_config = cast(type[Any], BitsAndBytesConfig)
                    load_kwargs["quantization_config"] = bits_and_bytes_config(
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
                model_source,
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
    from .._device import release_torch_cuda

    release_torch_cuda()
    logger.info("NLI model cache cleared")


def nli_available() -> bool:
    """Check whether torch + transformers are importable.

    ``importlib.util.find_spec`` raises :class:`ValueError` when a
    test mocks a module via ``sys.modules`` without setting
    ``__spec__``; in that case the caller has deliberately
    installed a fake, so treat the package as available.
    """
    import importlib.util

    for name in ("torch", "transformers"):
        try:
            if importlib.util.find_spec(name) is None:
                return False
        except (ImportError, ValueError):
            # ValueError = mocked module without __spec__ — trust
            # the test-time injection and continue.
            continue
    return True


# ── Scorer ───────────────────────────────────────────────────────


class NLIScorer(ChunkingMixin):
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

    # ── Claim decomposition ────────────────────────────────────────

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

        if _nli_accel._RUST_NLI:
            with mandatory_execution(__name__, component="mandatory accelerated path"):
                coverage, _supported = _nli_accel.rust_coverage_from_divergences(
                    [float(d) for d in divs],
                    float(support_threshold),
                )
                return float(coverage), divs, claims
        supported = _count_below_threshold(divs, support_threshold)
        coverage = supported / len(claims)
        return coverage, divs, claims

    def score_claim_coverage_with_attribution(
        self,
        source: str,
        summary: str,
        support_threshold: float = 0.6,
    ) -> tuple[float, list[float], list[str], list[ClaimAttribution]]:
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
        if _nli_accel._RUST_NLI:
            try:
                per_claim_divs, best_indices = _nli_accel.rust_reduce_claim_attribution(
                    [float(v) for v in all_divs],
                    len(claims),
                    n_src,
                )
            except Exception:
                per_claim_divs, best_indices = [], []
        else:
            per_claim_divs, best_indices = [], []

        if not per_claim_divs:
            per_claim_divs = []
            best_indices = []
            for c_idx in range(len(claims)):
                claim_scores = all_divs[c_idx * n_src : (c_idx + 1) * n_src]
                best_idx = int(np.argmin(claim_scores))
                per_claim_divs.append(claim_scores[best_idx])
                best_indices.append(best_idx)

        attributions: list[ClaimAttribution] = []
        for c_idx, claim in enumerate(claims):
            best_idx = int(best_indices[c_idx])
            best_div = float(per_claim_divs[c_idx])
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

        if _nli_accel._RUST_NLI:
            with mandatory_execution(__name__, component="mandatory accelerated path"):
                coverage, _supported = _nli_accel.rust_coverage_from_divergences(
                    [float(d) for d in per_claim_divs],
                    float(support_threshold),
                )
                return float(coverage), per_claim_divs, claims, attributions
        supported = _count_below_threshold(per_claim_divs, support_threshold)
        coverage = supported / len(claims)
        return coverage, per_claim_divs, claims, attributions

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
