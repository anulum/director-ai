# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Real NLI Backend (DeBERTa)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
NLI-based logical divergence scorer with batched inference and ONNX.

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

from .metrics import metrics

logger = logging.getLogger("DirectorAI.NLI")

_DEFAULT_MODEL = "yaxili96/FactCG-DeBERTa-v3-Large"
_RECOMMENDED_MODEL = "lytang/MiniCheck-DeBERTa-L"

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


# ── Helpers ──────────────────────────────────────────────────────


def _softmax_np(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax for 2D numpy array."""
    e = np.exp(x - x.max(axis=1, keepdims=True))
    s: np.ndarray = e / e.sum(axis=1, keepdims=True)
    return s


def _probs_to_divergence(probs: np.ndarray) -> list[float]:
    """Convert softmax rows to divergence scores.

    2-class: divergence = 1 - P(supported).
    3-class: divergence = P(contradiction) + 0.5 * P(neutral).
    """
    ncols = probs.shape[1]
    if ncols == 2:
        return [float(1.0 - row[1]) for row in probs]
    return [float(row[2]) + float(row[1]) * 0.5 for row in probs]


# ── Model Loaders ───────────────────────────────────────────────


@lru_cache(maxsize=4)
def _load_nli_model(
    model_name: str = _DEFAULT_MODEL,
    quantize_8bit: bool = False,
    device: str | None = None,
    torch_dtype: str | None = None,
):
    """Lazily load an NLI model + tokenizer (cached by model_name)."""
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        logger.info("Loading NLI model: %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

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
                    load_in_8bit=True
                )
                load_kwargs["device_map"] = "auto"
                logger.info("Loading with 8-bit quantization")
            except ImportError:
                logger.warning(
                    "bitsandbytes not installed — loading without quantization"
                )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, **load_kwargs
        )

        if device and "device_map" not in load_kwargs:
            model = model.to(device)

        model.eval()
        logger.info("NLI model loaded successfully.")
        return tokenizer, model
    except (ImportError, RuntimeError, OSError, ValueError) as e:
        logger.warning("NLI model unavailable: %s — using heuristic fallback", e)
        return None, None


@lru_cache(maxsize=4)
def _load_onnx_session(
    onnx_path: str,
    device: str | None = None,
):
    """Load ONNX Runtime session + tokenizer from exported directory."""
    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        if not os.path.isdir(onnx_path):
            raise FileNotFoundError(f"Not a directory: {onnx_path}")
        tokenizer = AutoTokenizer.from_pretrained(onnx_path)

        model_file = os.path.join(onnx_path, "model.onnx")
        if not os.path.exists(model_file):
            for f in os.listdir(onnx_path):
                if f.endswith(".onnx"):
                    model_file = os.path.join(onnx_path, f)
                    break

        providers: list[str | tuple[str, dict[str, object]]] = [
            "CPUExecutionProvider",
        ]
        available = ort.get_available_providers()
        if (device and "cuda" in device) or "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")

        trt_requested = os.environ.get("DIRECTOR_ENABLE_TRT") == "1"
        if trt_requested and "TensorrtExecutionProvider" in available:
            trt_opts: dict[str, object] = {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": os.path.join(onnx_path, "trt_cache"),
                "trt_fp16_enable": True,
            }
            providers.insert(0, ("TensorrtExecutionProvider", trt_opts))

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3  # suppress Memcpy transformer warnings
        session = ort.InferenceSession(model_file, opts, providers=providers)
        logger.info(
            "ONNX session: %s (%s)",
            model_file,
            session.get_providers()[0],
        )
        return tokenizer, session
    except (
        ImportError,
        RuntimeError,
        OSError,
        FileNotFoundError,
    ) as e:
        logger.warning("ONNX session unavailable: %s", e)
        return None, None


def nli_available() -> bool:
    """Check whether torch + transformers are importable."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def export_onnx(
    model_name: str = _DEFAULT_MODEL,
    output_dir: str = "factcg_onnx",
) -> str:
    """Export NLI model to ONNX format.

    Requires ``pip install optimum[onnxruntime]``.
    Load the result with
    ``NLIScorer(backend="onnx", onnx_path=output_dir)``.
    """
    from optimum.onnxruntime import (
        ORTModelForSequenceClassification,
    )
    from transformers import AutoTokenizer

    model = ORTModelForSequenceClassification.from_pretrained(model_name, export=True)
    model.save_pretrained(output_dir)
    AutoTokenizer.from_pretrained(model_name).save_pretrained(output_dir)
    logger.info("ONNX model exported to %s", output_dir)
    return output_dir


# ── Scorer ───────────────────────────────────────────────────────


class NLIScorer:
    """NLI-based logical divergence scorer.

    Parameters
    ----------
    use_model : bool — attempt to load model on first score().
    max_length : int — max token length for NLI input.
    model_name : str | None — HuggingFace model ID or local path.
    backend : str — "deberta", "onnx", or "minicheck".
    quantize_8bit : bool — 8-bit quantization (requires bitsandbytes).
    device : str | None — torch device ("cpu", "cuda", "cuda:0").
    torch_dtype : str | None — "float16", "bfloat16", or "float32".
    onnx_path : str | None — directory with exported ONNX model.
    """

    _BACKENDS = ("deberta", "minicheck", "onnx")

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
    ) -> None:
        if backend not in self._BACKENDS:
            raise ValueError(
                f"backend must be one of {self._BACKENDS}, got {backend!r}"
            )
        self.use_model = use_model
        self.max_length = max_length
        self.backend = backend
        self._model_name = model_name or _DEFAULT_MODEL
        self._quantize_8bit = quantize_8bit
        self._device = device
        self._torch_dtype = torch_dtype
        self._onnx_path = onnx_path
        self._tokenizer = None
        self._model = None
        self._onnx_session = None
        self._model_loaded = False
        self._minicheck = None
        self._minicheck_loaded = False

    @property
    def _backend_ready(self) -> bool:
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
                    "onnx backend requires onnx_path — falling back to heuristic"
                )
                self._model_loaded = True
                return False
            self._tokenizer, self._onnx_session = _load_onnx_session(
                self._onnx_path, device=self._device
            )
        else:
            self._tokenizer, self._model = _load_nli_model(
                self._model_name,
                quantize_8bit=self._quantize_8bit,
                device=self._device,
                torch_dtype=self._torch_dtype,
            )
        self._model_loaded = True
        return self._backend_ready

    @property
    def model_available(self) -> bool:
        return self._ensure_model()

    def score(self, premise: str, hypothesis: str) -> float:
        """Compute logical divergence between premise and hypothesis.

        Returns float in [0, 1]: 0 = entailment, 1 = contradiction.
        """
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
        if self._minicheck_loaded:
            return self._minicheck is not None
        self._minicheck_loaded = True
        try:
            from minicheck import MiniCheck

            self._minicheck = MiniCheck(model_name="MiniCheck-DeBERTa-L")
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
            return False

    def _minicheck_score(self, premise: str, hypothesis: str) -> float:
        if not self._ensure_minicheck() or self._minicheck is None:
            return self._heuristic_score(premise, hypothesis)
        pred = self._minicheck.score(docs=[premise], claims=[hypothesis])
        return float(1.0 - pred[0])

    def _minicheck_score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not self._ensure_minicheck() or self._minicheck is None:
            return [self._heuristic_score(p, h) for p, h in pairs]
        docs = [p for p, _ in pairs]
        claims = [h for _, h in pairs]
        preds = self._minicheck.score(docs=docs, claims=claims)
        return [float(1.0 - s) for s in preds]

    # ── PyTorch backend ──────────────────────────────────────────

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

            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self._model(**inputs).logits

            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        if len(probs) == 2:
            return float(1.0 - probs[1])
        return float(probs[2]) + float(probs[1]) * 0.5

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

            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self._model(**inputs).logits

            probs = torch.softmax(logits, dim=1).cpu().numpy()

        return _probs_to_divergence(probs)

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

            # Feed only inputs the ONNX graph expects, cast to int64
            expected = {i.name for i in self._onnx_session.get_inputs()}
            feed = {
                k: v.astype(np.int64) if v.dtype != np.int64 else v
                for k, v in inputs.items()
                if k in expected
            }
            logits = self._onnx_session.run(None, feed)[0]

        return _probs_to_divergence(_softmax_np(logits))

    # ── Chunked scoring ──────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split on sentence-ending punctuation + whitespace."""
        return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token count (~4 chars/token for English)."""
        return len(text) // 4 + 1

    def _build_chunks(self, sentences: list[str], budget: int) -> list[str]:
        """Group sentences into chunks within *budget* tokens,
        1-sentence overlap."""
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

    def _score_chunked_with_counts(
        self,
        premise: str,
        hypothesis: str,
        outer_agg: str = "max",
    ) -> tuple[float, list[float], int, int]:
        """Bidirectional chunked scoring with chunk counts.

        Returns (agg_score, per_hyp_scores, prem_count, hyp_count).
        Inner aggregation (across premise chunks per hypothesis chunk): min.
        Outer aggregation (across hypothesis chunks): max (default) or mean.
        """
        hyp_budget = int(self.max_length * 0.6)
        prem_budget = int(self.max_length * 0.4)

        hyp_fits = self._estimate_tokens(hypothesis) <= hyp_budget
        prem_fits = self._estimate_tokens(premise) <= prem_budget

        if hyp_fits and prem_fits:
            s = self.score(premise, hypothesis)
            metrics.observe("nli_premise_chunks", 1)
            metrics.observe("nli_hypothesis_chunks", 1)
            return s, [s], 1, 1

        hyp_sents = self._split_sentences(hypothesis)
        hyp_chunks = (
            self._build_chunks(hyp_sents, hyp_budget)
            if not hyp_fits and len(hyp_sents) > 1
            else [hypothesis]
        )

        prem_sents = self._split_sentences(premise)
        prem_chunks = (
            self._build_chunks(prem_sents, prem_budget)
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
            per_hyp.append(min(scores_h))

        agg = max(per_hyp) if outer_agg == "max" else sum(per_hyp) / len(per_hyp)

        metrics.observe("nli_premise_chunks", n_prem)
        metrics.observe("nli_hypothesis_chunks", n_hyp)
        return agg, per_hyp, n_prem, n_hyp

    def score_chunked(
        self,
        premise: str,
        hypothesis: str,
        outer_agg: str = "max",
    ) -> tuple[float, list[float]]:
        """Bidirectional chunked scoring for long premises and hypotheses.

        Returns (aggregated_score, per_hypothesis_chunk_scores).
        """
        agg, per_hyp, _, _ = self._score_chunked_with_counts(
            premise,
            hypothesis,
            outer_agg,
        )
        return agg, per_hyp

    # ── Heuristic fallback ───────────────────────────────────────

    @staticmethod
    def _heuristic_score(premise: str, hypothesis: str) -> float:
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
        return float(np.clip(raw, _DIVERGENCE_ALIGNED, _DIVERGENCE_CONTRADICTED))
