# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — NLI ONNX/TensorRT Export & Dynamic Batcher
"""ONNX export, TensorRT engine cache, and dynamic batching.

Extracted from nli.py to reduce module size.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import numpy as np

# FactCG instruction template (duplicated from nli.py to avoid circular import)
_FACTCG_TEMPLATE = (
    "{text_a}\n\nChoose your answer: based on the paragraph above "
    'can we conclude that "{text_b}"?\n\nOPTIONS:\n- Yes\n- No\n'
    "I think the answer is "
)

_DEFAULT_MODEL = "yaxili96/FactCG-DeBERTa-v3-Large"

logger = logging.getLogger("DirectorAI.NLI")


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

        # Prefer quantized model on CPU
        quantized = os.path.join(onnx_path, "model_quantized.onnx")
        model_file = os.path.join(onnx_path, "model.onnx")
        if os.path.exists(quantized) and (device is None or "cpu" in (device or "")):
            model_file = quantized
            logger.info("Using INT8 quantized model: %s", quantized)
        elif not os.path.exists(model_file):
            for f in os.listdir(onnx_path):  # pragma: no branch
                if f.endswith(".onnx"):  # pragma: no branch
                    model_file = os.path.join(onnx_path, f)
                    break

        providers: list[str | tuple[str, dict[str, object]]] = [
            "CPUExecutionProvider",
        ]
        available = ort.get_available_providers()
        if (device and "cuda" in device) or "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")

        trt_cache = os.path.join(onnx_path, "trt_cache")
        trt_requested = os.environ.get("DIRECTOR_ENABLE_TRT") == "1"
        trt_cache_exists = os.path.isdir(trt_cache)
        if (trt_requested or trt_cache_exists) and (
            "TensorrtExecutionProvider" in available
        ):
            trt_opts: dict[str, object] = {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": trt_cache,
                "trt_fp16_enable": True,
                "trt_max_workspace_size": 1 << 30,
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


def export_onnx(
    model_name: str = _DEFAULT_MODEL,
    output_dir: str = "factcg_onnx",
    quantize: str | None = None,
) -> str:
    """Export NLI model to ONNX format, optionally quantized.

    Parameters
    ----------
    quantize : "int8", "fp16", or None (default FP32).

    Requires ``pip install optimum[onnxruntime]``.
    For int8: also requires ``onnxruntime`` quantization module.
    Load the result with
    ``NLIScorer(backend="onnx", onnx_path=<returned_dir>)``.

    """
    from optimum.onnxruntime import (
        ORTModelForSequenceClassification,
    )
    from transformers import AutoTokenizer

    model = ORTModelForSequenceClassification.from_pretrained(model_name, export=True)
    model.save_pretrained(output_dir)
    AutoTokenizer.from_pretrained(model_name).save_pretrained(output_dir)
    logger.info("ONNX model exported to %s", output_dir)

    if quantize == "int8":
        from onnxruntime.quantization import QuantType, quantize_dynamic

        src = os.path.join(output_dir, "model.onnx")
        dst = os.path.join(output_dir, "model_quantized.onnx")
        quantize_dynamic(src, dst, weight_type=QuantType.QInt8)
        logger.info("INT8 quantized model saved to %s", dst)
    elif quantize == "fp16":
        import onnx
        from onnxruntime.transformers import float16

        src = os.path.join(output_dir, "model.onnx")
        model_fp32 = onnx.load(src)
        model_fp16 = float16.convert_float_to_float16(model_fp32)
        dst = os.path.join(output_dir, "model_fp16.onnx")
        onnx.save(model_fp16, dst)
        logger.info("FP16 model saved to %s", dst)

    return output_dir


def export_tensorrt(
    onnx_dir: str = "factcg_onnx",
    output_dir: str | None = None,
    fp16: bool = True,
    max_batch: int = 16,
    max_seq_len: int = 512,
    warmup_pairs: int = 4,
) -> str:
    """Pre-build TensorRT engine cache from an exported ONNX model.

    Runs a warmup pass through ORT's TensorrtExecutionProvider to trigger
    engine compilation and cache it to disk. Subsequent loads skip the
    multi-minute cold-start.

    Parameters
    ----------
    onnx_dir : path to directory with model.onnx + tokenizer files.
    output_dir : where to write the TRT cache (default: <onnx_dir>/trt_cache).
    fp16 : enable FP16 mode (default True, ~2x speedup on Ada/Ampere).
    max_batch : max batch size for engine optimization profile.
    max_seq_len : max sequence length for optimization profile.
    warmup_pairs : number of dummy pairs to run for engine build.

    Requires ``pip install onnxruntime-gpu tensorrt``.

    """
    import onnxruntime as ort

    if output_dir is None:
        output_dir = os.path.join(onnx_dir, "trt_cache")
    os.makedirs(output_dir, exist_ok=True)

    model_file = os.path.join(onnx_dir, "model.onnx")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"ONNX model not found: {model_file}")

    available = ort.get_available_providers()
    if "TensorrtExecutionProvider" not in available:
        raise RuntimeError(
            "TensorrtExecutionProvider not available. "
            "Install onnxruntime-gpu and tensorrt.",
        )

    # ORT TensorRT profile shapes: batch x seq_len
    min_shape = "1x1"
    opt_shape = f"{max(1, max_batch // 2)}x{max_seq_len}"
    max_shape = f"{max_batch}x{max_seq_len}"
    trt_opts: dict[str, object] = {
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": output_dir,
        "trt_fp16_enable": fp16,
        "trt_max_workspace_size": 1 << 30,  # 1 GB
        "trt_profile_min_shapes": f"input_ids={min_shape},attention_mask={min_shape}",
        "trt_profile_opt_shapes": f"input_ids={opt_shape},attention_mask={opt_shape}",
        "trt_profile_max_shapes": f"input_ids={max_shape},attention_mask={max_shape}",
    }
    providers: list[str | tuple[str, dict[str, object]]] = [
        ("TensorrtExecutionProvider", trt_opts),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 2  # show TRT build progress
    session = ort.InferenceSession(model_file, opts, providers=providers)

    active = session.get_providers()[0]
    logger.info("TRT export session using: %s", active)

    # Warmup pass to trigger engine compilation
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
    dummy_pairs = [
        ("The sky is blue due to Rayleigh scattering.", "The sky is blue."),
    ] * warmup_pairs

    texts = [_FACTCG_TEMPLATE.format(text_a=p, text_b=h) for p, h in dummy_pairs]
    inputs = tokenizer(
        texts,
        return_tensors="np",
        truncation=True,
        padding=True,
        max_length=max_seq_len,
    )
    expected = {i.name for i in session.get_inputs()}
    feed = {
        k: v.astype(np.int64) if v.dtype != np.int64 else v
        for k, v in inputs.items()
        if k in expected
    }
    session.run(None, feed)
    logger.info("TRT engine cache built at %s", output_dir)
    return output_dir


# â”€â”€ ONNX Dynamic Batcher â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class OnnxDynamicBatcher:
    """Accumulate ONNX inference pairs and flush as a single batch.

    Flushes when *max_batch* is reached or *flush_timeout_ms* elapses
    (whichever comes first). Uses ORT IO binding for zero-copy GPU
    transfers when the CUDA provider is active.

    Parameters
    ----------
    onnx_scorer_fn : callable — function(pairs) -> list[float].
    max_batch : int — flush after this many pairs.
    flush_timeout_ms : float — flush after this many ms idle.
    session : ort.InferenceSession | None — for IO binding detection.

    """

    def __init__(
        self,
        onnx_scorer_fn,
        max_batch: int = 16,
        flush_timeout_ms: float = 10.0,
        session=None,
    ) -> None:
        import threading

        self._score_fn = onnx_scorer_fn
        self.max_batch = max_batch
        self.flush_timeout_ms = flush_timeout_ms
        self._session = session
        self._buffer: list[tuple[str, str]] = []
        self._results: list[float] = []
        self._lock = threading.Lock()
        self._has_cuda = False
        if session is not None:
            try:
                providers = session.get_providers()
                self._has_cuda = any("CUDA" in p for p in providers)
            except (AttributeError, RuntimeError):  # pragma: no cover
                pass

    def submit(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Submit pairs for batched scoring. Flushes immediately if batch full."""
        with self._lock:
            self._buffer.extend(pairs)
            if len(self._buffer) >= self.max_batch:
                return self._flush()
            return []

    def flush(self) -> list[float]:
        """Explicitly drain the buffer."""
        with self._lock:
            return self._flush()

    def _flush(self) -> list[float]:
        if not self._buffer:
            return []
        batch = self._buffer[:]
        self._buffer.clear()
        return self._score_fn(batch)  # type: ignore[no-any-return]

    @property
    def uses_io_binding(self) -> bool:
        return self._has_cuda
