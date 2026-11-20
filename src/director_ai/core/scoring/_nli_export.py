# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — NLI ONNX/TensorRT Export & Dynamic Batcher
"""ONNX export, TensorRT engine cache, and dynamic batching.

Extracted from nli.py to reduce module size.

An export directory written by :func:`export_onnx` carries a manifest
(:data:`ONNX_MANIFEST_FILENAME`) that names the served model file, its
SHA-256 digest, its precision and, for INT8, the quantisation range
policy. Loaders serve exactly that file: a missing bound file or a digest
mismatch is refused, and the verified bytes are what ONNX Runtime loads.
A full-range INT8 model on an x86 CPU without VNNI or AMX is warned about,
or refused in strict mode. Load-policy refusals (``PermissionError``) are
never turned into an unavailable backend.

Directories without a manifest still load, with one warning per directory.
``DIRECTOR_ONNX_REQUIRE_MANIFEST=1`` refuses them and also refuses a
full-range INT8 model whose CPU support cannot be confirmed.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
import os
import platform
import re
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

# FactCG instruction template (duplicated from nli.py to avoid circular import)
_FACTCG_TEMPLATE = (
    "{text_a}\n\nChoose your answer: based on the paragraph above "
    'can we conclude that "{text_b}"?\n\nOPTIONS:\n- Yes\n- No\n'
    "I think the answer is "
)

_DEFAULT_MODEL = "yaxili96/FactCG-DeBERTa-v3-Large"
_LOCAL_ARTIFACT_REVISION = "local-artifact"

ONNX_MANIFEST_FILENAME = "director_onnx_manifest.json"
"""Manifest naming the served ONNX file of an export directory."""

_ONNX_MANIFEST_SCHEMA = "director-ai.onnx-export.v1"
_PRECISION_FILES = {
    "fp32": "model.onnx",
    "int8": "model_quantized.onnx",
    "fp16": "model_fp16.onnx",
}
_DIGEST = re.compile(r"[0-9a-f]{64}")
_HASH_CHUNK = 1 << 20
_REQUIRE_MANIFEST_ENV = "DIRECTOR_ONNX_REQUIRE_MANIFEST"
_FULL_RANGE_INT8_CPU_FLAGS = frozenset({"avx512_vnni", "avx_vnni", "amx_int8"})
_X86_MACHINES = frozenset({"x86_64", "amd64", "i386", "i686", "x86"})
_LEGACY_WARNED: set[Path] = set()

logger = logging.getLogger("DirectorAI.NLI")


def _validate_onnx_scheduler_config(
    max_batch: int,
    flush_timeout_ms: float,
) -> None:
    """Reject impossible ONNX dynamic scheduler settings."""
    if isinstance(max_batch, bool) or max_batch < 1:
        raise ValueError(f"max_batch must be >= 1, got {max_batch!r}")
    if not math.isfinite(flush_timeout_ms) or flush_timeout_ms < 0.0:
        raise ValueError(
            f"flush_timeout_ms must be a finite non-negative value, "
            f"got {flush_timeout_ms!r}",
        )


def _validate_tensorrt_export_config(
    max_batch: int,
    max_seq_len: int,
    warmup_pairs: int,
) -> None:
    """Reject impossible TensorRT export profile settings."""
    if isinstance(max_batch, bool) or max_batch < 1:
        raise ValueError(f"max_batch must be >= 1, got {max_batch!r}")
    if isinstance(max_seq_len, bool) or max_seq_len < 1:
        raise ValueError(f"max_seq_len must be >= 1, got {max_seq_len!r}")
    if isinstance(warmup_pairs, bool) or warmup_pairs < 1:
        raise ValueError(f"warmup_pairs must be >= 1, got {warmup_pairs!r}")


def _configured_onnx_allowed_dirs() -> tuple[Path, ...]:
    raw = os.environ.get("DIRECTOR_ONNX_ALLOWED_DIRS", "").strip()
    if not raw:
        return ()
    roots: list[Path] = []
    for part in raw.split(os.pathsep):
        item = part.strip()
        if item:
            roots.append(Path(item).expanduser().resolve())
    return tuple(roots)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_onnx_manifest(
    output_path: Path,
    *,
    precision: str,
    source_model: str,
    source_revision: str | None,
    int8_reduce_range: bool | None,
) -> Path:
    """Bind an export directory to the file it serves and that file's digest.

    The manifest is written to a temporary file in the same directory and
    moved into place, so a crash never leaves a half-written manifest.
    """
    model_file = output_path / _PRECISION_FILES[precision]
    payload = {
        "schema": _ONNX_MANIFEST_SCHEMA,
        "model_file": model_file.name,
        "sha256": _sha256_file(model_file),
        "precision": precision,
        "int8_reduce_range": int8_reduce_range,
        "source_model": source_model,
        "source_revision": source_revision,
        "opset": 17,
    }
    manifest = output_path / ONNX_MANIFEST_FILENAME
    descriptor, temporary = tempfile.mkstemp(
        dir=output_path, prefix=f".{ONNX_MANIFEST_FILENAME}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, manifest)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return manifest


def _strict_manifest_mode() -> bool:
    """Return whether ``DIRECTOR_ONNX_REQUIRE_MANIFEST=1`` is set."""
    return os.environ.get(_REQUIRE_MANIFEST_ENV, "").strip() == "1"


def _cpu_flags(cpuinfo: Path = Path("/proc/cpuinfo")) -> frozenset[str] | None:
    """Return the Linux CPU feature flags, or ``None`` when unreadable."""
    try:
        text = cpuinfo.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "flags":
            return frozenset(value.split())
    return None


def _full_range_int8_supported() -> bool | None:
    """Return whether this CPU serves full-range INT8 without saturation.

    Non-x86 CPUs (ARM) are not affected. On x86 the answer needs one of
    the VNNI or AMX flags; ``None`` means the flags could not be read.
    """
    if platform.machine().lower() not in _X86_MACHINES:
        return True
    flags = _cpu_flags()
    if flags is None:
        return None
    return bool(flags & _FULL_RANGE_INT8_CPU_FLAGS)


def _check_int8_range_policy(model_file: Path) -> None:
    """Warn about, or in strict mode refuse, a full-range INT8 mismatch."""
    supported = _full_range_int8_supported()
    if supported:
        return
    reason = (
        "this x86 CPU has no AVX512-VNNI, AVX-VNNI or AMX"
        if supported is False
        else "the CPU flags could not be read"
    )
    message = (
        f"Full-range INT8 ONNX model may saturate here because {reason}; "
        f"re-export it with int8_reduce_range=True: {model_file}"
    )
    if _strict_manifest_mode():
        raise PermissionError(message)
    logger.warning(message)


@dataclass(frozen=True)
class _ManifestBinding:
    """The served file, digest and precision named by an export manifest."""

    model_file: Path
    sha256: str
    precision: str
    int8_reduce_range: bool | None


def _manifest_binding(onnx_dir: Path) -> _ManifestBinding | None:
    """Return the manifest's binding after checking its form and file.

    Returns ``None`` when the directory has no manifest. A malformed
    manifest, a missing bound file, or a full-range INT8 model refused in
    strict mode raises ``PermissionError``, so it is never treated as an
    unavailable model. The digest is checked by the caller on the bytes it
    loads.
    """
    manifest = onnx_dir / ONNX_MANIFEST_FILENAME
    if not manifest.exists():
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PermissionError(f"ONNX manifest is unreadable: {manifest}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != _ONNX_MANIFEST_SCHEMA:
        raise PermissionError(f"ONNX manifest has an unknown schema: {manifest}")
    name, digest = payload.get("model_file"), payload.get("sha256")
    precision, reduce_range = payload.get("precision"), payload.get("int8_reduce_range")
    if (
        not isinstance(name, str)
        or name not in _PRECISION_FILES.values()
        or precision not in _PRECISION_FILES
        or _PRECISION_FILES[precision] != name
        or not isinstance(digest, str)
        or not _DIGEST.fullmatch(digest)
        or (precision == "int8") != isinstance(reduce_range, bool)
        or (precision != "int8" and reduce_range is not None)
    ):
        raise PermissionError(f"ONNX manifest is malformed: {manifest}")
    model_file = onnx_dir / name
    if not model_file.is_file():
        raise PermissionError(f"ONNX manifest names a missing model file: {model_file}")
    if precision == "int8" and reduce_range is False:
        _check_int8_range_policy(model_file)
    return _ManifestBinding(model_file, digest, precision, reduce_range)


def _verified_model_bytes(binding: _ManifestBinding) -> bytes:
    """Read the bound model once and return the bytes that match its digest."""
    data = binding.model_file.read_bytes()
    if hashlib.sha256(data).hexdigest() != binding.sha256:
        raise PermissionError(
            f"ONNX model digest does not match its manifest: {binding.model_file}"
        )
    return data


def _legacy_directory(onnx_dir: Path) -> None:
    """Refuse in strict mode, else warn once, about a manifest-less export."""
    if _strict_manifest_mode():
        raise PermissionError(
            f"ONNX directory has no export manifest and "
            f"{_REQUIRE_MANIFEST_ENV}=1 is set: {onnx_dir}"
        )
    if onnx_dir not in _LEGACY_WARNED:
        _LEGACY_WARNED.add(onnx_dir)
        logger.warning(
            "ONNX directory has no export manifest, so its served file and "
            "digest are not bound; re-export it with `director-ai export "
            "--format onnx --output %s` and set %s=1 in production: %s",
            onnx_dir,
            _REQUIRE_MANIFEST_ENV,
            onnx_dir,
        )


def _resolve_served_model(
    onnx_path: str,
) -> tuple[Path, Path, _ManifestBinding | None]:
    """Return the export directory, the served file and its manifest binding.

    A manifest selects the file. Without one the legacy layout applies,
    with a warning (or a refusal in strict mode), but a directory holding
    an FP16 export next to another ONNX file is refused, because its
    intended precision cannot be told. The digest is not checked here.
    """
    onnx_dir = Path(onnx_path).expanduser().resolve()
    allowed_dirs = _configured_onnx_allowed_dirs()
    if allowed_dirs and not any(
        _is_relative_to(onnx_dir, root) for root in allowed_dirs
    ):
        raise PermissionError(
            f"ONNX path is outside DIRECTOR_ONNX_ALLOWED_DIRS: {onnx_dir}",
        )
    if not onnx_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {onnx_path}")

    binding = _manifest_binding(onnx_dir)
    quantized = onnx_dir / "model_quantized.onnx"
    model_file = onnx_dir / "model.onnx"
    if binding is not None:
        model_file = binding.model_file
    else:
        _legacy_directory(onnx_dir)
        exports = sorted(p.name for p in onnx_dir.iterdir() if p.suffix == ".onnx")
        if "model_fp16.onnx" in exports and len(exports) > 1:
            raise PermissionError(
                f"ONNX directory without a manifest holds an FP16 export next to "
                f"other models; re-export it to bind the served file: {onnx_dir}"
            )
        if not quantized.exists() and not model_file.exists():
            for candidate in sorted(onnx_dir.iterdir()):  # pragma: no branch
                if candidate.name.endswith(".onnx"):  # pragma: no branch
                    model_file = candidate
                    break

    resolved_model = model_file.resolve()
    if not _is_relative_to(resolved_model, onnx_dir):
        raise PermissionError(f"ONNX model file escapes model directory: {model_file}")
    if allowed_dirs and not any(
        _is_relative_to(resolved_model, root) for root in allowed_dirs
    ):
        raise PermissionError(
            f"ONNX model file is outside DIRECTOR_ONNX_ALLOWED_DIRS: {resolved_model}",
        )
    return onnx_dir, model_file, binding


def _resolve_onnx_model_file(onnx_path: str) -> tuple[Path, Path]:
    """Return the export directory and the model file it serves.

    The manifest-bound file's digest is verified by streaming it, so callers
    that open the path themselves (the distilled scorer) get the same
    integrity check as the ONNX session loader.
    """
    onnx_dir, model_file, binding = _resolve_served_model(onnx_path)
    if binding is not None and _sha256_file(model_file) != binding.sha256:
        raise PermissionError(
            f"ONNX model digest does not match its manifest: {model_file}"
        )
    return onnx_dir, model_file


@lru_cache(maxsize=4)
def _load_onnx_session(
    onnx_path: str,
    device: str | None = None,
) -> tuple[Any, Any]:
    """Load ONNX Runtime session + tokenizer from exported directory."""
    try:
        # Resolve and verify first, so a policy refusal is raised even when
        # the runtime itself is not installed. A manifest-bound model is read
        # once, and those verified bytes are what ONNX Runtime loads.
        onnx_dir, model_file_path, binding = _resolve_served_model(onnx_path)
        model_source: str | bytes = str(model_file_path)
        if binding is not None:
            model_source = _verified_model_bytes(binding)
        import onnxruntime as ort
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(onnx_dir),
            revision=_LOCAL_ARTIFACT_REVISION,
            local_files_only=True,
        )

        # Legacy directories without a manifest prefer the quantized model on
        # CPU; a manifest serves exactly the file it names.
        quantized = onnx_dir / "model_quantized.onnx"
        if (
            binding is None
            and quantized.exists()
            and (device is None or "cpu" in (device or ""))
            and _is_relative_to(quantized.resolve(), onnx_dir)
        ):
            model_file_path = quantized
            model_source = str(quantized)
            logger.info("Using INT8 quantized model: %s", quantized)

        providers: list[str | tuple[str, dict[str, object]]] = [
            "CPUExecutionProvider",
        ]
        available = ort.get_available_providers()
        # GPU/TensorRT provider selection only fires on accelerator hardware,
        # which the CPU-only test environment does not expose.
        if (  # pragma: no cover — hardware-gated: CUDA provider
            device and "cuda" in device
        ) or "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")

        trt_cache = str(onnx_dir / "trt_cache")
        trt_requested = os.environ.get("DIRECTOR_ENABLE_TRT") == "1"
        trt_cache_exists = os.path.isdir(trt_cache)
        if (
            trt_requested or trt_cache_exists
        ) and (  # pragma: no cover — hardware-gated: TensorRT provider
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
        session = ort.InferenceSession(model_source, opts, providers=providers)
        logger.info(
            "ONNX session: %s (%s)",
            model_file_path,
            session.get_providers()[0],
        )
        return tokenizer, session
    except PermissionError:
        # A load-policy or integrity refusal must reach the caller; turning
        # it into an unavailable backend would fail open.
        raise
    except (ImportError, RuntimeError, OSError) as e:
        logger.warning("ONNX session unavailable: %s", e)
        return None, None


def export_onnx(
    model_name: str = _DEFAULT_MODEL,
    output_dir: str = "factcg_onnx",
    quantize: str | None = None,
    revision: str | None = None,
    int8_reduce_range: bool | None = None,
) -> str:
    """Export NLI model to ONNX format, optionally quantized.

    Parameters
    ----------
    quantize : "int8", "fp16", or None (default FP32).
    int8_reduce_range : bool or None
        Required for ``quantize="int8"`` and refused otherwise. Choose it
        for the CPU that will serve the model: ``True`` for x86 CPUs
        without VNNI, whose 8-bit dot products can saturate at full range;
        ``False`` for CPUs with AVX512-VNNI, AVX-VNNI or AMX, and for ARM.

    Requires ``torch``, ``transformers``, and ``onnx``.
    For int8: also requires ``onnxruntime`` quantization module.
    The directory gets a manifest naming the served file and its digest.
    Load the result with
    ``NLIScorer(backend="onnx", onnx_path=<returned_dir>)``.

    """
    if quantize not in {None, "int8", "fp16"}:
        raise ValueError("quantize must be one of: None, 'int8', 'fp16'")
    if quantize == "int8" and int8_reduce_range is None:
        raise ValueError(
            "int8 export needs int8_reduce_range for the target CPU: True "
            "without VNNI, False with AVX512-VNNI, AVX-VNNI, AMX or on ARM"
        )
    if quantize != "int8" and int8_reduce_range is not None:
        raise ValueError("int8_reduce_range applies only to int8 export")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from .nli import _resolve_revision

    rev = _resolve_revision(model_name, revision)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=rev)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        revision=rev,
    )
    model.eval()

    encoded = tokenizer(
        _FACTCG_TEMPLATE.format(text_a="Premise.", text_b="Hypothesis."),
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    input_names = [
        name
        for name in ("input_ids", "attention_mask", "token_type_ids")
        if name in encoded
    ]
    if not input_names:
        raise RuntimeError("Tokenizer did not produce ONNX-exportable tensor inputs")

    class _SequenceClassifierOnnxWrapper(
        torch.nn.Module  # type: ignore[misc,unused-ignore] # torch.nn.Module may be Any without torch stubs in the lean CI type env.
    ):
        def __init__(self, wrapped_model: Any, names: list[str]) -> None:
            super().__init__()
            self.wrapped_model = wrapped_model
            self.names = names

        def forward(self, *inputs: Any) -> Any:
            outputs = self.wrapped_model(
                **dict(zip(self.names, inputs, strict=True)),
            )
            return outputs.logits

    dynamic_axes: dict[str, dict[int, str]] = {
        name: {0: "batch", 1: "sequence"} for name in input_names
    }
    dynamic_axes["logits"] = {0: "batch"}
    model_file = output_path / "model.onnx"

    export_kwargs: dict[str, Any] = {
        "input_names": input_names,
        "output_names": ["logits"],
        "dynamic_axes": dynamic_axes,
        "opset_version": 17,
        "do_constant_folding": True,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    with torch.no_grad():
        torch.onnx.export(
            _SequenceClassifierOnnxWrapper(model, input_names),
            tuple(encoded[name] for name in input_names),
            str(model_file),
            **export_kwargs,
        )

    model.config.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info("ONNX model exported to %s", output_dir)

    if quantize == "int8":
        from onnxruntime.quantization import QuantType, quantize_dynamic

        src = str(model_file)
        dst = str(output_path / "model_quantized.onnx")
        quantize_dynamic(
            src,
            dst,
            weight_type=QuantType.QInt8,
            reduce_range=bool(int8_reduce_range),
        )
        logger.info(
            "INT8 quantized model saved to %s (reduce_range=%s)",
            dst,
            int8_reduce_range,
        )
    elif quantize == "fp16":
        import onnx
        from onnxruntime.transformers import float16

        src = str(model_file)
        model_fp32 = onnx.load(src)
        model_fp16 = float16.convert_float_to_float16(model_fp32)
        dst = str(output_path / "model_fp16.onnx")
        onnx.save(model_fp16, dst)
        logger.info("FP16 model saved to %s", dst)

    _write_onnx_manifest(
        output_path,
        precision=quantize or "fp32",
        source_model=model_name,
        source_revision=rev,
        int8_reduce_range=int8_reduce_range,
    )
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
    _validate_tensorrt_export_config(max_batch, max_seq_len, warmup_pairs)

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

    tokenizer = AutoTokenizer.from_pretrained(
        onnx_dir,
        revision=_LOCAL_ARTIFACT_REVISION,
        local_files_only=True,
    )
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


# ONNX Dynamic Batcher


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
        onnx_scorer_fn: Callable[[list[tuple[str, str]]], list[float]],
        max_batch: int = 16,
        flush_timeout_ms: float = 10.0,
        session: Any = None,
    ) -> None:
        import threading

        _validate_onnx_scheduler_config(max_batch, flush_timeout_ms)
        self._score_fn = onnx_scorer_fn
        self.max_batch = max_batch
        self.flush_timeout_ms = float(flush_timeout_ms)
        self._session = session
        self._buffer: list[tuple[str, str]] = []
        self._buffer_started_at: float | None = None
        self._lock = threading.Lock()
        self._has_cuda = False
        if session is not None:
            try:
                providers = session.get_providers()
                self._has_cuda = any("CUDA" in p for p in providers)
            except (
                AttributeError,
                RuntimeError,
            ):  # pragma: no cover — defensive: foreign session probe
                pass

    def submit(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Submit pairs and flush when the batch is full or timeout expires."""
        with self._lock:
            now = time.monotonic()
            if pairs:
                if not self._buffer:
                    self._buffer_started_at = now
                self._buffer.extend(pairs)
            if self._buffer and (
                len(self._buffer) >= self.max_batch or self._timeout_elapsed(now)
            ):
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
        self._buffer_started_at = None
        return self._score_fn(batch)

    def _timeout_elapsed(self, now: float) -> bool:
        """Return whether the pending batch exceeded its flush timeout."""
        if self.flush_timeout_ms == 0.0:
            return True
        if self._buffer_started_at is None:
            return False
        elapsed_ms = (now - self._buffer_started_at) * 1000.0
        return elapsed_ms >= self.flush_timeout_ms

    @property
    def uses_io_binding(self) -> bool:
        """Return whether the scheduler detected a CUDA ONNX provider."""
        return self._has_cuda
