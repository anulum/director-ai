# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — NLI Model Provisioning (revision pins, GCS artefacts, loaders)
"""Model provisioning for the NLI scorer.

Everything that turns a model name into loadable weights lives here:
immutable revision resolution against the pinned registry, managed
GCS artefact download with a deterministic local cache, the cached
PyTorch model/tokenizer loader, cache eviction, and the availability
probe for the optional torch + transformers dependencies. Inference
itself stays in :mod:`.nli`.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from ..model_revisions import (
    DEFAULT_NLI_MODEL,
    DEFAULT_NLI_MODEL_REVISION,
    MODEL_REVISION_REGISTRY,
    resolve_model_revision,
)
from ._nli_export import _load_onnx_session

__all__ = [
    "MODEL_REGISTRY",
    "clear_model_cache",
    "nli_available",
]

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
        import google.cloud.storage as storage
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
