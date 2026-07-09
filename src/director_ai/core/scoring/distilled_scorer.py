# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Distilled lightweight NLI scorer
"""Distilled NLI scorer for locally validated student artefacts.

This is Tier 4 in the 5-tier scoring pyramid — between embedding
similarity and full NLI. Public accuracy and latency claims require a
held-out evaluation packet plus ONNX and quantized latency evidence for
the exact student artefact.

The model is loaded from HuggingFace Hub (``anulum/director-ai-nli-lite``)
or from a local path. ONNX Runtime is used for inference; PyTorch execution
remains available for environments that intentionally select it.

Install::

    pip install director-ai[nli-lite]

Usage::

    from director_ai.core.scoring.distilled_scorer import DistilledNLIBackend

    backend = DistilledNLIBackend()
    score = backend.score("Water boils at 100°C.", "Water boils at 500°C.")
    # score ≈ 0.2 (low — NLI detects factual contradiction)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..mandatory import mandatory_execution, require_rust_kernel

logger = logging.getLogger("DirectorAI.DistilledNLI")

try:
    from backfire_kernel import rust_softmax, rust_sum_f64

    _RUST_DISTILLED = True
except ImportError:
    _RUST_DISTILLED = True

    def rust_sum_f64(_values: list[float]) -> float:
        """Raise to signal the mandatory Rust sum accelerator is missing."""
        require_rust_kernel("rust_sum_f64")

    def rust_softmax(_flat: list[float], _cols: int) -> list[float]:
        """Raise to signal the mandatory Rust softmax accelerator is missing."""
        require_rust_kernel("rust_softmax")


DEFAULT_DISTILLED_MODEL = "anulum/director-ai-nli-lite"
DEFAULT_DISTILLED_REVISION = "f88222676f64b698c1fcb394f4eeb8da40405027"
_DISTILLED_TOKENISER_MODEL_FILES = ("tokenizer.json", "vocab.txt", "spiece.model")
_DISTILLED_TOKENISER_SUPPORT_FILES = (
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merges.txt",
)


@dataclass(frozen=True, slots=True)
class DistilledOnnxArtifact:
    """Resolved local ONNX artefact accepted by the distilled scorer loader.

    Attributes
    ----------
    model_dir:
        Resolved directory containing the local artefact.
    model_file:
        Resolved ONNX model file selected for inference.
    tokeniser_files:
        Resolved tokenizer assets that remain inside ``model_dir``.
    """

    model_dir: Path
    model_file: Path
    tokeniser_files: tuple[Path, ...]


def _path_is_relative_to(path: Path, root: Path) -> bool:
    """Return whether ``path`` resolves under ``root``."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _local_model_path_exists(model_path: str) -> bool:
    """Return whether ``model_path`` points at an existing local path."""
    return Path(model_path).expanduser().exists()


def _resolve_tokeniser_files(model_dir: Path) -> tuple[Path, ...]:
    """Resolve tokeniser assets and reject symlink escapes."""
    tokeniser_model_files = [
        model_dir / filename
        for filename in _DISTILLED_TOKENISER_MODEL_FILES
        if (model_dir / filename).exists()
    ]
    if not tokeniser_model_files:
        raise FileNotFoundError(
            "Distilled local ONNX artefact requires tokenizer.json, "
            "vocab.txt, or spiece.model"
        )

    candidates = tokeniser_model_files + [
        model_dir / filename
        for filename in _DISTILLED_TOKENISER_SUPPORT_FILES
        if (model_dir / filename).exists()
    ]
    resolved_files: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if not _path_is_relative_to(resolved, model_dir):
            raise PermissionError(
                f"Distilled tokeniser file escapes model directory: {candidate}"
            )
        resolved_files.append(resolved)
    return tuple(resolved_files)


def validate_local_distilled_onnx_artifact(
    model_path: str | Path,
) -> DistilledOnnxArtifact:
    """Validate a local distilled ONNX artefact before runtime loading.

    The check runs before importing ONNX Runtime or Transformers so incomplete
    local artefacts fail closed without falling through to a hub download or a
    misleading PyTorch fallback.

    Parameters
    ----------
    model_path:
        Local directory that should contain a distilled ONNX model and tokenizer
        assets.

    Returns
    -------
    DistilledOnnxArtifact
        Resolved artefact paths safe for the runtime loader.

    Raises
    ------
    FileNotFoundError
        If the directory, ONNX model, or tokenizer assets are missing.
    PermissionError
        If the artefact or tokeniser files escape the configured local roots.
    """
    from director_ai.core.scoring._nli_export import _resolve_onnx_model_file

    model_dir, model_file = _resolve_onnx_model_file(str(model_path))
    resolved_model_dir = model_dir.resolve()
    resolved_model_file = model_file.resolve()
    if not resolved_model_file.is_file():
        raise FileNotFoundError(f"Distilled ONNX model file not found: {model_file}")

    return DistilledOnnxArtifact(
        model_dir=resolved_model_dir,
        model_file=resolved_model_file,
        tokeniser_files=_resolve_tokeniser_files(resolved_model_dir),
    )


class DistilledNLIBackend:
    """Distilled NLI scorer for validated small-model artefacts.

    Loads a small NLI model distilled from a stronger NLI teacher.
    Supports ONNX Runtime and explicit PyTorch execution.

    Parameters
    ----------
    model_path : str
        HuggingFace model ID or local directory path.
    use_onnx : bool
        If True (default), use ONNX Runtime inference.
    device : str
        ``"cpu"`` or ``"cuda"``. ONNX auto-detects; PyTorch uses this.
    max_length : int
        Maximum token sequence length.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_DISTILLED_MODEL,
        use_onnx: bool = True,
        device: str = "cpu",
        max_length: int = 256,
    ) -> None:
        self._model_path = model_path
        self._use_onnx = use_onnx
        self._device = device
        self._max_length = max_length
        self._tokeniser: Any | None = None
        self._session: Any | None = None  # ONNX session
        self._model: Any | None = None  # PyTorch model
        self._torch: Any | None = None
        self._ready = False

    def _ensure_loaded(self) -> None:
        """Lazy-load model on first use."""
        if self._ready:
            return

        # Try ONNX first
        if self._use_onnx:
            local_model_path_exists = _local_model_path_exists(self._model_path)
            try:
                self._load_onnx()
                self._ready = True
                return
            except PermissionError:
                raise
            except FileNotFoundError:
                if local_model_path_exists:
                    raise
                logger.warning("ONNX load failed, falling back to PyTorch")
            except (ImportError, OSError, RuntimeError) as exc:
                logger.warning("ONNX load failed, falling back to PyTorch: %s", exc)

        # PyTorch fallback
        self._load_pytorch()
        self._ready = True

    def _load_onnx(self) -> None:
        """Load ONNX Runtime session + tokeniser."""
        model_dir = Path(self._model_path)
        artifact: DistilledOnnxArtifact | None = None
        if model_dir.is_dir():
            artifact = validate_local_distilled_onnx_artifact(self._model_path)
            onnx_path = artifact.model_file
            tokenizer_model_path = str(artifact.model_dir)
        else:
            onnx_path = None
            tokenizer_model_path = self._model_path

        if onnx_path is None or not onnx_path.exists():
            # Try downloading from HF Hub
            from huggingface_hub import hf_hub_download

            onnx_path = Path(
                hf_hub_download(
                    self._model_path,
                    "model.onnx",
                    revision=DEFAULT_DISTILLED_REVISION,
                )
            )

        import onnxruntime as ort
        from transformers import AutoTokenizer

        if artifact is not None:
            self._tokeniser = AutoTokenizer.from_pretrained(
                tokenizer_model_path,
                revision=DEFAULT_DISTILLED_REVISION,
                local_files_only=True,
            )
        else:
            self._tokeniser = AutoTokenizer.from_pretrained(
                tokenizer_model_path,
                revision=DEFAULT_DISTILLED_REVISION,
            )
        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        logger.info("Distilled NLI loaded (ONNX): %s", self._model_path)

    def _load_pytorch(self) -> None:
        """Load PyTorch model + tokeniser."""
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise ImportError(
                "DistilledNLIBackend requires transformers + torch. "
                "Install with: pip install director-ai[nli-lite]"
            ) from exc

        self._tokeniser = AutoTokenizer.from_pretrained(
            self._model_path,
            revision=DEFAULT_DISTILLED_REVISION,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_path,
            revision=DEFAULT_DISTILLED_REVISION,
        )
        if self._model is None:
            raise RuntimeError("Distilled NLI model did not load")
        self._model.to(self._device).eval()
        self._torch = torch
        logger.info("Distilled NLI loaded (PyTorch): %s", self._model_path)

    def _infer(self, premise: str, hypothesis: str) -> float:
        """Run inference, return P(entailment) in [0, 1]."""
        if self._tokeniser is None:
            raise RuntimeError("Distilled NLI tokeniser not loaded")
        inputs = self._tokeniser(
            premise,
            hypothesis,
            return_tensors="np" if self._session else "pt",
            truncation=True,
            max_length=self._max_length,
            padding=True,
        )

        if self._session is not None:
            # ONNX path
            ort_inputs = {
                k: v
                for k, v in inputs.items()
                if k in {inp.name for inp in self._session.get_inputs()}
            }
            logits = self._session.run(None, ort_inputs)[0]
            probs = _softmax(logits[0])
        else:
            if self._model is None or self._torch is None:
                raise RuntimeError("Distilled NLI PyTorch model not loaded")
            # PyTorch path
            with self._torch.no_grad():
                pt_inputs = {k: v.to(self._device) for k, v in inputs.items()}
                logits = self._model(**pt_inputs).logits
                probs = self._torch.softmax(logits, dim=-1)[0].cpu().numpy()

        # Convention: label 0 = entailment/supported, label 1 = contradiction
        # Return P(supported) as the score
        return float(probs[0]) if len(probs) >= 2 else float(probs[0])

    def score(self, premise: str, hypothesis: str) -> float:
        """Score groundedness. Returns [0, 1] where 1 = supported."""
        self._ensure_loaded()
        return self._infer(premise, hypothesis)

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score multiple pairs."""
        self._ensure_loaded()
        return [self._infer(p, h) for p, h in pairs]


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    if _RUST_DISTILLED:
        with mandatory_execution(__name__, component="mandatory accelerated path"):
            flat = [float(v) for v in np.asarray(logits, dtype=float).ravel()]
            probs = rust_softmax(flat, len(flat))
            probs_array: np.ndarray = np.asarray(probs, dtype=float)
            return probs_array
    e: np.ndarray = np.exp(logits - np.max(logits))
    denom = _sum_float_list(e.ravel().tolist())
    result: np.ndarray = e / denom
    return result


def _sum_float_list(values: list[float]) -> float:
    if _RUST_DISTILLED:
        with mandatory_execution(__name__, component="mandatory accelerated path"):
            return float(rust_sum_f64(values))
    return float(sum(values))
