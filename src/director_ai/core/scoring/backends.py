# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Pluggable Scorer Backend Registry

"""ABC + registry for scorer backends. Third-party backends register
via ``director_ai.backends`` entry points.

Usage::

    from director_ai.core.backends import ScorerBackend, register_backend

    class MyBackend(ScorerBackend):
        def score(self, premise, hypothesis): ...
        def score_batch(self, pairs): ...

    register_backend("mybackend", MyBackend)
"""

from __future__ import annotations

import abc
import importlib.util
import logging
from typing import Any

__all__ = [
    "ScorerBackend",
    "get_backend",
    "list_backends",
    "register_backend",
]

logger = logging.getLogger("DirectorAI.Backends")

_REGISTRY: dict[str, type[ScorerBackend]] = {}
_ENTRY_POINTS_LOADED = False


class ScorerBackend(abc.ABC):
    """Abstract base for scorer backends."""

    @abc.abstractmethod
    def score(self, premise: str, hypothesis: str) -> float:
        """Score divergence between premise and hypothesis. Returns [0, 1]."""

    @abc.abstractmethod
    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score multiple pairs."""


def register_backend(name: str, cls: type[ScorerBackend]) -> None:
    """Register a scorer backend class under *name*."""
    if not (isinstance(cls, type) and issubclass(cls, ScorerBackend)):
        raise TypeError(f"{cls!r} must be a ScorerBackend subclass")
    _REGISTRY[name] = cls
    logger.debug("Registered backend: %s", name)


def get_backend(name: str) -> type[ScorerBackend]:
    """Look up a registered backend by name. Raises KeyError if unknown."""
    _load_entry_points()
    if name not in _REGISTRY:
        raise KeyError(f"Unknown backend {name!r}. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]


def list_backends() -> dict[str, type[ScorerBackend]]:
    """Return all registered backends."""
    _load_entry_points()
    return dict(_REGISTRY)


def _load_entry_points() -> None:
    """Discover backends from ``director_ai.backends`` entry points (once)."""
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    try:
        from importlib.metadata import entry_points

        eps = entry_points()
        group: Any = (
            eps.get("director_ai.backends", [])
            if isinstance(eps, dict)
            else eps.select(group="director_ai.backends")
        )
        for ep in group:
            try:
                cls = ep.load()
                if ep.name not in _REGISTRY:  # pragma: no cover
                    register_backend(ep.name, cls)
            except (ImportError, AttributeError, TypeError) as exc:  # pragma: no cover
                logger.warning(
                    "Failed to load backend entry point %s: %s",
                    ep.name,
                    exc,
                )
    except ImportError:
        pass


# â”€â”€ Built-in backend wrappers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class DeBERTaBackend(ScorerBackend):
    """Wraps PyTorch NLI inference."""

    def __init__(self, **kwargs) -> None:
        from .nli import NLIScorer

        self._nli = NLIScorer(backend="deberta", **kwargs)

    def score(self, premise: str, hypothesis: str) -> float:
        return self._nli.score(premise, hypothesis)

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        return self._nli.score_batch(pairs)


class OnnxBackend(ScorerBackend):
    """Wraps ONNX Runtime inference."""

    def __init__(self, **kwargs) -> None:
        from .nli import NLIScorer

        self._nli = NLIScorer(backend="onnx", **kwargs)

    def score(self, premise: str, hypothesis: str) -> float:
        return self._nli.score(premise, hypothesis)

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        return self._nli.score_batch(pairs)


class MiniCheckBackend(ScorerBackend):
    """Wraps MiniCheck inference."""

    def __init__(self, **kwargs) -> None:
        from .nli import NLIScorer

        self._nli = NLIScorer(backend="minicheck", **kwargs)

    def score(self, premise: str, hypothesis: str) -> float:
        return self._nli.score(premise, hypothesis)

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        return self._nli.score_batch(pairs)


class LiteBackend(ScorerBackend):
    """Wraps LiteScorer (no ML dependencies)."""

    def __init__(self) -> None:
        from .lite_scorer import LiteScorer

        self._scorer = LiteScorer()

    def score(self, premise: str, hypothesis: str) -> float:
        return self._scorer.score(premise, hypothesis)

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        return self._scorer.score_batch(pairs)


class RustBackend(ScorerBackend):
    """Wraps backfire_kernel.RustCoherenceScorer (heuristic-only, sub-1ms)."""

    def __init__(self, **kwargs) -> None:
        from backfire_kernel import BackfireConfig, RustCoherenceScorer

        self._scorer = RustCoherenceScorer(
            config=BackfireConfig(coherence_threshold=kwargs.get("threshold", 0.6)),
            knowledge_callback=kwargs.get("knowledge_callback"),
        )

    def score(self, premise: str, hypothesis: str) -> float:
        approved, score_obj = self._scorer.review(premise, hypothesis)
        return score_obj.score if hasattr(score_obj, "score") else float(score_obj)

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [self.score(p, h) for p, h in pairs]


class RulesBackendWrapper(ScorerBackend):
    """Wraps RulesBackend (zero ML deps, <1ms, rule-based)."""

    def __init__(self, **kwargs) -> None:
        from .rules_scorer import RulesBackend

        self._scorer = RulesBackend(
            rules_file=kwargs.get("rules_file", ""),
        )

    def score(self, premise: str, hypothesis: str) -> float:
        return self._scorer.score(premise, hypothesis)

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        return self._scorer.score_batch(pairs)


# Auto-register built-ins
register_backend("deberta", DeBERTaBackend)
register_backend("onnx", OnnxBackend)
register_backend("minicheck", MiniCheckBackend)
register_backend("lite", LiteBackend)
register_backend("rules", RulesBackendWrapper)


class EmbedBackendWrapper(ScorerBackend):
    """Wraps EmbedBackend (sentence-transformers, ~65% BA, 3ms CPU)."""

    def __init__(self, **kwargs) -> None:
        from .embed_scorer import EmbedBackend

        self._scorer = EmbedBackend(
            model_name=kwargs.get("model_name", "BAAI/bge-small-en-v1.5"),
            device=kwargs.get("device", "cpu"),
            cache_dir=kwargs.get("cache_dir"),
        )

    def score(self, premise: str, hypothesis: str) -> float:
        return self._scorer.score(premise, hypothesis)

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        return self._scorer.score_batch(pairs)


try:
    import importlib.util

    if importlib.util.find_spec("sentence_transformers") is not None:
        register_backend("embed", EmbedBackendWrapper)
except ImportError:  # pragma: no cover
    pass


class DistilledNLIBackendWrapper(ScorerBackend):
    """Wraps DistilledNLIBackend (~70% BA, 5ms, ONNX INT8)."""

    def __init__(self, **kwargs) -> None:
        from .distilled_scorer import DistilledNLIBackend

        self._scorer = DistilledNLIBackend(
            model_path=kwargs.get("model_path", "anulum/director-ai-nli-lite"),
            use_onnx=kwargs.get("use_onnx", True),
            device=kwargs.get("device", "cpu"),
        )

    def score(self, premise: str, hypothesis: str) -> float:
        return self._scorer.score(premise, hypothesis)

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        return self._scorer.score_batch(pairs)


# nli-lite registered only when model is available (lazy — doesn't
# check at import time, will fail gracefully at score() time).
register_backend("nli-lite", DistilledNLIBackendWrapper)

if importlib.util.find_spec("backfire_kernel") is not None:
    try:
        # Availability probe — attribute access verifies the kernel
        # ships the symbols the RustBackend needs. Both register
        # calls are idempotent if the probe passes.
        import backfire_kernel as _bk

        _ = _bk.BackfireConfig
        _ = _bk.RustCoherenceScorer
        register_backend("rust", RustBackend)
        register_backend("backfire", RustBackend)
    except AttributeError as _attr_err:  # pragma: no cover
        logger.warning("backfire_kernel found but broken: %s", _attr_err)
