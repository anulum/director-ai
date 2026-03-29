# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
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
import logging

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
        group: list = (  # type: ignore[assignment]
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


# Auto-register built-ins
register_backend("deberta", DeBERTaBackend)
register_backend("onnx", OnnxBackend)
register_backend("minicheck", MiniCheckBackend)
register_backend("lite", LiteBackend)

try:
    from backfire_kernel import BackfireConfig as _BkCfg  # noqa: F401
    from backfire_kernel import RustCoherenceScorer as _RustScorer  # noqa: F401

    register_backend("rust", RustBackend)
    register_backend("backfire", RustBackend)
except ImportError:  # pragma: no cover
    pass
except AttributeError as _attr_err:  # pragma: no cover
    logger.warning("backfire_kernel found but broken: %s", _attr_err)
