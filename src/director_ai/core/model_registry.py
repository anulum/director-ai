# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — fallback model registry

"""Degrade to a vetted alternate model when the primary is unavailable.

``DirectorConfig.nli_model`` is a configurable, revision-pinned default, but a
single hardcoded id has no recourse if that repository is delisted or otherwise
unreachable on the Hugging Face Hub: the NLI scorer then drops all the way to its
word-overlap heuristic. This registry keeps an ordered chain of vetted,
revision-pinned alternates per role and resolves the primary to the first
*available* model in the chain — so a deployment degrades to a strong alternate
model rather than to the heuristic floor.

Availability is decided by an injected :class:`AvailabilityProbe` (default: a
cheap Hugging Face ``model_info`` metadata call, no download), so resolution is
deterministic and fully tested offline. The registry never raises on a missing
model: if nothing in the chain is reachable it returns the configured primary
unchanged and lets the scorer's own heuristic fallback take over.

Only the ``nli`` role is populated: its alternates are MNLI-style
sequence-classifiers the DeBERTa backend loads directly, all already pinned in
:data:`~director_ai.core.model_revisions.MODEL_REVISION_REGISTRY`. Embedding and
reranker fallbacks are intentionally omitted — a different-dimension embedding
model is not a drop-in replacement (it invalidates an existing index), so that
swap is a deliberate re-indexing decision, not an automatic failover.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .model_revisions import resolve_model_revision

__all__ = [
    "AvailabilityProbe",
    "FALLBACK_CHAINS",
    "FallbackModelRegistry",
    "ResolvedModel",
]

# Vetted, revision-pinned drop-in alternates per role, ordered strongest →
# weakest. Every entry is an MNLI-style sequence classifier the DeBERTa NLI
# backend can load directly and is pinned in MODEL_REVISION_REGISTRY.
FALLBACK_CHAINS: dict[str, tuple[str, ...]] = {
    "nli": (
        "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        "roberta-large-mnli",
    ),
}


@dataclass(frozen=True)
class ResolvedModel:
    """The model a role resolved to, and whether it is a fallback."""

    model_id: str
    revision: str | None
    role: str
    is_fallback: bool


class AvailabilityProbe(Protocol):
    """Return whether ``model_id`` at ``revision`` is reachable."""

    def __call__(self, model_id: str, revision: str | None) -> bool:
        """Return True when the model revision is currently available."""
        ...  # pragma: no cover


class FallbackModelRegistry:
    """Resolve a role's primary model to the first available vetted alternate.

    Parameters
    ----------
    probe : AvailabilityProbe | None
        Availability check; defaults to a Hugging Face ``model_info`` probe.
    chains : Mapping[str, tuple[str, ...]] | None
        Per-role fallback chains; defaults to :data:`FALLBACK_CHAINS`. Every
        chain entry must resolve to an immutable revision (be pinned), else a
        :class:`ValueError` is raised at construction.
    """

    def __init__(
        self,
        *,
        probe: AvailabilityProbe | None = None,
        chains: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        self._probe = probe if probe is not None else _hub_availability
        self._chains = dict(chains) if chains is not None else dict(FALLBACK_CHAINS)
        for role, models in self._chains.items():
            for model_id in models:
                try:
                    resolve_model_revision(model_id)
                except ValueError as exc:
                    raise ValueError(
                        f"fallback chain {role!r} entry {model_id!r} is not "
                        f"revision-pinned: {exc}"
                    ) from exc
        self._cache: dict[tuple[str, str | None], bool] = {}

    def candidates(
        self, role: str, primary: str, *, primary_revision: str | None = None
    ) -> list[tuple[str, str | None]]:
        """Return the primary followed by the role's fallbacks (primary not repeated)."""
        out: list[tuple[str, str | None]] = [(primary, primary_revision)]
        for model_id in self._chains.get(role, ()):
            if model_id != primary:
                out.append((model_id, resolve_model_revision(model_id)))
        return out

    def resolve(
        self, role: str, primary: str, *, primary_revision: str | None = None
    ) -> ResolvedModel:
        """Return the first available model for ``role``.

        Tries the primary first, then each vetted fallback. If none is reachable,
        returns the primary unchanged (the scorer's heuristic floor then applies).
        """
        for model_id, revision in self.candidates(
            role, primary, primary_revision=primary_revision
        ):
            if self._is_available(model_id, revision):
                return ResolvedModel(
                    model_id, revision, role, is_fallback=model_id != primary
                )
        return ResolvedModel(primary, primary_revision, role, is_fallback=False)

    def _is_available(self, model_id: str, revision: str | None) -> bool:
        key = (model_id, revision)
        if key not in self._cache:
            self._cache[key] = bool(self._probe(model_id, revision))
        return self._cache[key]


def _hub_availability(  # pragma: no cover -- requires network
    model_id: str, revision: str | None
) -> bool:
    """Default probe: a Hugging Face metadata call (no weights downloaded)."""
    try:
        from huggingface_hub import model_info

        model_info(model_id, revision=revision)
        return True
    except Exception:
        return False
