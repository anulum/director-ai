# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — human-gated knowledge supersession policy

"""Decide which existing documents an incoming document supersedes.

A self-updating knowledge base must recognise when a freshly ingested
document replaces older material — a re-published policy, a corrected
manual, a source the operator has explicitly marked as the successor of
another. This policy turns three signals into a reviewable decision:

* **explicit hint** — the incoming document names the documents it
  supersedes;
* **same-source revision** — an existing document shares the incoming
  document's source identity, so the new one is a newer cut of it;
* **contradiction score** — a caller-supplied per-document score (from an
  NLI or similarity verifier) marks documents the incoming one contradicts.

The policy is deliberately side-effect free, mirroring
:class:`~director_ai.core.calibration.adaptive_threshold.AdaptiveThresholdLearner`:
it returns a :class:`SupersessionDecision` and never mutates the store. By
default every non-empty decision is gated on human approval; auto-promotion
is opt-in and only fires when *every* candidate clears a high score bar.
Execution — retiring the superseded chunks and recording the lineage event
— is the caller's job (see
:meth:`director_ai.core.ingestion.pipeline.DocumentIngestionPipeline.apply_supersession`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from director_ai.core.retrieval.doc_registry import DocRecord

__all__ = [
    "KnowledgeSupersessionPolicy",
    "SupersessionCandidate",
    "SupersessionDecision",
]

SupersessionAction = Literal["none", "recommend", "promote"]

_REASON_EXPLICIT = "explicit_supersedes"
_REASON_SAME_SOURCE = "same_source_revision"
_REASON_CONTRADICTION = "contradiction"


def _unit_interval(name: str, value: float) -> float:
    """Return a finite value in ``[0, 1]`` or raise."""
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]; got {value!r}")
    return float(value)


@dataclass(frozen=True)
class SupersessionCandidate:
    """One existing document the incoming document would supersede."""

    superseded_doc_id: str
    reason: str
    score: float
    evidence_ref: str

    def __post_init__(self) -> None:
        if not self.superseded_doc_id:
            raise ValueError("superseded_doc_id must be non-empty")
        _unit_interval("score", self.score)


@dataclass(frozen=True)
class SupersessionDecision:
    """Reviewable supersession outcome for one incoming document.

    ``action`` is ``"none"`` when nothing is superseded, ``"recommend"``
    when supersession is proposed but withheld for human approval, and
    ``"promote"`` when auto-promotion criteria were met. ``candidates`` is
    ordered by descending score.
    """

    incoming_doc_id: str
    tenant_id: str
    incoming_source: str
    candidates: tuple[SupersessionCandidate, ...]
    action: SupersessionAction
    requires_human_approval: bool
    evidence_refs: tuple[str, ...] = field(default_factory=tuple)

    @property
    def superseded_doc_ids(self) -> tuple[str, ...]:
        """Return the ids of every document this decision would retire."""
        return tuple(candidate.superseded_doc_id for candidate in self.candidates)

    @property
    def has_candidates(self) -> bool:
        """Return ``True`` when at least one document would be superseded."""
        return bool(self.candidates)


class KnowledgeSupersessionPolicy:
    """Map supersession signals to a human-gated decision.

    Parameters
    ----------
    min_contradiction_score :
        A contradiction score at or above this value makes a document a
        supersession candidate. Default 0.65.
    auto_promote :
        When ``True``, a decision whose every candidate scores at or above
        ``auto_promote_threshold`` is returned with ``action="promote"`` and
        ``requires_human_approval=False``. When ``False`` (default), every
        non-empty decision is gated on human approval.
    auto_promote_threshold :
        Score bar each candidate must clear for auto-promotion. Default 0.95.
    same_source_score :
        Confidence assigned to a same-source revision candidate. Default 0.9.
    """

    def __init__(
        self,
        *,
        min_contradiction_score: float = 0.65,
        auto_promote: bool = False,
        auto_promote_threshold: float = 0.95,
        same_source_score: float = 0.9,
    ) -> None:
        self.min_contradiction_score = _unit_interval(
            "min_contradiction_score", min_contradiction_score
        )
        self.auto_promote_threshold = _unit_interval(
            "auto_promote_threshold", auto_promote_threshold
        )
        if self.auto_promote_threshold < self.min_contradiction_score:
            raise ValueError(
                "auto_promote_threshold must be >= min_contradiction_score"
            )
        self.same_source_score = _unit_interval("same_source_score", same_source_score)
        self.auto_promote = bool(auto_promote)

    def evaluate(
        self,
        *,
        incoming_doc_id: str,
        incoming_source: str,
        tenant_id: str,
        existing: Sequence[DocRecord],
        explicit_supersedes: Sequence[str] = (),
        contradiction_scores: Mapping[str, float] | None = None,
    ) -> SupersessionDecision:
        """Return the supersession decision for one incoming document.

        ``existing`` is the tenant's current document set (e.g. from
        :meth:`DocRegistry.list_for_tenant`). ``contradiction_scores`` maps
        an existing document id to a per-document contradiction score in
        ``[0, 1]`` from a caller-supplied verifier. The incoming document
        itself is never treated as a candidate.
        """
        explicit = {ref for ref in explicit_supersedes if ref}
        scores = contradiction_scores or {}
        candidates: list[SupersessionCandidate] = []
        for record in existing:
            if record.doc_id == incoming_doc_id:
                continue
            candidate = self._candidate_for(
                record,
                incoming_source=incoming_source,
                explicit=explicit,
                contradiction_scores=scores,
            )
            if candidate is not None:
                candidates.append(candidate)
        candidates.sort(key=lambda item: item.score, reverse=True)
        return self._decide(
            incoming_doc_id=incoming_doc_id,
            incoming_source=incoming_source,
            tenant_id=tenant_id,
            candidates=tuple(candidates),
        )

    def _candidate_for(
        self,
        record: DocRecord,
        *,
        incoming_source: str,
        explicit: set[str],
        contradiction_scores: Mapping[str, float],
    ) -> SupersessionCandidate | None:
        """Return the highest-priority candidate for one existing record."""
        evidence = f"doc://{record.tenant_id}/{record.doc_id}"
        if record.doc_id in explicit or (record.source and record.source in explicit):
            return SupersessionCandidate(
                superseded_doc_id=record.doc_id,
                reason=_REASON_EXPLICIT,
                score=1.0,
                evidence_ref=evidence,
            )
        if record.source and record.source == incoming_source:
            return SupersessionCandidate(
                superseded_doc_id=record.doc_id,
                reason=_REASON_SAME_SOURCE,
                score=self.same_source_score,
                evidence_ref=evidence,
            )
        score = contradiction_scores.get(record.doc_id)
        if score is not None:
            score = _unit_interval("contradiction_score", float(score))
            if score >= self.min_contradiction_score:
                return SupersessionCandidate(
                    superseded_doc_id=record.doc_id,
                    reason=_REASON_CONTRADICTION,
                    score=score,
                    evidence_ref=evidence,
                )
        return None

    def _decide(
        self,
        *,
        incoming_doc_id: str,
        incoming_source: str,
        tenant_id: str,
        candidates: tuple[SupersessionCandidate, ...],
    ) -> SupersessionDecision:
        """Apply the gating rules to a candidate set."""
        evidence_refs = tuple(candidate.evidence_ref for candidate in candidates)
        if not candidates:
            return SupersessionDecision(
                incoming_doc_id=incoming_doc_id,
                tenant_id=tenant_id,
                incoming_source=incoming_source,
                candidates=candidates,
                action="none",
                requires_human_approval=False,
                evidence_refs=evidence_refs,
            )
        auto = self.auto_promote and all(
            candidate.score >= self.auto_promote_threshold for candidate in candidates
        )
        return SupersessionDecision(
            incoming_doc_id=incoming_doc_id,
            tenant_id=tenant_id,
            incoming_source=incoming_source,
            candidates=candidates,
            action="promote" if auto else "recommend",
            requires_human_approval=not auto,
            evidence_refs=evidence_refs,
        )
