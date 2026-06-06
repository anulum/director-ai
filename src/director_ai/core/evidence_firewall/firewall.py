# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Evidence firewall

"""Run the admission checks over a set of retrieved chunks before the model
sees them.

:class:`EvidenceFirewall` is the single entry point. It maps each retrieved
chunk to a :class:`ChunkVerdict` (admitted or quarantined, with the per-check
outcomes) and returns a :class:`FirewallReport` that separates the chunks safe
to hand to the model from those held back, with a tenant-safe reason for every
exclusion. It is side-effect-free apart from counter metrics.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..metrics import metrics
from . import checks
from .chunk import RetrievedChunk
from .poison import default_poison_scan
from .policy import CheckOutcome, FirewallContext, FirewallPolicy

__all__ = ["ChunkVerdict", "EvidenceFirewall", "FirewallReport"]

_FIREWALL_SCREENED = "evidence_firewall_chunks_screened_total"
_FIREWALL_QUARANTINED = "evidence_firewall_chunks_quarantined_total"


@dataclass(frozen=True)
class ChunkVerdict:
    """Admission outcome for one chunk.

    Parameters
    ----------
    chunk:
        The chunk that was screened.
    admitted:
        ``True`` when every enforced check passed.
    outcomes:
        The :class:`CheckOutcome` for every check the policy ran on this chunk,
        in execution order.
    """

    chunk: RetrievedChunk
    admitted: bool
    outcomes: tuple[CheckOutcome, ...]

    @property
    def failed_reasons(self) -> tuple[str, ...]:
        """Tenant-safe codes for the checks this chunk failed."""
        return tuple(o.reason for o in self.outcomes if not o.passed)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict without raw chunk text."""
        return {
            "chunk_id": self.chunk.chunk_id,
            "admitted": self.admitted,
            "checks": [
                {"name": o.name, "passed": o.passed, "reason": o.reason}
                for o in self.outcomes
            ],
            "failed_reasons": list(self.failed_reasons),
        }


@dataclass(frozen=True)
class FirewallReport:
    """The outcome of screening one retrieval batch."""

    verdicts: tuple[ChunkVerdict, ...] = field(default_factory=tuple)

    @property
    def admitted(self) -> tuple[RetrievedChunk, ...]:
        """Chunks that passed every enforced check, in input order."""
        return tuple(v.chunk for v in self.verdicts if v.admitted)

    @property
    def quarantined(self) -> tuple[ChunkVerdict, ...]:
        """Verdicts for chunks held back, in input order."""
        return tuple(v for v in self.verdicts if not v.admitted)

    @property
    def all_admitted(self) -> bool:
        """Whether no chunk was quarantined."""
        return all(v.admitted for v in self.verdicts)

    def admitted_results(self) -> list[dict[str, Any]]:
        """Admitted chunks back in the vector-store ``{id,text,metadata}`` shape.

        Lets a caller swap the firewall in front of a retriever and pass the
        survivors straight to the existing grounding path.
        """
        return [
            {
                "id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": dict(chunk.metadata),
            }
            for chunk in self.admitted
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the whole report to a JSON-safe, tenant-safe dict."""
        return {
            "admitted_count": len(self.admitted),
            "quarantined_count": len(self.quarantined),
            "verdicts": [v.to_dict() for v in self.verdicts],
        }


class EvidenceFirewall:
    """Screen retrieved chunks against a :class:`FirewallPolicy`.

    Parameters
    ----------
    policy:
        The admission policy. Defaults to the fail-closed
        :class:`FirewallPolicy` defaults.
    poison_scan:
        Callable mapping chunk text to a score in ``[0, 1]``. Defaults to the
        dependency-free :func:`~director_ai.core.evidence_firewall.poison.default_poison_scan`;
        the model-backed ``InjectionDetector`` can be supplied instead.
    """

    def __init__(
        self,
        policy: FirewallPolicy | None = None,
        *,
        poison_scan: Callable[[str], float] | None = None,
    ) -> None:
        self.policy = policy or FirewallPolicy()
        self._poison_scan = poison_scan or default_poison_scan

    def screen(
        self,
        chunks: Iterable[RetrievedChunk | Mapping[str, Any]],
        context: FirewallContext,
    ) -> FirewallReport:
        """Screen ``chunks`` for ``context`` and return a :class:`FirewallReport`.

        Each item may be an :class:`RetrievedChunk` or a raw vector-store result
        mapping; the latter is normalised via
        :meth:`RetrievedChunk.from_query_result`.
        """
        verdicts: list[ChunkVerdict] = []
        for item in chunks:
            chunk = (
                item
                if isinstance(item, RetrievedChunk)
                else RetrievedChunk.from_query_result(item)
            )
            verdicts.append(self._screen_one(chunk, context))
        return FirewallReport(verdicts=tuple(verdicts))

    def screen_results(
        self,
        results: Sequence[Mapping[str, Any]],
        context: FirewallContext,
    ) -> list[dict[str, Any]]:
        """Screen raw vector-store results and return only the admitted rows.

        A drop-in filter for an existing retrieval call site that does not yet
        consume the full report.
        """
        return self.screen(results, context).admitted_results()

    def _screen_one(
        self,
        chunk: RetrievedChunk,
        context: FirewallContext,
    ) -> ChunkVerdict:
        outcomes: list[CheckOutcome] = []
        for outcome in (
            checks.check_tenant_authorisation(chunk, self.policy, context),
            checks.check_provenance_present(chunk, self.policy, context),
            checks.check_signature_verified(chunk, self.policy, context),
            checks.check_content_hash(chunk, self.policy, context),
            checks.check_expiry(chunk, self.policy, context),
            checks.check_max_age(chunk, self.policy, context),
            checks.check_source_owner(chunk, self.policy, context),
            checks.check_sensitivity(chunk, self.policy, context),
            checks.check_allowed_use_case(chunk, self.policy, context),
            checks.check_poisoning(chunk, self.policy, context, scan=self._poison_scan),
        ):
            if outcome is not None:
                outcomes.append(outcome)

        admitted = all(o.passed for o in outcomes)
        metrics.inc(_FIREWALL_SCREENED)
        if not admitted:
            for outcome in outcomes:
                if not outcome.passed:
                    metrics.inc_labeled(
                        _FIREWALL_QUARANTINED, {"reason": outcome.reason}
                    )
        return ChunkVerdict(chunk=chunk, admitted=admitted, outcomes=tuple(outcomes))
