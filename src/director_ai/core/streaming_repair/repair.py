# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Streaming repair

"""Repair an unsupported clause instead of only halting on it.

The streaming kernel halts a stream when coherence drops and drops the unsafe
tail. That is safe but blunt: one bad clause discards the rest of a good answer.
:class:`StreamingRepairer` turns the halt into a corrective pass — it pauses,
finds the unsupported clause, retrieves corrective evidence, rewrites only that
clause (or redacts it when no rewrite path is configured), and resumes with the
rest of the answer intact, emitting a tenant-safe repair event for each fix.

Scoring, retrieval, and rewriting are injected so the repairer stays decoupled
from any particular scorer, store, or model:

* ``score_fn(clause) -> float`` — support score in ``[0, 1]`` for one clause.
* ``retrieve_fn(clause) -> evidence rows`` — corrective evidence for a clause.
* ``rewrite_fn(clause, [evidence_text, …]) -> str`` — a grounded rewrite.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..metrics import metrics
from ..safety_event import SafetyEvent
from .clauses import join_clauses, split_clauses

__all__ = ["RepairAction", "RepairResult", "StreamingRepairer"]

_REPAIR_TOTAL = "streaming_repair_clauses_total"
_REPAIR_ACTION = "streaming_repair_actions_total"

# Action vocabulary.
_KEEP = "keep"
_REWRITE = "rewrite"
_REDACT = "redact"


@dataclass(frozen=True)
class RepairAction:
    """What the repairer did to one clause.

    Parameters
    ----------
    clause_index:
        Position of the clause in the split answer.
    action:
        One of ``keep`` / ``rewrite`` / ``redact``.
    support:
        The clause's support score in ``[0, 1]``.
    evidence_ids:
        Corrective evidence ids used for a rewrite; empty otherwise.
    reason:
        Tenant-safe reason code for the action.
    """

    clause_index: int
    action: str
    support: float
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    reason: str = ""

    def __post_init__(self) -> None:
        """Reject an unknown action code and freeze ``evidence_ids`` to a tuple."""
        if self.action not in (_KEEP, _REWRITE, _REDACT):
            raise ValueError(f"unsupported repair action {self.action!r}")
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))

    @property
    def repaired(self) -> bool:
        """Whether this action changed the clause."""
        return self.action != _KEEP

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe, tenant-safe dict (no clause text)."""
        return {
            "clause_index": self.clause_index,
            "action": self.action,
            "support": self.support,
            "evidence_ids": list(self.evidence_ids),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RepairResult:
    """Outcome of repairing one answer."""

    repaired_text: str
    actions: tuple[RepairAction, ...] = field(default_factory=tuple)
    events: tuple[SafetyEvent, ...] = field(default_factory=tuple)

    @property
    def repaired(self) -> bool:
        """Whether any clause was changed."""
        return any(a.repaired for a in self.actions)

    @property
    def repaired_count(self) -> int:
        """Number of clauses changed."""
        return sum(1 for a in self.actions if a.repaired)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe, tenant-safe dict.

        ``repaired_text`` is the answer for the requesting tenant, so it is
        included; the per-clause actions and events carry no raw clause text.
        """
        return {
            "repaired_text": self.repaired_text,
            "repaired": self.repaired,
            "repaired_count": self.repaired_count,
            "actions": [a.to_dict() for a in self.actions],
            "events": [e.to_dict() for e in self.events],
        }


class StreamingRepairer:
    """Repair unsupported clauses in a generated answer.

    Parameters
    ----------
    score_fn:
        Maps a clause to a support score in ``[0, 1]``.
    threshold:
        A clause scoring below this is treated as unsupported and repaired.
    retrieve_fn:
        Optional; maps a clause to corrective evidence rows (mappings with
        ``id``/``source`` and ``text``, or objects exposing those attributes).
    rewrite_fn:
        Optional; given the clause and the corrective evidence texts, returns a
        grounded rewrite. When absent (or when no evidence is found), the clause
        is redacted instead.
    redaction:
        The text a clause is replaced with when it cannot be rewritten.
    """

    def __init__(
        self,
        score_fn: Callable[[str], float],
        *,
        threshold: float = 0.6,
        retrieve_fn: Callable[[str], Sequence[Any]] | None = None,
        rewrite_fn: Callable[[str, list[str]], str] | None = None,
        redaction: str = "[removed: unsupported by the knowledge base]",
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self._score_fn = score_fn
        self.threshold = threshold
        self._retrieve_fn = retrieve_fn
        self._rewrite_fn = rewrite_fn
        self.redaction = redaction

    def repair(
        self,
        text: str,
        *,
        tenant_id: str = "",
        request_id: str = "",
    ) -> RepairResult:
        """Repair ``text`` clause by clause and return the corrected answer."""
        clauses = split_clauses(text)
        repaired: list[str] = []
        actions: list[RepairAction] = []
        events: list[SafetyEvent] = []
        for index, segment in enumerate(clauses):
            core = segment.rstrip()
            trailing = segment[len(core) :]
            if not core:
                repaired.append(segment)
                continue
            metrics.inc(_REPAIR_TOTAL)
            support = float(self._score_fn(core))
            if support >= self.threshold:
                repaired.append(segment)
                actions.append(
                    RepairAction(index, _KEEP, round(support, 4), reason="supported")
                )
                metrics.inc_labeled(_REPAIR_ACTION, {"action": _KEEP})
                continue
            replacement, action, evidence_ids = self._fix_clause(core)
            repaired.append(replacement + trailing)
            actions.append(
                RepairAction(
                    index,
                    action,
                    round(support, 4),
                    evidence_ids=evidence_ids,
                    reason="unsupported_clause",
                )
            )
            metrics.inc_labeled(_REPAIR_ACTION, {"action": action})
            events.append(
                self._repair_event(
                    index=index,
                    action=action,
                    support=support,
                    evidence_ids=evidence_ids,
                    tenant_id=tenant_id,
                    request_id=request_id,
                )
            )
        return RepairResult(
            repaired_text=join_clauses(repaired),
            actions=tuple(actions),
            events=tuple(events),
        )

    def _fix_clause(self, clause: str) -> tuple[str, str, tuple[str, ...]]:
        evidence = self._retrieve(clause)
        if self._rewrite_fn is not None and evidence:
            texts = [text for _id, text in evidence]
            rewritten = self._rewrite_fn(clause, texts)
            if rewritten.strip():
                return rewritten, _REWRITE, tuple(eid for eid, _text in evidence)
        return self.redaction, _REDACT, ()

    def _retrieve(self, clause: str) -> list[tuple[str, str]]:
        if self._retrieve_fn is None:
            return []
        rows = self._retrieve_fn(clause)
        evidence: list[tuple[str, str]] = []
        for row in rows:
            evidence.append((_evidence_id(row), _evidence_text(row)))
        return evidence

    @staticmethod
    def _repair_event(
        *,
        index: int,
        action: str,
        support: float,
        evidence_ids: tuple[str, ...],
        tenant_id: str,
        request_id: str,
    ) -> SafetyEvent:
        explanation = (
            "An unsupported clause was rewritten from grounded evidence."
            if action == _REWRITE
            else "An unsupported clause was removed."
        )
        return SafetyEvent.from_policy_decision(
            hook_id="streaming_repair",
            hook_scope="streaming",
            policy_decision="warn",
            halt_reason=f"unsupported_clause_{action}",
            tenant_safe_explanation=explanation,
            request_id=request_id,
            tenant_id=tenant_id,
            observed_score=max(0.0, min(1.0, support)),
            evidence_refs=evidence_ids,
            attributes={"clause_index": str(index), "action": action},
        )


def _evidence_id(row: Any) -> str:
    if isinstance(row, Mapping):
        return str(row.get("id") or row.get("source") or "")
    return str(getattr(row, "source", "") or getattr(row, "id", ""))


def _evidence_text(row: Any) -> str:
    if isinstance(row, Mapping):
        return str(row.get("text", ""))
    return str(getattr(row, "text", ""))
