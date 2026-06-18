# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Answer Bill of Materials schema

"""Machine-readable per-response evidence manifest.

An :class:`AnswerBOM` is the "bill of materials" for one guarded answer: which
model produced it, which scorer judged it against which threshold, and — claim
by claim — what evidence supports each claim, how strongly, and the verdict the
guard reached. It is the artefact a reviewer or auditor reads to answer *"which
parts of this answer are grounded, and in what?"* without re-running the guard.

The schema is versioned and round-trips through :meth:`AnswerBOM.to_dict` /
:meth:`AnswerBOM.from_dict` so it can be written to an audit log and read back.
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

__all__ = [
    "ANSWER_BOM_SCHEMA_VERSION",
    "CLAIM_VERDICTS",
    "AnswerBOM",
    "ClaimRecord",
    "new_answer_id",
    "utc_timestamp",
]

ANSWER_BOM_SCHEMA_VERSION = "director.answer_bom.v1"
CLAIM_VERDICTS = frozenset({"supported", "unsupported", "contradicted"})


def utc_timestamp() -> str:
    """Return an RFC-3339 UTC timestamp for a manifest."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def new_answer_id() -> str:
    """Return an opaque identifier for one answer manifest."""
    return f"abom_{uuid.uuid4().hex}"


@dataclass(frozen=True)
class ClaimRecord:
    """Evidence record for one atomic claim in the answer.

    Parameters
    ----------
    claim:
        The atomic claim text extracted from the answer.
    verdict:
        One of ``supported`` / ``unsupported`` / ``contradicted``.
    support:
        Support strength in ``[0, 1]`` (``1.0`` = fully supported by evidence).
    evidence_ids:
        Identifiers of the evidence the claim was attributed to. Empty when the
        claim was attributed to no evidence (an unsupported claim).
    freshness:
        Freshness marker for the supporting evidence (e.g. an ISO date or a
        bucket like ``"current"``); empty when unknown.
    tenant:
        The tenant the answer was produced for.
    policy_refs:
        Policy identifiers this claim is governed by, if any.
    """

    claim: str
    verdict: str
    support: float
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    freshness: str = ""
    tenant: str = ""
    policy_refs: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate the verdict, support score, and immutable tuple fields."""
        if self.verdict not in CLAIM_VERDICTS:
            raise ValueError(f"unsupported claim verdict {self.verdict!r}")
        if not math.isfinite(self.support) or not 0.0 <= self.support <= 1.0:
            raise ValueError("support must be a finite value in [0, 1]")
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(self, "policy_refs", tuple(self.policy_refs))

    @property
    def supported(self) -> bool:
        """Whether the verdict is ``supported``."""
        return self.verdict == "supported"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {
            "claim": self.claim,
            "verdict": self.verdict,
            "support": self.support,
            "evidence_ids": list(self.evidence_ids),
            "freshness": self.freshness,
            "tenant": self.tenant,
            "policy_refs": list(self.policy_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ClaimRecord:
        """Reconstruct a claim record from its serialised form."""
        return cls(
            claim=str(payload["claim"]),
            verdict=str(payload["verdict"]),
            support=float(payload["support"]),
            evidence_ids=tuple(str(e) for e in payload.get("evidence_ids", ())),
            freshness=str(payload.get("freshness", "")),
            tenant=str(payload.get("tenant", "")),
            policy_refs=tuple(str(p) for p in payload.get("policy_refs", ())),
        )


@dataclass(frozen=True)
class AnswerBOM:
    """Per-response evidence manifest.

    Parameters
    ----------
    answer_id:
        Opaque identifier for this answer.
    model:
        Identifier of the model that produced the answer.
    scorer:
        Identifier of the scorer that judged it.
    threshold:
        The approval threshold the scorer applied, in ``[0, 1]``.
    claims:
        One :class:`ClaimRecord` per atomic claim.
    tenant:
        The tenant the answer was produced for.
    timestamp:
        RFC-3339 UTC creation time.
    schema_version:
        Manifest schema version.
    """

    answer_id: str
    model: str
    scorer: str
    threshold: float
    claims: tuple[ClaimRecord, ...] = field(default_factory=tuple)
    tenant: str = ""
    timestamp: str = ""
    schema_version: str = ANSWER_BOM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate manifest identity fields and fill a missing timestamp."""
        if self.schema_version != ANSWER_BOM_SCHEMA_VERSION:
            raise ValueError("unsupported AnswerBOM schema_version")
        if not self.answer_id.strip():
            raise ValueError("answer_id is required")
        if not math.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be a finite value in [0, 1]")
        object.__setattr__(self, "claims", tuple(self.claims))
        if not self.timestamp.strip():
            object.__setattr__(self, "timestamp", utc_timestamp())

    @property
    def unsupported_claims(self) -> tuple[ClaimRecord, ...]:
        """Claims whose verdict is not ``supported``."""
        return tuple(c for c in self.claims if not c.supported)

    @property
    def support_coverage(self) -> float:
        """Fraction of claims that are supported, in ``[0, 1]``.

        ``1.0`` for an answer with no claims (nothing was left unsupported).
        """
        if not self.claims:
            return 1.0
        supported = sum(1 for c in self.claims if c.supported)
        return supported / len(self.claims)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the whole manifest to a JSON-safe dict."""
        return {
            "schema_version": self.schema_version,
            "answer_id": self.answer_id,
            "timestamp": self.timestamp,
            "model": self.model,
            "scorer": self.scorer,
            "threshold": self.threshold,
            "tenant": self.tenant,
            "claims": [c.to_dict() for c in self.claims],
            "unsupported_claims": [c.claim for c in self.unsupported_claims],
            "support_coverage": self.support_coverage,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialise the manifest to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AnswerBOM:
        """Reconstruct a manifest from its serialised form.

        Ignores derived fields (``unsupported_claims``, ``support_coverage``);
        they are recomputed from ``claims``.
        """
        claims = payload.get("claims", ())
        if not isinstance(claims, Sequence) or isinstance(claims, str | bytes):
            raise ValueError("claims must be an array")
        return cls(
            answer_id=str(payload["answer_id"]),
            model=str(payload.get("model", "")),
            scorer=str(payload.get("scorer", "")),
            threshold=float(payload["threshold"]),
            claims=tuple(ClaimRecord.from_dict(c) for c in claims),
            tenant=str(payload.get("tenant", "")),
            timestamp=str(payload.get("timestamp", "")),
            schema_version=str(
                payload.get("schema_version", ANSWER_BOM_SCHEMA_VERSION)
            ),
        )
