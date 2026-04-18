# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ProvenanceVerifier

"""Compose :class:`MerkleTree` + :class:`ProvenanceChain` +
:class:`SourceCredibility` into a single verify-and-score API.

The verifier takes a response's fact list and returns a
:class:`ProvenanceVerdict` with:

* ``merkle_root`` — root hash of this response's Merkle tree.
* ``chain_entry`` — the chain entry appended for this root.
* ``fact_verdicts`` — per-fact integrity + credibility record.
* ``trust_score`` — weighted mean of per-source credibility
  across all facts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .chain import ProvenanceChain, ProvenanceEntry
from .credibility import SourceCredibility
from .facts import CitationFact, FactVerificationError
from .merkle import MerkleTree


@dataclass(frozen=True)
class FactVerdict:
    """Per-fact outcome."""

    fact: CitationFact
    integrity_ok: bool
    source_score: float
    reason: str = ""


@dataclass(frozen=True)
class ProvenanceVerdict:
    """Result of one :meth:`ProvenanceVerifier.verify` call."""

    merkle_root: str
    chain_entry: ProvenanceEntry
    fact_verdicts: tuple[FactVerdict, ...]
    trust_score: float = 0.0
    failures: tuple[FactVerdict, ...] = field(default_factory=tuple)

    @property
    def all_ok(self) -> bool:
        return not self.failures


class ProvenanceVerifier:
    """Verify a response's citation set and fold it into the
    provenance chain.

    Parameters
    ----------
    chain :
        The :class:`ProvenanceChain` this verifier appends to.
    credibility :
        The :class:`SourceCredibility` tracker.
    min_source_score :
        Per-source score below which the fact is flagged as a
        credibility failure. Default 0.2.
    """

    def __init__(
        self,
        *,
        chain: ProvenanceChain,
        credibility: SourceCredibility,
        min_source_score: float = 0.2,
    ) -> None:
        if not 0.0 <= min_source_score <= 1.0:
            raise ValueError("min_source_score must be in [0, 1]")
        self._chain = chain
        self._credibility = credibility
        self._min_source_score = min_source_score

    def verify(self, facts: Sequence[CitationFact]) -> ProvenanceVerdict:
        if not facts:
            raise ValueError("facts must be non-empty")
        verdicts: list[FactVerdict] = []
        failures: list[FactVerdict] = []
        trust_sum = 0.0
        for fact in facts:
            integrity_ok = True
            reason = ""
            try:
                fact.verify_integrity()
            except FactVerificationError as exc:
                integrity_ok = False
                reason = str(exc)
            score = self._credibility.score(fact.source_id)
            verdict = FactVerdict(
                fact=fact,
                integrity_ok=integrity_ok,
                source_score=score,
                reason=reason
                or ("below minimum" if score < self._min_source_score else ""),
            )
            verdicts.append(verdict)
            trust_sum += score
            if not integrity_ok or score < self._min_source_score:
                failures.append(verdict)
        tree = MerkleTree(tuple(facts))
        chain_entry = self._chain.append(merkle_root=tree.root)
        trust_score = trust_sum / len(facts)
        return ProvenanceVerdict(
            merkle_root=tree.root,
            chain_entry=chain_entry,
            fact_verdicts=tuple(verdicts),
            trust_score=trust_score,
            failures=tuple(failures),
        )
