# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — swarm-level coherence (cross-agent cascade halt)

"""Halt a multi-agent cascade the moment one agent contradicts another.

In a swarm, agents run in sequence and each builds on what the earlier ones
produced. If one agent asserts something that contradicts an established claim,
every downstream agent inherits the error — the contradiction *cascades*. This
monitor watches the running swarm: as each agent emits text, its claims are
checked against the claims every earlier agent already established, and the first
real contradiction halts the cascade before the next agent consumes the poisoned
context.

Contradiction is judged by an injected NLI scorer (the same
``ContradictionScorer`` the streaming halt uses): a new claim and a prior claim
are scored in both directions and the stronger ``P(contradiction)`` decides,
because "A contradicts B" and "B contradicts A" are scored separately. Each
flagged conflict carries its evidence — which agent contradicted which, the two
claims, the contradiction strength, and their lexical (topical) overlap via the
Rust ``rust_word_overlap`` kernel with a bit-exact Python fallback — so an
operator sees exactly where the swarm went incoherent.

Without an NLI scorer the monitor still accumulates claims and reports lexical
novelty, but cross-agent *contradiction* detection needs the scorer; that is
documented on :class:`SwarmCoherenceMonitor`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

try:
    from backfire_kernel import rust_word_overlap

    _RUST_SWARM = True
except ImportError:  # pragma: no cover - exercised on installs without the kernel
    rust_word_overlap = None
    _RUST_SWARM = False

__all__ = [
    "ContradictionEngine",
    "AgentMessage",
    "CascadeContradiction",
    "CoherenceUpdate",
    "SwarmCoherenceMonitor",
]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-zA-Z0-9']+")


@runtime_checkable
class ContradictionEngine(Protocol):
    """An NLI contradiction scorer (duck-typed against ``ContradictionScorer``)."""

    def contradiction(self, premise: str, hypothesis: str) -> float: ...

    @property
    def threshold(self) -> float: ...


def _lexical_overlap(text_a: str, text_b: str) -> float:
    """Lexical Jaccard overlap in ``[0, 1]`` (Rust fast path or Python).

    The Python fallback mirrors the Rust ``rust_word_overlap`` kernel exactly —
    case-folded, whitespace-split, punctuation retained on the token — so the two
    backends are bit-for-bit identical and the dispatch is purely a speed choice.
    """
    if _RUST_SWARM and rust_word_overlap is not None:
        return float(rust_word_overlap(text_a, text_b))
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    union = words_a | words_b
    return len(words_a & words_b) / len(union) if union else 0.0


def _split_claims(text: str, *, min_words: int = 3, cap: int = 24) -> list[str]:
    """Split *text* into claim-sized sentences, dropping fragments and capping count."""
    claims: list[str] = []
    for raw in _SENTENCE_SPLIT.split(text.strip()):
        sentence = raw.strip()
        if len(_WORD_RE.findall(sentence)) >= min_words:
            claims.append(sentence)
        if len(claims) >= cap:
            break
    return claims


@dataclass(frozen=True)
class AgentMessage:
    """One agent's output in the swarm, with its established claims."""

    agent_id: str
    claims: tuple[str, ...]


@dataclass(frozen=True)
class CascadeContradiction:
    """A cross-agent contradiction that halts the cascade, with evidence."""

    new_agent: str
    prior_agent: str
    new_claim: str
    prior_claim: str
    contradiction: float
    topical_overlap: float


@dataclass(frozen=True)
class CoherenceUpdate:
    """Result of observing one agent message."""

    agent_id: str
    coherence: float
    halted: bool
    contradictions: tuple[CascadeContradiction, ...]
    rationale: tuple[str, ...]


@dataclass
class SwarmCoherenceMonitor:
    """Stateful cross-agent contradiction monitor with cascade halt.

    Parameters
    ----------
    nli:
        Optional :class:`ContradictionEngine`. Cross-agent contradiction
        detection requires it; without it the monitor accumulates claims and
        reports lexical novelty but never flags a contradiction.
    contradiction_threshold:
        Minimum ``P(contradiction)`` for a claim pair to halt the cascade.
        Defaults to the injected scorer's own ``threshold``.
    max_claims:
        Cap on accumulated established claims, oldest dropped first, to bound the
        per-message O(new · established) comparison in a long-running swarm.

    Once a contradiction halts the cascade, the monitor stays halted: further
    :meth:`observe` calls short-circuit and return the halted state, since the
    point is to stop downstream agents from consuming a contradicted context.
    Call :meth:`reset` to start a new cascade.
    """

    nli: ContradictionEngine | None = None
    contradiction_threshold: float | None = None
    max_claims: int = 256
    _established: list[tuple[str, str]] = field(default_factory=list, repr=False)
    _claim_total: int = field(default=0, repr=False)
    _contradicted: int = field(default=0, repr=False)
    _halted: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.contradiction_threshold is not None and not (
            0.0 <= self.contradiction_threshold <= 1.0
        ):
            raise ValueError("contradiction_threshold must be in [0, 1]")
        if self.max_claims < 1:
            raise ValueError("max_claims must be positive")

    @property
    def halted(self) -> bool:
        """True once a contradiction has halted the cascade."""
        return self._halted

    @property
    def _flag_threshold(self) -> float:
        if self.contradiction_threshold is not None:
            return self.contradiction_threshold
        if self.nli is not None:
            return float(self.nli.threshold)
        return 0.5

    def _coherence(self) -> float:
        if self._claim_total == 0:
            return 1.0
        return round(1.0 - self._contradicted / self._claim_total, 4)

    def observe(self, agent_id: str, text: str) -> CoherenceUpdate:
        """Observe one agent's output; update state and report cascade status."""
        if self._halted:
            return CoherenceUpdate(
                agent_id=agent_id,
                coherence=self._coherence(),
                halted=True,
                contradictions=(),
                rationale=("cascade already halted",),
            )

        new_claims = _split_claims(text)
        found: list[CascadeContradiction] = []
        flag = self._flag_threshold

        if self.nli is not None:
            for claim in new_claims:
                for prior_agent, prior_claim in self._established:
                    if prior_agent == agent_id:
                        continue
                    contra = max(
                        self.nli.contradiction(prior_claim, claim),
                        self.nli.contradiction(claim, prior_claim),
                    )
                    if contra >= flag:
                        found.append(
                            CascadeContradiction(
                                new_agent=agent_id,
                                prior_agent=prior_agent,
                                new_claim=claim,
                                prior_claim=prior_claim,
                                contradiction=round(contra, 4),
                                topical_overlap=round(
                                    _lexical_overlap(claim, prior_claim), 4
                                ),
                            )
                        )

        self._claim_total += len(new_claims)
        for claim in new_claims:
            self._established.append((agent_id, claim))
        if len(self._established) > self.max_claims:
            del self._established[: len(self._established) - self.max_claims]

        if found:
            self._contradicted += len(found)
            self._halted = True
            found.sort(key=lambda c: c.contradiction, reverse=True)
            rationale = (f"{len(found)} cross-agent contradiction(s) — cascade halted",)
        elif self.nli is None:
            rationale = ("lexical novelty only (no NLI scorer supplied)",)
        else:
            rationale = ("coherent with established claims",)

        return CoherenceUpdate(
            agent_id=agent_id,
            coherence=self._coherence(),
            halted=self._halted,
            contradictions=tuple(found),
            rationale=rationale,
        )

    def observe_message(self, message: AgentMessage) -> CoherenceUpdate:
        """Observe a pre-split :class:`AgentMessage` (claims joined as one text)."""
        return self.observe(message.agent_id, " ".join(message.claims))

    def reset(self) -> None:
        """Clear all accumulated state and un-halt for a fresh cascade."""
        self._established.clear()
        self._claim_total = 0
        self._contradicted = 0
        self._halted = False
