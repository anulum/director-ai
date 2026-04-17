# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — PatternMiner

"""Mine recurring adversarial patterns from failure events.

Two passes:

* **N-gram frequency** — tokenises each prompt with lowercase +
  whitespace split, extracts ``ngram_size``-grams, and keeps the
  grams whose document frequency meets ``min_support``.
* **Edit-distance clustering** — normalised Levenshtein distance
  groups prompts that share structure even when their n-grams
  diverge. Each representative carries a ``support`` count and a
  prototype prompt.

The miner returns both signals as :class:`FailurePattern`
records so downstream components can treat structural and
lexical patterns on the same footing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .failure import FailureEvent

PatternKind = Literal["ngram", "edit_cluster"]


@dataclass(frozen=True)
class FailurePattern:
    """One mined pattern."""

    kind: PatternKind
    signature: str
    support: int
    label: str
    prototype: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            raise ValueError("FailurePattern.signature must be non-empty")
        if self.support <= 0:
            raise ValueError("FailurePattern.support must be positive")
        if self.kind not in {"ngram", "edit_cluster"}:
            raise ValueError(f"unknown kind {self.kind!r}")


class PatternMiner:
    """Mine :class:`FailurePattern` records from a batch of events.

    Parameters
    ----------
    ngram_size :
        Width of the token n-grams to mine. Default 3.
    min_support :
        Minimum document-frequency for an n-gram to be kept. Default 2.
    max_edit_distance :
        Maximum normalised Levenshtein distance (distance / max
        length) for two prompts to join the same cluster. Default
        0.35.
    lowercase :
        Whether tokenisation lowercases the prompt first. Default
        ``True`` so case variation does not fragment patterns.
    """

    def __init__(
        self,
        *,
        ngram_size: int = 3,
        min_support: int = 2,
        max_edit_distance: float = 0.35,
        lowercase: bool = True,
    ) -> None:
        if ngram_size <= 0:
            raise ValueError("ngram_size must be positive")
        if min_support <= 0:
            raise ValueError("min_support must be positive")
        if not 0.0 <= max_edit_distance <= 1.0:
            raise ValueError("max_edit_distance must be in [0, 1]")
        self._ngram_size = ngram_size
        self._min_support = min_support
        self._max_edit = max_edit_distance
        self._lowercase = lowercase

    def mine(self, events: Sequence[FailureEvent]) -> tuple[FailurePattern, ...]:
        if not events:
            return ()
        ngram_patterns = self._mine_ngrams(events)
        cluster_patterns = self._mine_clusters(events)
        return ngram_patterns + cluster_patterns

    def _mine_ngrams(
        self, events: Sequence[FailureEvent]
    ) -> tuple[FailurePattern, ...]:
        counter: Counter[tuple[str, str]] = Counter()
        for event in events:
            tokens = self._tokenise(event.prompt)
            if len(tokens) < self._ngram_size:
                continue
            seen_in_event: set[tuple[str, str]] = set()
            for i in range(len(tokens) - self._ngram_size + 1):
                gram = " ".join(tokens[i : i + self._ngram_size])
                key = (gram, event.label)
                if key in seen_in_event:
                    continue
                seen_in_event.add(key)
                counter[key] += 1
        patterns: list[FailurePattern] = []
        for (gram, label), support in counter.most_common():
            if support < self._min_support:
                break
            patterns.append(
                FailurePattern(
                    kind="ngram",
                    signature=gram,
                    support=support,
                    label=label,
                )
            )
        return tuple(patterns)

    def _mine_clusters(
        self, events: Sequence[FailureEvent]
    ) -> tuple[FailurePattern, ...]:
        """Leader-based online clustering: each event either joins
        an existing cluster (nearest leader within
        ``max_edit_distance``) or spawns a new one with itself as
        the leader. Simple, deterministic, no RNG.
        """
        clusters: list[_LeaderCluster] = []
        for event in events:
            prompt_tokens = self._tokenise(event.prompt)
            if not prompt_tokens:
                continue
            assigned = False
            for cluster in clusters:
                distance = _normalised_edit_distance(
                    prompt_tokens, cluster.leader_tokens
                )
                if distance <= self._max_edit and cluster.label == event.label:
                    cluster.members += 1
                    assigned = True
                    break
            if not assigned:
                clusters.append(
                    _LeaderCluster(
                        leader_tokens=prompt_tokens,
                        leader_prompt=event.prompt,
                        label=event.label,
                        members=1,
                    )
                )
        patterns: list[FailurePattern] = []
        for cluster in clusters:
            if cluster.members < self._min_support:
                continue
            patterns.append(
                FailurePattern(
                    kind="edit_cluster",
                    signature=" ".join(cluster.leader_tokens),
                    support=cluster.members,
                    label=cluster.label,
                    prototype=cluster.leader_prompt,
                )
            )
        return tuple(patterns)

    def _tokenise(self, prompt: str) -> list[str]:
        text = prompt.lower() if self._lowercase else prompt
        return text.split()


@dataclass
class _LeaderCluster:
    """Mutable state for an in-progress leader-based cluster."""

    leader_tokens: list[str]
    leader_prompt: str
    label: str
    members: int


def _normalised_edit_distance(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    max_len = max(len(a), len(b))
    return _levenshtein(a, b) / max_len


def _levenshtein(a: list[str], b: list[str]) -> int:
    """Classical dynamic-programming edit distance at token
    granularity. Space O(min(|a|, |b|)) via the two-row trick so
    the miner stays cheap on long traces."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, token_b in enumerate(b, start=1):
            cost = 0 if token_a == token_b else 1
            current[j] = min(
                previous[j] + 1,  # deletion
                current[j - 1] + 1,  # insertion
                previous[j - 1] + cost,  # substitution
            )
        previous = current
    return previous[-1]
