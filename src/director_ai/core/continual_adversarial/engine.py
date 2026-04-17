# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ContinualEngine

"""Orchestrate mining + suite promotion + scorer retraining.

The engine reads recent failure events from a
:class:`FailureStore`, asks :class:`PatternMiner` for the mined
patterns, derives a fresh :class:`SuiteVersion`, promotes it
into :class:`AdversarialSuite`, and retrains the
:class:`PerceptronAdversaryScorer` against the combined
adversarial + safe corpus.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field

from .failure import FailureEvent, FailureStore
from .miner import FailurePattern, PatternMiner
from .scorer import PerceptronAdversaryScorer, TrainedAdversaryScorer
from .suite import AdversarialCase, AdversarialSuite, SuiteVersion


@dataclass(frozen=True)
class EvolveReport:
    """Outcome of one :meth:`ContinualEngine.evolve` call."""

    version: SuiteVersion
    scorer: TrainedAdversaryScorer
    mined_pattern_count: int
    adversarial_case_count: int
    promotion_reason: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


class ContinualEngine:
    """Mine recent failures, promote a new suite, retrain the
    adversary scorer.

    Parameters
    ----------
    store :
        :class:`FailureStore` with the production failure events.
    miner :
        :class:`PatternMiner`. Default miner with ``ngram_size=3``
        and ``min_support=2``.
    suite :
        :class:`AdversarialSuite` to promote into. Default a
        fresh registry.
    scorer_trainer :
        :class:`PerceptronAdversaryScorer` used to retrain on
        every evolve. Default a fresh trainer.
    window_last_n :
        How many recent failures the miner processes per evolve.
        Default 512.
    min_failures :
        Minimum failure count required before evolve will run.
        Default 16.
    """

    def __init__(
        self,
        *,
        store: FailureStore,
        miner: PatternMiner | None = None,
        suite: AdversarialSuite | None = None,
        scorer_trainer: PerceptronAdversaryScorer | None = None,
        window_last_n: int = 512,
        min_failures: int = 16,
    ) -> None:
        if window_last_n <= 0:
            raise ValueError("window_last_n must be positive")
        if min_failures <= 0:
            raise ValueError("min_failures must be positive")
        self._store = store
        self._miner = miner or PatternMiner()
        self._suite = suite or AdversarialSuite()
        self._trainer = scorer_trainer or PerceptronAdversaryScorer()
        self._window = window_last_n
        self._min_failures = min_failures
        self._lock = threading.Lock()
        self._next_version = 1

    @property
    def suite(self) -> AdversarialSuite:
        return self._suite

    def evolve(
        self, *, safe_corpus: Sequence[str]
    ) -> EvolveReport:
        """Run one cycle. ``safe_corpus`` is a list of prompts the
        guardrail believes are safe — they supply the negative
        class for the perceptron."""
        if not safe_corpus:
            raise ValueError("safe_corpus must be non-empty")
        if len(self._store) < self._min_failures:
            raise ValueError(
                f"failure store holds {len(self._store)} events; "
                f"need at least {self._min_failures}"
            )
        events = self._store.window(last_n=self._window)
        patterns = self._miner.mine(events)
        if not patterns:
            raise ValueError(
                "miner returned no patterns — try a lower min_support"
            )
        cases = self._cases_from_patterns(patterns, events)
        with self._lock:
            version_number = self._next_version
            self._next_version += 1
        version = SuiteVersion(
            version=version_number,
            cases=cases,
            patterns=patterns,
            promotion_reason="continual evolve cycle",
        )
        self._suite.promote(version)
        scorer = self._trainer.train(
            adversarial=cases,
            safe=tuple(safe_corpus),
            version=version_number,
        )
        return EvolveReport(
            version=version,
            scorer=scorer,
            mined_pattern_count=len(patterns),
            adversarial_case_count=len(cases),
            promotion_reason=version.promotion_reason,
        )

    def _cases_from_patterns(
        self,
        patterns: Sequence[FailurePattern],
        events: Sequence[FailureEvent],
    ) -> tuple[AdversarialCase, ...]:
        """Turn mined patterns into concrete test cases. For each
        ngram pattern we use the pattern signature itself as the
        case prompt; for edit-distance clusters we use the
        cluster prototype. Pattern-to-case is 1-to-1."""
        cases: list[AdversarialCase] = []
        for pattern in patterns:
            prompt = pattern.prototype if pattern.kind == "edit_cluster" else pattern.signature
            cases.append(
                AdversarialCase(
                    prompt=prompt,
                    expected_label=pattern.label,
                    source_pattern=pattern.signature,
                )
            )
        # Deduplicate cases sharing the same prompt + label pair —
        # the keep-first-seen rule prefers the earlier (higher
        # support) pattern.
        seen: set[tuple[str, str]] = set()
        deduped: list[AdversarialCase] = []
        for case in cases:
            key = (case.prompt, case.expected_label)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(case)
        # Guard against the theoretical case where every case
        # deduplicates away.
        if not deduped:
            raise ValueError("no unique cases survived deduplication")
        # _events is intentionally unused by the default policy;
        # subclasses can override _cases_from_patterns to sample
        # real prompts from the event window.
        _ = events
        return tuple(deduped)
