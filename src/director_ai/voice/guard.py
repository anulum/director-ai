# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Async Voice Guard
"""Async token-by-token hallucination filter for voice AI pipelines.

Async-native counterpart of :class:`~director_ai.integrations.voice.VoiceGuard`.
Reuses the same :class:`~director_ai.core.scoring.scorer.CoherenceScorer` and
:class:`~director_ai.integrations.voice.VoiceToken` dataclass.

Usage::

    from director_ai.voice import AsyncVoiceGuard

    guard = AsyncVoiceGuard(facts={"refund": "30-day refund policy"})

    async for result in guard.feed_stream(llm_token_stream):
        if result.halted:
            await tts.stop()
            await tts.speak(result.recovery_text)
            break
        await tts.speak_chunk(result.token)
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Iterator

from director_ai.core.retrieval.knowledge import GroundTruthStore
from director_ai.core.scoring.scorer import CoherenceScorer
from director_ai.integrations.voice import VoiceToken

__all__ = ["AsyncVoiceGuard"]

_SENTENCE_ENDS = frozenset(".!?")


class AsyncVoiceGuard:
    """Async real-time token filter for voice AI pipelines.

    Accumulates tokens, scores the growing text against a knowledge base
    (or prompt-only NLI), and signals halt before hallucinated content
    reaches the TTS engine. Runs on a single event loop — no locks.

    Parameters
    ----------
    facts : dict or GroundTruthStore — grounding knowledge.
    threshold : float — coherence floor (default 0.3).
    score_every : int — score every N-th token (default 4).
    hard_limit : float — immediate halt below this score (default 0.25).
    window_size : int — sliding window for trend detection (default 8).
    soft_halt : bool — if True, finish current sentence before halting.
    recovery : str — text spoken when halt fires.
    use_nli : bool — enable NLI model (default True).

    """

    def __init__(
        self,
        facts: dict[str, str] | None = None,
        store: GroundTruthStore | None = None,
        threshold: float = 0.3,
        score_every: int = 4,
        hard_limit: float = 0.25,
        window_size: int = 8,
        soft_halt: bool = True,
        recovery: str = "I need to verify that information. One moment.",
        use_nli: bool = True,
        prompt: str = "",
    ) -> None:
        if store is not None:
            self._store = store
        else:
            self._store = GroundTruthStore()
            if facts:
                for k, v in facts.items():
                    self._store.add(k, v)

        self._scorer = CoherenceScorer(
            threshold=threshold,
            ground_truth_store=self._store,
            use_nli=use_nli,
        )
        self._prompt = prompt
        self._threshold = threshold
        self._hard_limit = hard_limit
        self._score_every = score_every
        self._window_size = window_size
        self._soft_halt = soft_halt
        self._recovery = recovery

        self._tokens: list[str] = []
        self._scores: deque[float] = deque(maxlen=window_size)
        self._index = 0
        self._halted = False
        self._pending_halt = False
        self._last_score = 1.0

    @property
    def accumulated_text(self) -> str:
        return "".join(self._tokens)

    @property
    def halted(self) -> bool:
        return self._halted

    def set_prompt(self, prompt: str) -> None:
        """Set the user prompt for the current utterance."""
        self._prompt = prompt

    def reset(self) -> None:
        """Reset state for a new utterance (new turn in conversation)."""
        self._tokens.clear()
        self._scores.clear()
        self._index = 0
        self._halted = False
        self._pending_halt = False
        self._last_score = 1.0

    async def feed(self, token: str) -> VoiceToken:
        """Feed one token. Returns scoring result.

        Scoring calls CoherenceScorer.review() which is synchronous.
        When NLI is enabled the scorer does CPU-heavy inference, so we
        offload to a thread to avoid blocking the event loop.
        """
        if self._halted:
            return VoiceToken(
                token=token,
                index=self._index,
                approved=False,
                coherence=0.0,
                halted=True,
                halt_reason="already_halted",
            )

        self._tokens.append(token)
        self._index += 1

        if self._pending_halt:
            stripped = token.rstrip()
            if stripped and stripped[-1] in _SENTENCE_ENDS:
                self._halted = True
                return VoiceToken(
                    token=token,
                    index=self._index - 1,
                    approved=True,
                    coherence=self._last_score,
                    halted=True,
                    halt_reason="soft_halt_sentence_end",
                    recovery_text=self._recovery,
                )
            return VoiceToken(
                token=token,
                index=self._index - 1,
                approved=True,
                coherence=self._last_score,
            )

        if self._index % self._score_every != 0:
            return VoiceToken(
                token=token,
                index=self._index - 1,
                approved=True,
                coherence=self._last_score,
            )

        text = self.accumulated_text
        premise = self._prompt if self._prompt else text
        _approved, cs = await asyncio.to_thread(
            self._scorer.review,
            premise,
            text,
        )
        score = cs.score
        self._last_score = score
        self._scores.append(score)

        if score < self._hard_limit:
            self._halted = True
            return VoiceToken(
                token=token,
                index=self._index - 1,
                approved=False,
                coherence=score,
                halted=True,
                halt_reason="hard_limit",
                recovery_text=self._recovery,
            )

        if len(self._scores) >= self._window_size:
            avg = sum(self._scores) / len(self._scores)
            if avg < self._threshold:
                if self._soft_halt:
                    self._pending_halt = True
                    return VoiceToken(
                        token=token,
                        index=self._index - 1,
                        approved=True,
                        coherence=score,
                    )
                self._halted = True
                return VoiceToken(
                    token=token,
                    index=self._index - 1,
                    approved=False,
                    coherence=score,
                    halted=True,
                    halt_reason="window_avg",
                    recovery_text=self._recovery,
                )

        return VoiceToken(
            token=token,
            index=self._index - 1,
            approved=True,
            coherence=score,
        )

    async def feed_stream(
        self,
        tokens: AsyncIterator[str] | Iterator[str],
    ) -> AsyncIterator[VoiceToken]:
        """Feed a token stream. Yields VoiceToken per token. Stops after halt."""
        if hasattr(tokens, "__aiter__"):
            async for token in tokens:
                result = await self.feed(token)
                yield result
                if result.halted:
                    return
        else:
            for token in tokens:
                result = await self.feed(token)
                yield result
                if result.halted:
                    return
