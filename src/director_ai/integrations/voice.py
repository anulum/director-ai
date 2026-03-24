# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Voice AI streaming filter — real-time hallucination prevention for TTS pipelines.

Sits between LLM text generation and TTS synthesis. Scores tokens as they
arrive and halts the stream before hallucinated text reaches the speaker.

Usage::

    from director_ai.integrations.voice import VoiceGuard

    guard = VoiceGuard(facts={"refund": "30-day refund policy"})

    # Filter an LLM token stream before TTS
    for token in llm_stream:
        result = guard.feed(token)
        if result.halted:
            tts.stop()
            tts.speak(result.recovery_text)
            break
        tts.speak_chunk(result.token)
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

from director_ai.core.retrieval.knowledge import GroundTruthStore
from director_ai.core.scoring.scorer import CoherenceScorer

__all__ = ["VoiceGuard", "VoiceToken"]

_SENTENCE_ENDS = frozenset(".!?")


@dataclass
class VoiceToken:
    """Result of feeding one token to the voice guard."""

    token: str
    index: int
    approved: bool
    coherence: float
    halted: bool = False
    halt_reason: str = ""
    recovery_text: str = ""


class VoiceGuard:
    """Real-time token filter for voice AI pipelines.

    Accumulates tokens, scores the growing text against a knowledge base
    (or prompt-only NLI), and signals halt before hallucinated content
    reaches the TTS engine.

    Parameters
    ----------
    facts : dict or GroundTruthStore — grounding knowledge.
    threshold : float — coherence floor (default 0.3).
    score_every : int — score every N-th token (default 4, ~20ms at 5 tokens/s).
    hard_limit : float — immediate halt below this score (default 0.25).
    window_size : int — sliding window for trend detection (default 8).
    soft_halt : bool — if True, finish current sentence before halting.
    recovery : str — text spoken when halt fires (default: brief apology).
    use_nli : bool — enable NLI model (default True).

    """

    def __init__(
        self,
        facts=None,
        store=None,
        threshold=0.3,
        score_every=4,
        hard_limit=0.25,
        window_size=8,
        soft_halt=True,
        recovery="I need to verify that information. One moment.",
        use_nli=True,
        prompt="",
    ):
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

        self._lock = threading.Lock()
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
        with self._lock:
            self._tokens.clear()
            self._scores.clear()
            self._index = 0
            self._halted = False
            self._pending_halt = False
            self._last_score = 1.0

    def feed(self, token: str) -> VoiceToken:
        """Feed one token from the LLM stream. Returns scoring result.

        Call this for every token the LLM generates. Pass the returned
        token to TTS only if ``result.halted`` is False.
        """
        with self._lock:
            return self._feed_locked(token)

    def _feed_locked(self, token: str) -> VoiceToken:
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

        # Pending soft halt — wait for sentence boundary
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

        # Score at cadence
        if self._index % self._score_every != 0:
            return VoiceToken(
                token=token,
                index=self._index - 1,
                approved=True,
                coherence=self._last_score,
            )

        # Score accumulated text
        text = self.accumulated_text
        premise = self._prompt if self._prompt else text
        _approved, cs = self._scorer.review(premise, text)
        score = cs.score
        self._last_score = score
        self._scores.append(score)

        # Hard halt — immediate
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

        # Window average check
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

    def feed_all(self, tokens) -> list[VoiceToken]:
        """Feed multiple tokens, return results. Stops after halt."""
        results = []
        for token in tokens:
            result = self.feed(token)
            results.append(result)
            if result.halted:
                break
        return results
