# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — multi-turn transcript runner

"""Elicit a multi-turn, citing answer from a model under test.

HalluHard scores groundedness over a short conversation — a seed question
followed by a couple of follow-ups — where the model is asked to cite a source
for every factual claim. This runner conducts that conversation: it threads the
prior exchanges back into each prompt (so the model sees the dialogue), asks the
generator for one response per turn, and returns the :class:`Transcript`. The
concatenated responses are what
:class:`~director_ai.core.citation_grounding.judge.CitationGroundingJudge`
assesses.

The generator is injected through the :class:`Generator` protocol — satisfied by
:class:`~director_ai.core.actor.LLMGenerator` and
:class:`~director_ai.core.actor.MockGenerator` — so the conversation control flow
and prompt construction are deterministic and fully tested with a stub, no model
or network required.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "ExchangeTurn",
    "Generator",
    "MultiTurnRunner",
    "Transcript",
]

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful research assistant. Answer accurately and concisely. "
    "Support every factual claim with an inline citation marker like [1], and "
    "end your answer with a numbered References list giving each source's DOI, "
    "arXiv id, or URL."
)


class Generator(Protocol):
    """A text generator returning candidate dicts with a ``text`` key."""

    def generate_candidates(self, prompt: str, n: int) -> Sequence[Mapping[str, Any]]:
        """Generate up to ``n`` candidate responses for ``prompt``."""
        ...


@dataclass(frozen=True)
class ExchangeTurn:
    """One user prompt and the model's response to it."""

    prompt: str
    response: str

    def to_dict(self) -> dict[str, str]:
        """Serialise the exchange turn for transcript reports."""
        return {"prompt": self.prompt, "response": self.response}


@dataclass(frozen=True)
class Transcript:
    """A complete multi-turn exchange with a model under test."""

    turns: tuple[ExchangeTurn, ...]

    @property
    def full_text(self) -> str:
        """The model's responses concatenated — the text the judge assesses."""
        return "\n\n".join(t.response for t in self.turns if t.response)

    def to_dict(self) -> dict[str, object]:
        """Serialise every exchange turn in this transcript."""
        return {"turns": [t.to_dict() for t in self.turns]}


class MultiTurnRunner:
    """Run a model through a seed question and follow-ups, capturing the answers.

    Parameters
    ----------
    generator : Generator
        The model under test (e.g. an
        :class:`~director_ai.core.actor.LLMGenerator`).
    system_prompt : str
        Instruction prepended to every turn; the default asks for inline
        citations and a reference list. Pass ``""`` to omit it.
    """

    def __init__(
        self, *, generator: Generator, system_prompt: str = DEFAULT_SYSTEM_PROMPT
    ) -> None:
        self._generator = generator
        self._system_prompt = system_prompt

    def run(self, seed: str, followups: Sequence[str] = ()) -> Transcript:
        """Conduct the conversation and return its :class:`Transcript`.

        ``seed`` is the opening question; each entry in ``followups`` is asked in
        order with the running dialogue threaded into the prompt. An empty seed
        raises :class:`ValueError`.
        """
        if not seed.strip():
            raise ValueError("seed question must be non-empty")
        turns: list[ExchangeTurn] = []
        for user_prompt in (seed, *followups):
            model_input = self._build_input(turns, user_prompt)
            response = self._generate(model_input)
            turns.append(ExchangeTurn(user_prompt, response))
        return Transcript(tuple(turns))

    def _generate(self, model_input: str) -> str:
        candidates = self._generator.generate_candidates(model_input, 1)
        if not candidates:
            return ""
        return str(candidates[0].get("text", "")).strip()

    def _build_input(
        self, prior_turns: Sequence[ExchangeTurn], user_prompt: str
    ) -> str:
        """Render the dialogue so far plus the new user turn as a single prompt."""
        parts: list[str] = []
        if self._system_prompt:
            parts.append(self._system_prompt)
        for turn in prior_turns:
            parts.append(f"User: {turn.prompt}")
            parts.append(f"Assistant: {turn.response}")
        parts.append(f"User: {user_prompt}")
        parts.append("Assistant:")
        return "\n".join(parts)
