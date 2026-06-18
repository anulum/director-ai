# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — pre-generation hallucination forecasting

"""Score a prompt's hallucination risk *before* the model generates.

Every other guard in this package scores the *response*. The forecaster acts one
step earlier: given the prompt (and optionally the knowledge base it will be
grounded against) it estimates how likely the answer is to hallucinate, so a
caller can pre-emptively retrieve more context, route to a stronger model, or ask
for human review instead of paying for a generation that is likely to be halted.

Three measurable signals combine into a calibrated risk in ``[0, 1]``:

* **ambiguity** — an under-specified prompt (too few content words, vague terms,
  no concrete anchors like names or numbers, several stacked questions) gives the
  model room to invent;
* **kb_coverage** — when a knowledge base is supplied, the best lexical overlap
  between the prompt and the facts it would retrieve; low coverage means the
  answer cannot be grounded;
* **pattern_history** — the observed hallucination rate of past answers whose
  prompt shared this one's coarse shape (optional, learned online).

The lexical overlap behind ``kb_coverage`` uses the Rust ``rust_word_overlap``
kernel when the compiled extension is installed and a pure-Python Jaccard
fallback otherwise, so the forecaster runs everywhere and the fast path is used
when present.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from ..text_overlap import word_overlap

__all__ = [
    "ForecastResult",
    "ForecastHistory",
    "HallucinationForecaster",
]

_WORD_RE = re.compile(r"[a-zA-Z0-9']+")
_VAGUE_TERMS = frozenset(
    {
        "something",
        "stuff",
        "things",
        "thing",
        "whatever",
        "somehow",
        "someone",
        "anything",
        "everything",
        "etc",
        "some",
        "any",
        "various",
        "general",
    }
)
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "do",
        "does",
        "what",
        "how",
        "why",
        "when",
        "who",
        "tell",
        "me",
        "about",
        "please",
        "can",
        "you",
        "give",
        "i",
        "it",
    }
)


class _RetrievalStore(Protocol):
    """Typed retrieval boundary used for forecasting KB coverage."""

    def retrieve_context(self, prompt: str) -> str | None:
        """Return semicolon-separated grounding facts for ``prompt``."""
        ...


def _lexical_overlap(text_a: str, text_b: str) -> float:
    """Lexical Jaccard overlap in ``[0, 1]``.

    Delegates to the shared measured-fast-path helper (pure Python below a large
    -input threshold, Rust above it). See :mod:`director_ai.core.text_overlap`.
    """
    return word_overlap(text_a, text_b, logger_name=__name__)


def _as_facts(grounding: str | None) -> list[str]:
    if not grounding:
        return []
    return [part.strip() for part in grounding.split(";") if part.strip()]


@dataclass(frozen=True)
class ForecastResult:
    """A pre-generation hallucination-risk forecast for one prompt."""

    risk: float
    ambiguity: float
    kb_coverage: float | None
    pattern_risk: float
    recommendation: str
    rationale: tuple[str, ...]


class ForecastHistory:
    """Online memory of hallucination outcomes keyed by prompt shape.

    The signature is deliberately coarse — leading interrogative, content-word
    count bucket, and whether the prompt carries a concrete anchor — so a handful
    of observations generalise to similar prompts rather than memorising exact
    strings.
    """

    def __init__(self) -> None:
        self._counts: dict[str, list[int]] = {}  # signature -> [hallucinations, total]

    @staticmethod
    def signature(prompt: str) -> str:
        """Return the coarse prompt-shape signature used for history lookup."""
        words = _WORD_RE.findall(prompt.lower())
        lead = words[0] if words else ""
        content = [w for w in words if w not in _STOPWORDS]
        bucket = (
            "short" if len(content) < 4 else "medium" if len(content) < 12 else "long"
        )
        anchored = "anchored" if _has_anchor(prompt) else "vague"
        return f"{lead}|{bucket}|{anchored}"

    def record(self, prompt: str, *, hallucinated: bool) -> None:
        """Record one observed hallucination outcome for ``prompt``."""
        entry = self._counts.setdefault(self.signature(prompt), [0, 0])
        entry[0] += 1 if hallucinated else 0
        entry[1] += 1

    def rate(self, prompt: str) -> float | None:
        """Return the historical hallucination rate for ``prompt`` shape."""
        entry = self._counts.get(self.signature(prompt))
        if entry is None or entry[1] == 0:
            return None
        return entry[0] / entry[1]


def _has_anchor(prompt: str) -> bool:
    """Return whether the prompt carries a digit or likely proper noun."""
    if any(ch.isdigit() for ch in prompt):
        return True
    tokens = prompt.split()
    # A capitalised token that is not the first word is a likely proper noun.
    return any(tok[:1].isupper() for tok in tokens[1:] if tok[:1].isalpha())


@dataclass
class HallucinationForecaster:
    """Estimate a prompt's hallucination risk before generation.

    Parameters
    ----------
    weight_ambiguity / weight_kb / weight_history:
        Convex combination of the three signals; re-normalised on use so any
        non-negative triple is valid. History is dropped from the mix when no
        prior outcomes match the prompt, with its weight redistributed.
    ground_threshold / review_threshold:
        Risk band edges: below ``ground_threshold`` recommend ``proceed``; below
        ``review_threshold`` recommend ``ground`` (retrieve/augment); otherwise
        ``human_review``.
    no_kb_prior:
        kb-coverage risk assumed when no knowledge base is supplied.
    history:
        Optional :class:`ForecastHistory` for the online pattern signal.
    """

    weight_ambiguity: float = 0.4
    weight_kb: float = 0.45
    weight_history: float = 0.15
    ground_threshold: float = 0.34
    review_threshold: float = 0.66
    no_kb_prior: float = 0.5
    history: ForecastHistory | None = None
    _content_floor: int = field(default=4, repr=False)

    def __post_init__(self) -> None:
        """Validate weights, thresholds, and no-KB prior."""
        for name in ("weight_ambiguity", "weight_kb", "weight_history"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= self.ground_threshold <= self.review_threshold <= 1.0:
            raise ValueError("require 0 <= ground_threshold <= review_threshold <= 1")
        if not 0.0 <= self.no_kb_prior <= 1.0:
            raise ValueError("no_kb_prior must be in [0, 1]")

    def ambiguity(self, prompt: str) -> float:
        """Return an under-specification score in ``[0, 1]``."""
        words = _WORD_RE.findall(prompt.lower())
        if not words:
            return 1.0
        content = [w for w in words if w not in _STOPWORDS]
        brevity = max(0.0, 1.0 - len(content) / self._content_floor)
        vague = sum(1 for w in words if w in _VAGUE_TERMS) / len(words)
        unanchored = 0.0 if _has_anchor(prompt) else 1.0
        multi_intent = 1.0 if prompt.count("?") > 1 else 0.0
        score = (
            0.4 * brevity
            + 0.3 * min(1.0, vague * 4.0)
            + 0.2 * unanchored
            + 0.1 * multi_intent
        )
        return max(0.0, min(1.0, score))

    def kb_coverage(self, prompt: str, store: _RetrievalStore | None) -> float | None:
        """Best lexical overlap of *prompt* with the facts *store* would retrieve."""
        if store is None:
            return None
        facts = _as_facts(store.retrieve_context(prompt))
        if not facts:
            return 0.0
        return max(_lexical_overlap(prompt, fact) for fact in facts)

    def forecast(
        self, prompt: str, *, store: _RetrievalStore | None = None
    ) -> ForecastResult:
        """Return a :class:`ForecastResult` for *prompt*."""
        ambiguity = self.ambiguity(prompt)
        coverage = self.kb_coverage(prompt, store)
        kb_risk = (1.0 - coverage) if coverage is not None else self.no_kb_prior
        history_rate = self.history.rate(prompt) if self.history is not None else None

        w_amb, w_kb, w_hist = (
            self.weight_ambiguity,
            self.weight_kb,
            self.weight_history,
        )
        if history_rate is None:
            w_hist = 0.0
        total = w_amb + w_kb + w_hist
        if total <= 0:
            risk = 0.0
        else:
            risk = (
                w_amb * ambiguity + w_kb * kb_risk + w_hist * (history_rate or 0.0)
            ) / total
        risk = max(0.0, min(1.0, risk))

        if risk < self.ground_threshold:
            recommendation = "proceed"
        elif risk < self.review_threshold:
            recommendation = "ground"
        else:
            recommendation = "human_review"

        rationale: list[str] = []
        if ambiguity >= 0.5:
            rationale.append("under-specified prompt")
        if coverage is not None and coverage < 0.25:
            rationale.append("weak knowledge-base coverage")
        if coverage is None:
            rationale.append("no knowledge base supplied")
        if history_rate is not None and history_rate >= 0.5:
            rationale.append("this prompt shape has hallucinated before")
        if not rationale:
            rationale.append("well-specified and grounded")

        return ForecastResult(
            risk=round(risk, 4),
            ambiguity=round(ambiguity, 4),
            kb_coverage=None if coverage is None else round(coverage, 4),
            pattern_risk=round(history_rate or 0.0, 4),
            recommendation=recommendation,
            rationale=tuple(rationale),
        )
