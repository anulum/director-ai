# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — counterfactual contradiction injection

"""Turn a grounded (supported) claim into one that contradicts its grounding.

The AggreFact ``unsupported`` class mixes genuine contradictions with merely
unsupported claims, so recall measured against it is a muddy lower bound. To
measure contradiction recall cleanly we synthesise guaranteed contradictions:
take a claim the document *supports* and apply a meaning-flipping edit, so the
edited claim contradicts the same document. Scoring the original gives a clean
false-halt rate; scoring the injected variant gives a clean recall.

Four deterministic strategies, applied in priority order until one changes the
claim:

* ``negation`` — flip the polarity of the first copula/auxiliary ("was" →
  "was not"), or strip an existing negation;
* ``antonym`` — replace the first word with a curated antonym ("rose" → "fell");
* ``numeric`` — perturb the first number by a large, format-preserving delta
  ("five" stays text, "2014" → "2004", "37%" → "touched" differently);
* ``quantifier`` — swap a universal/existential quantifier ("all" → "no").

Each edit changes a fact the document asserts, so the result is a contradiction
rather than a merely-unsupported addition. Not every edit is guaranteed to
contradict (a perturbed number absent from the document is unsupported, not
contradictory), so injected recall is an estimate; the strategy mix and the
per-strategy counts are reported so the estimate is auditable.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["InjectionResult", "ContradictionInjector"]

# Bidirectional antonym pairs. Each maps to its opposite; the table is expanded
# to both directions at construction so either member triggers a flip.
_ANTONYM_PAIRS: tuple[tuple[str, str], ...] = (
    ("increase", "decrease"),
    ("increased", "decreased"),
    ("increases", "decreases"),
    ("rise", "fall"),
    ("rose", "fell"),
    ("rises", "falls"),
    ("rising", "falling"),
    ("grew", "shrank"),
    ("grow", "shrink"),
    ("gained", "lost"),
    ("win", "lose"),
    ("won", "lost"),
    ("wins", "loses"),
    ("open", "close"),
    ("opened", "closed"),
    ("accept", "reject"),
    ("accepted", "rejected"),
    ("approve", "reject"),
    ("approved", "rejected"),
    ("support", "oppose"),
    ("supported", "opposed"),
    ("agree", "disagree"),
    ("agreed", "disagreed"),
    ("include", "exclude"),
    ("included", "excluded"),
    ("succeed", "fail"),
    ("succeeded", "failed"),
    ("success", "failure"),
    ("true", "false"),
    ("alive", "dead"),
    ("higher", "lower"),
    ("highest", "lowest"),
    ("more", "fewer"),
    ("most", "least"),
    ("before", "after"),
    ("above", "below"),
    ("positive", "negative"),
    ("present", "absent"),
    ("possible", "impossible"),
    ("legal", "illegal"),
    ("guilty", "innocent"),
    ("married", "divorced"),
    ("began", "ended"),
    ("start", "end"),
    ("started", "ended"),
)

_QUANTIFIER_PAIRS: tuple[tuple[str, str], ...] = (
    ("all", "no"),
    ("every", "no"),
    ("always", "never"),
    ("everyone", "no one"),
    ("everybody", "nobody"),
    ("none", "all"),
    ("never", "always"),
)

# Copula / auxiliary verbs that take a trailing "not" to flip polarity.
_AUX_NEGATE: dict[str, str] = {
    "is": "is not",
    "are": "are not",
    "was": "was not",
    "were": "were not",
    "has": "has not",
    "have": "have not",
    "had": "had not",
    "can": "cannot",
    "could": "could not",
    "will": "will not",
    "would": "would not",
    "should": "should not",
    "must": "must not",
    "does": "does not",
    "did": "did not",
}
_NEGATION_TOKENS = ("not", "never", "no longer")

_NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")


def _build_word_map(pairs: Sequence[tuple[str, str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for a, b in pairs:
        out.setdefault(a, b)
        out.setdefault(b, a)
    return out


@dataclass(frozen=True)
class InjectionResult:
    """A claim and its meaning-flipped, contradicting counterpart."""

    original: str
    perturbed: str
    strategy: str
    changed: bool


class ContradictionInjector:
    """Synthesise a contradiction from a grounded claim.

    Parameters
    ----------
    strategies:
        Strategy names tried in order until one changes the claim. Defaults to
        ``("negation", "antonym", "numeric", "quantifier")``.
    """

    _DEFAULT = ("negation", "antonym", "numeric", "quantifier")

    def __init__(self, strategies: Sequence[str] | None = None) -> None:
        self._strategies = tuple(strategies) if strategies else self._DEFAULT
        self._antonyms = _build_word_map(_ANTONYM_PAIRS)
        self._quantifiers = _build_word_map(_QUANTIFIER_PAIRS)

    def inject(self, claim: str) -> InjectionResult:
        """Return the first strategy's contradicting edit, or unchanged."""
        for name in self._strategies:
            edited = getattr(self, f"_{name}")(claim)
            if edited is not None and edited != claim:
                return InjectionResult(claim, edited, name, True)
        return InjectionResult(claim, claim, "", False)

    def inject_batch(self, claims: Sequence[str]) -> list[InjectionResult]:
        return [self.inject(c) for c in claims]

    def inject_all(self, claim: str) -> list[InjectionResult]:
        """Every applicable strategy's edit — at most one per strategy.

        Unlike :meth:`inject` (first match only), this lets one claim contribute
        a variant to each strategy it supports, for balanced per-strategy recall
        and richer training data.
        """
        out: list[InjectionResult] = []
        for name in self._strategies:
            edited = getattr(self, f"_{name}")(claim)
            if edited is not None and edited != claim:
                out.append(InjectionResult(claim, edited, name, True))
        return out

    def _replace_word(self, text: str, mapping: dict[str, str]) -> str | None:
        """Replace the first whole-word hit in *mapping*, case-insensitively."""
        for match in re.finditer(r"[A-Za-z][A-Za-z'-]*", text):
            repl = mapping.get(match.group(0).lower())
            if repl is not None:
                return text[: match.start()] + repl + text[match.end() :]
        return None

    def _negation(self, claim: str) -> str | None:
        # Strip an existing negation first (negative → positive is a clean flip).
        for token in _NEGATION_TOKENS:
            stripped = re.sub(
                rf"\b{re.escape(token)}\b\s*", "", claim, count=1, flags=re.IGNORECASE
            )
            if stripped != claim:
                return re.sub(r"\s{2,}", " ", stripped).strip()
        # Else insert "not" after the first copula/auxiliary.
        for match in re.finditer(r"[A-Za-z]+", claim):
            repl = _AUX_NEGATE.get(match.group(0).lower())
            if repl is not None:
                return claim[: match.start()] + repl + claim[match.end() :]
        return None

    def _antonym(self, claim: str) -> str | None:
        return self._replace_word(claim, self._antonyms)

    def _quantifier(self, claim: str) -> str | None:
        return self._replace_word(claim, self._quantifiers)

    def _numeric(self, claim: str) -> str | None:
        match = _NUMBER_RE.search(claim)
        if match is None:
            return None
        raw = match.group(0)
        digits = raw.replace(",", "")
        if "." in digits:
            value = float(digits)
            new = f"{value + 100.0 if value < 100 else value / 2.0:.1f}"
        else:
            n = int(digits)
            # Years shift by a decade; other integers take a large, unambiguous
            # delta so the perturbed value plainly contradicts the original.
            new = str(n - 10) if 1900 <= n <= 2099 else str(n * 2 + 7)
        if new == raw:
            return None
        return claim[: match.start()] + new + claim[match.end() :]
