# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Rule-based scorer backend
"""Zero-dependency rule-based hallucination scorer.

Each rule checks a specific property of the (premise, hypothesis) pair
and returns a score in [0, 1] where 1 = fully grounded and 0 = flagged.
The backend aggregates rule scores with configurable weights.

This is the Guardrails AI-competitive tier: no ML models, no GPU,
sub-millisecond latency, ships in the base ``pip install director-ai``.

Usage::

    from director_ai.core.scoring.rules_scorer import RulesBackend

    backend = RulesBackend()
    score = backend.score("Water boils at 100 degrees.", "Water boils at 500 degrees.")
    # score ≈ 0.72 — the numeric-consistency rule flags the 100 vs 500 mismatch
"""

from __future__ import annotations

import abc
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..mandatory import mandatory_execution
from ._heuristics import STOP_WORDS
from .lexical_signals import entity_overlap as _py_entity_overlap
from .lexical_signals import negation_flip as _py_negation_flip
from .lexical_signals import numerical_consistency as _py_numerical_consistency
from .lexical_signals import word_overlap_jaccard as _py_word_overlap

# The lexical signals are accelerated by backfire-kernel. ADR-0001 makes the
# kernel the opt-in ``[rust]`` extra, so a base ``pip install director-ai`` runs
# the bit-exact pure-Python ports in ``lexical_signals`` — proven byte-for-byte
# against the kernel by ``tests/test_lexical_signals_parity.py``. The scorer's
# earlier Python fallbacks had *drifted* (entity recall vs Jaccard, a numeric
# ratio vs the shared-number boolean, F1 vs Jaccard word overlap); the ports
# restore parity so the kernel-present and floor results are identical.
try:  # pragma: no cover - optional acceleration
    from backfire_kernel import (
        rust_entity_overlap as _rust_entity_overlap,
    )
    from backfire_kernel import (
        rust_negation_flip as _rust_negation_flip,
    )
    from backfire_kernel import (
        rust_numerical_consistency as _rust_numerical_consistency,
    )
    from backfire_kernel import (
        rust_word_overlap as _rust_word_overlap,
    )

    _RUST_AVAILABLE = True
except ImportError:  # pragma: no cover - bit-exact pure-Python fallback (ADR-0001)
    _RUST_AVAILABLE = False
    _rust_entity_overlap = _py_entity_overlap
    _rust_negation_flip = _py_negation_flip
    _rust_numerical_consistency = _py_numerical_consistency
    _rust_word_overlap = _py_word_overlap


def _entity_overlap(premise: str, hypothesis: str) -> float:
    """Jaccard proper-noun overlap — kernel fast-path or the bit-exact port."""
    if _RUST_AVAILABLE:
        with mandatory_execution(__name__, component="mandatory accelerated path"):
            return float(_rust_entity_overlap(premise, hypothesis))
    return _py_entity_overlap(premise, hypothesis)


def _numerical_consistency(premise: str, hypothesis: str) -> bool | None:
    """Shared-number boolean (``None`` if a side has no numbers)."""
    if _RUST_AVAILABLE:
        with mandatory_execution(__name__, component="mandatory accelerated path"):
            result = _rust_numerical_consistency(premise, hypothesis)
            return None if result is None else bool(result)
    return _py_numerical_consistency(premise, hypothesis)


def _negation_flip(premise: str, hypothesis: str) -> bool:
    """Polarity asymmetry gated by three shared content words."""
    if _RUST_AVAILABLE:
        with mandatory_execution(__name__, component="mandatory accelerated path"):
            return bool(_rust_negation_flip(premise, hypothesis))
    return _py_negation_flip(premise, hypothesis)


def _word_overlap(premise: str, hypothesis: str) -> float:
    """Jaccard word overlap — kernel fast-path or the bit-exact port."""
    if _RUST_AVAILABLE:
        with mandatory_execution(__name__, component="mandatory accelerated path"):
            return float(_rust_word_overlap(premise, hypothesis))
    return _py_word_overlap(premise, hypothesis)


# ── Rule ABC + result ───────────────────────────────────────────────────


@dataclass
class RuleResult:
    """Outcome of a single rule check."""

    rule_name: str
    score: float  # 1.0 = pass, 0.0 = fail
    reason: str = ""


class Rule(abc.ABC):
    """Abstract rule that checks a (premise, hypothesis) pair."""

    name: str = "unnamed"
    weight: float = 1.0

    @abc.abstractmethod
    def check(self, premise: str, hypothesis: str) -> RuleResult:
        """Return a RuleResult with score in [0, 1]."""


# ── Concrete rules ──────────────────────────────────────────────────────


class EntityGroundingRule(Rule):
    """Named entities in hypothesis must appear in premise."""

    name = "entity_grounding"

    def check(self, premise: str, hypothesis: str) -> RuleResult:
        """Score the Jaccard overlap of premise/hypothesis proper nouns."""
        return RuleResult(self.name, _entity_overlap(premise, hypothesis))


class NumericConsistencyRule(Rule):
    """Numbers in hypothesis should appear in premise."""

    name = "numeric_consistency"

    def check(self, premise: str, hypothesis: str) -> RuleResult:
        """Score whether premise and hypothesis share at least one number."""
        result = _numerical_consistency(premise, hypothesis)
        if result is None:
            return RuleResult(self.name, 1.0, "no numbers")
        return RuleResult(
            self.name,
            1.0 if result else 0.0,
            "" if result else "numeric mismatch",
        )


class NegationFlipRule(Rule):
    """Detect negation asymmetry between premise and hypothesis."""

    name = "negation_flip"

    def check(self, premise: str, hypothesis: str) -> RuleResult:
        """Penalise premise/hypothesis negation asymmetry."""
        if _negation_flip(premise, hypothesis):
            return RuleResult(self.name, 0.3, "negation mismatch")
        return RuleResult(self.name, 1.0)


class LengthRatioRule(Rule):
    """Hypothesis should not be disproportionately longer than premise."""

    name = "length_ratio"

    def check(self, premise: str, hypothesis: str) -> RuleResult:
        """Penalise hypotheses that are much longer than their premise."""
        p_len = max(len(premise.split()), 1)
        h_len = max(len(hypothesis.split()), 1)
        ratio = h_len / p_len
        if ratio <= 3.0:
            return RuleResult(self.name, 1.0)
        if ratio <= 6.0:
            return RuleResult(self.name, 0.6, f"response {ratio:.1f}× longer")
        return RuleResult(self.name, 0.2, f"response {ratio:.1f}× longer")


class WordOverlapRule(Rule):
    """Content word overlap between premise and hypothesis."""

    name = "word_overlap"

    def check(self, premise: str, hypothesis: str) -> RuleResult:
        """Score the Jaccard word overlap of premise and hypothesis."""
        return RuleResult(self.name, _word_overlap(premise, hypothesis))


class ContradictionKeywordRule(Rule):
    """Detect hedging/contradiction markers after assertions."""

    name = "contradiction_keywords"
    _MARKERS = re.compile(
        r"\b(?:however|but actually|in fact|contrary to|"
        r"on the other hand|that said|nevertheless|"
        r"this is (?:not|in)?correct|wrong|false)\b",
        re.IGNORECASE,
    )

    def check(self, premise: str, hypothesis: str) -> RuleResult:
        """Penalise contradiction markers found in hypothesis text."""
        matches = self._MARKERS.findall(hypothesis)
        if not matches:
            return RuleResult(self.name, 1.0)
        penalty = min(len(matches) * 0.25, 0.7)
        return RuleResult(self.name, 1.0 - penalty, f"contradiction markers: {matches}")


class SourceAttributionRule(Rule):
    """If premise has references, hypothesis should cite them."""

    name = "source_attribution"
    _REF_RE = re.compile(
        r"\[\d+\]|\((?:19|20)\d{2}\)|(?:doi|https?)://\S+", re.IGNORECASE
    )

    def check(self, premise: str, hypothesis: str) -> RuleResult:
        """Require response references when the premise contains references."""
        premise_refs = self._REF_RE.findall(premise)
        if not premise_refs:
            return RuleResult(self.name, 1.0, "no refs in premise")
        hyp_refs = self._REF_RE.findall(hypothesis)
        if hyp_refs:
            return RuleResult(self.name, 1.0, "refs present")
        return RuleResult(self.name, 0.5, "premise has refs but response does not cite")


# ── Default rule sets ───────────────────────────────────────────────────


class ContentWordDivergenceRule(Rule):
    """Penalise content words in hypothesis absent from premise.

    Unlike WordOverlapRule (which measures bidirectional F1), this
    rule focuses on **novel content words** — words the response
    introduces that have no grounding in the source. High novelty
    is a hallucination signal.
    """

    name = "content_word_divergence"

    def check(self, premise: str, hypothesis: str) -> RuleResult:
        """Score the share of hypothesis content words grounded in premise."""
        p_words = set(re.findall(r"\w+", premise.lower())) - STOP_WORDS
        h_words = set(re.findall(r"\w+", hypothesis.lower())) - STOP_WORDS
        if not h_words:
            return RuleResult(self.name, 1.0)
        novel = h_words - p_words
        ratio = 1.0 - (len(novel) / len(h_words))
        reason = f"novel words: {sorted(novel)[:5]}" if novel else ""
        return RuleResult(self.name, max(0.0, ratio), reason)


# Weight tuning: entity/numeric/negation/content-divergence rules
# get higher weight because they catch specific hallucination patterns.
# Word overlap and length ratio are softer signals.
_overlap = WordOverlapRule()
_overlap.weight = 0.8

_entity = EntityGroundingRule()
_entity.weight = 1.5

_numeric = NumericConsistencyRule()
_numeric.weight = 2.0

_negation = NegationFlipRule()
_negation.weight = 1.5

_length = LengthRatioRule()
_length.weight = 0.5

_contradiction = ContradictionKeywordRule()
_contradiction.weight = 1.2

_attribution = SourceAttributionRule()
_attribution.weight = 0.5

_content_div = ContentWordDivergenceRule()
_content_div.weight = 1.5

DEFAULT_RULES: list[Rule] = [
    _overlap,
    _entity,
    _numeric,
    _negation,
    _length,
    _contradiction,
    _attribution,
    _content_div,
]


# ── Backend ─────────────────────────────────────────────────────────────


@dataclass
class RulesConfig:
    """Configuration for the rules backend."""

    rules: list[Rule] = field(default_factory=lambda: list(DEFAULT_RULES))
    weights: dict[str, float] = field(default_factory=dict)


def load_rules_from_file(path: str | Path) -> list[Rule]:
    """Load rule configuration from a JSON file.

    The file maps rule names to ``{"enabled": bool, "weight": float}``.
    Only built-in rules can be toggled; custom rules require code.
    """
    data = json.loads(Path(path).read_text())
    name_to_cls: dict[str, type[Rule]] = {
        "entity_grounding": EntityGroundingRule,
        "numeric_consistency": NumericConsistencyRule,
        "negation_flip": NegationFlipRule,
        "length_ratio": LengthRatioRule,
        "word_overlap": WordOverlapRule,
        "contradiction_keywords": ContradictionKeywordRule,
        "source_attribution": SourceAttributionRule,
        "content_word_divergence": ContentWordDivergenceRule,
    }
    rules: list[Rule] = []
    for name, cfg in data.items():
        if not cfg.get("enabled", True):
            continue
        cls = name_to_cls.get(name)
        if cls is None:
            continue
        rule = cls()
        rule.weight = cfg.get("weight", 1.0)
        rules.append(rule)
    return rules


class RulesBackend:
    """Rule-based scorer: zero ML deps, <1ms, ships in base install.

    Implements the same interface as ``ScorerBackend`` so it plugs into
    the backend registry. The ``score()`` method returns a value in
    [0, 1] where 1 = fully grounded and 0 = flagged.
    """

    def __init__(
        self,
        rules: list[Rule] | None = None,
        rules_file: str = "",
    ) -> None:
        """Initialise from explicit rules, a JSON file, or defaults."""
        if rules_file:
            self._rules = load_rules_from_file(rules_file)
        elif rules is not None:
            self._rules = rules
        else:
            self._rules = list(DEFAULT_RULES)

    def score(self, premise: str, hypothesis: str) -> float:
        """Aggregate weighted rule scores into a single [0, 1] value."""
        if not self._rules:
            return 0.5
        total_weight = 0.0
        weighted_sum = 0.0
        for rule in self._rules:
            result = rule.check(premise, hypothesis)
            weighted_sum += result.score * rule.weight
            total_weight += rule.weight
        if total_weight == 0:
            return 0.5
        return weighted_sum / total_weight

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score multiple pairs."""
        return [self.score(p, h) for p, h in pairs]

    def score_detailed(
        self, premise: str, hypothesis: str
    ) -> tuple[float, list[RuleResult]]:
        """Return aggregate score + per-rule breakdown."""
        results: list[RuleResult] = []
        total_weight = 0.0
        weighted_sum = 0.0
        for rule in self._rules:
            result = rule.check(premise, hypothesis)
            results.append(result)
            weighted_sum += result.score * rule.weight
            total_weight += rule.weight
        agg = weighted_sum / total_weight if total_weight > 0 else 0.5
        return agg, results
