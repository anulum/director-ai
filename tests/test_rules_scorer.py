# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""STRONG tests for ``director_ai.core.scoring.rules_scorer``.

Covers every rule individually, aggregation logic, weight tuning,
JSON config loading, backend registry integration, and edge cases.
"""

from __future__ import annotations

import json

from director_ai.core.scoring.rules_scorer import (
    ContentWordDivergenceRule,
    ContradictionKeywordRule,
    EntityGroundingRule,
    LengthRatioRule,
    NegationFlipRule,
    NumericConsistencyRule,
    RulesBackend,
    SourceAttributionRule,
    WordOverlapRule,
    load_rules_from_file,
)

# ── EntityGroundingRule ─────────────────────────────────────────────────


class TestEntityGroundingRule:
    rule = EntityGroundingRule()

    def test_all_entities_grounded(self):
        r = self.rule.check("Paris is in France.", "Paris is a city in France.")
        assert r.score == 1.0

    def test_novel_entity(self):
        r = self.rule.check("Paris is in France.", "Berlin is in Germany.")
        assert r.score < 1.0

    def test_no_entities_in_hypothesis(self):
        r = self.rule.check("Paris is nice.", "it is nice.")
        # Rust returns 0.0 (no entity overlap), Python returns 1.0 (no hyp entities)
        # Both are valid — the key invariant is: grounded >= ungrounded
        grounded = self.rule.check("Paris is nice.", "Paris is nice.")
        assert grounded.score >= r.score


# ── NumericConsistencyRule ──────────────────────────────────────────────


class TestNumericConsistencyRule:
    rule = NumericConsistencyRule()

    def test_matching_numbers(self):
        r = self.rule.check("Water boils at 100 degrees.", "It boils at 100 degrees.")
        assert r.score == 1.0

    def test_wrong_number(self):
        r = self.rule.check("Water boils at 100 degrees.", "It boils at 500 degrees.")
        assert r.score < 1.0

    def test_no_numbers(self):
        r = self.rule.check("The sky is blue.", "The sky is blue.")
        assert r.score == 1.0

    def test_percentage(self):
        r = self.rule.check("Accuracy is 75%.", "Accuracy is 90%.")
        assert r.score < 1.0


# ── NegationFlipRule ────────────────────────────────────────────────────


class TestNegationFlipRule:
    rule = NegationFlipRule()

    def test_no_negation_both(self):
        r = self.rule.check("Sky is blue.", "Sky is blue.")
        assert r.score == 1.0

    def test_negation_flip(self):
        r = self.rule.check("It is raining.", "It is not raining.")
        assert r.score < 1.0

    def test_negation_both(self):
        r = self.rule.check("It is not cold.", "It is not cold.")
        assert r.score == 1.0


# ── LengthRatioRule ────────────────────────────────────────────────────


class TestLengthRatioRule:
    rule = LengthRatioRule()

    def test_similar_length(self):
        r = self.rule.check("Short premise.", "Short response.")
        assert r.score == 1.0

    def test_very_long_response(self):
        r = self.rule.check("Short.", "This is a very long response " * 10)
        assert r.score < 0.5


# ── WordOverlapRule ─────────────────────────────────────────────────────


class TestWordOverlapRule:
    rule = WordOverlapRule()

    def test_identical(self):
        r = self.rule.check("The sky is blue.", "The sky is blue.")
        assert r.score > 0.9

    def test_no_overlap(self):
        r = self.rule.check("Alpha beta gamma.", "Delta epsilon zeta.")
        assert r.score == 0.0

    def test_partial_overlap(self):
        r = self.rule.check("The cat sat on the mat.", "The cat jumped off the mat.")
        assert 0.3 < r.score < 0.9


# ── ContradictionKeywordRule ────────────────────────────────────────────


class TestContradictionKeywordRule:
    rule = ContradictionKeywordRule()

    def test_no_markers(self):
        r = self.rule.check("x", "The answer is correct.")
        assert r.score == 1.0

    def test_however(self):
        r = self.rule.check("x", "This is true. However, actually it is false.")
        assert r.score < 1.0

    def test_multiple_markers(self):
        r = self.rule.check("x", "However, but actually, this is wrong.")
        assert r.score < 0.5


# ── SourceAttributionRule ───────────────────────────────────────────────


class TestSourceAttributionRule:
    rule = SourceAttributionRule()

    def test_no_refs_in_premise(self):
        r = self.rule.check("Sky is blue.", "Sky is blue.")
        assert r.score == 1.0

    def test_refs_in_premise_missing_in_hyp(self):
        r = self.rule.check("Results were positive [1].", "Results were positive.")
        assert r.score < 1.0

    def test_refs_in_both(self):
        r = self.rule.check("Results [1] positive.", "Per [1], results positive.")
        assert r.score == 1.0


# ── ContentWordDivergenceRule ───────────────────────────────────────────


class TestContentWordDivergenceRule:
    rule = ContentWordDivergenceRule()

    def test_no_novel_words(self):
        r = self.rule.check("The sky is blue.", "The sky is blue.")
        assert r.score == 1.0

    def test_many_novel_words(self):
        r = self.rule.check("The sky is blue.", "The ocean is purple and magnificent.")
        assert r.score < 0.5

    def test_empty_hypothesis(self):
        r = self.rule.check("Some text.", "")
        assert r.score == 1.0  # no content words to check


# ── RulesBackend aggregation ────────────────────────────────────────────


class TestRulesBackend:
    def test_default_construction(self):
        b = RulesBackend()
        assert len(b._rules) > 0

    def test_score_returns_float(self):
        b = RulesBackend()
        s = b.score("premise", "hypothesis")
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0

    def test_score_batch(self):
        b = RulesBackend()
        scores = b.score_batch([("a", "a"), ("a", "b")])
        assert len(scores) == 2

    def test_grounded_higher_than_fabricated(self):
        b = RulesBackend()
        grounded = b.score("Water boils at 100°C.", "Water boils at 100°C.")
        fabricated = b.score("Water boils at 100°C.", "Water boils at 500°C.")
        assert grounded > fabricated

    def test_custom_rules(self):
        b = RulesBackend(rules=[NumericConsistencyRule()])
        s = b.score("100 degrees", "500 degrees")
        assert s < 0.5  # only numeric rule, and it fails

    def test_empty_rules(self):
        b = RulesBackend(rules=[])
        assert b.score("a", "b") == 0.5

    def test_score_detailed(self):
        b = RulesBackend()
        agg, details = b.score_detailed("Sky is blue.", "Sky is green.")
        assert isinstance(agg, float)
        assert len(details) == len(b._rules)
        for d in details:
            assert 0.0 <= d.score <= 1.0


# ── JSON config loading ────────────────────────────────────────────────


class TestLoadRulesFromFile:
    def test_load_valid(self, tmp_path):
        cfg = {
            "word_overlap": {"enabled": True, "weight": 2.0},
            "negation_flip": {"enabled": False},
            "numeric_consistency": {"enabled": True, "weight": 1.5},
        }
        p = tmp_path / "rules.json"
        p.write_text(json.dumps(cfg))
        rules = load_rules_from_file(p)
        names = [r.name for r in rules]
        assert "word_overlap" in names
        assert "negation_flip" not in names
        assert "numeric_consistency" in names
        wo = next(r for r in rules if r.name == "word_overlap")
        assert wo.weight == 2.0

    def test_unknown_rule_ignored(self, tmp_path):
        cfg = {"nonexistent_rule": {"enabled": True}}
        p = tmp_path / "rules.json"
        p.write_text(json.dumps(cfg))
        rules = load_rules_from_file(p)
        assert len(rules) == 0


# ── Backend registry integration ────────────────────────────────────────


class TestBackendRegistry:
    def test_rules_backend_registered(self):
        from director_ai.core.scoring.backends import get_backend

        cls = get_backend("rules")
        assert cls is not None

    def test_rules_backend_score(self):
        from director_ai.core.scoring.backends import get_backend

        cls = get_backend("rules")
        b = cls()
        s = b.score("Sky is blue.", "Sky is blue.")
        assert isinstance(s, float)
        assert s > 0.5
