# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — contradiction injection tests

from __future__ import annotations

from benchmarks.contradiction_injection import ContradictionInjector


def _inj() -> ContradictionInjector:
    return ContradictionInjector()


def test_negation_inserts_not_after_copula():
    r = _inj().inject("The merger was approved by the board.")
    assert r.strategy == "negation"
    assert r.perturbed == "The merger was not approved by the board."
    assert r.changed is True


def test_negation_strips_existing_negation():
    r = _inj().inject("The contract was not signed.")
    assert r.strategy == "negation"
    assert r.perturbed == "The contract was signed."


def test_antonym_flips_lexical_verb_without_auxiliary():
    # "rose" has no copula/auxiliary, so negation declines and antonym applies.
    r = _inj().inject("Quarterly revenue rose sharply.")
    assert r.strategy == "antonym"
    assert r.perturbed == "Quarterly revenue fell sharply."


def test_numeric_perturbs_plain_integer():
    r = _inj().inject("The team scored 20 points.")
    assert r.strategy == "numeric"
    assert r.perturbed == "The team scored 47 points."  # 20*2+7


def test_numeric_shifts_year_by_decade():
    r = _inj().inject("The company launched in 2014.")
    assert r.strategy == "numeric"
    assert r.perturbed == "The company launched in 2004."


def test_numeric_handles_decimal():
    r = _inj().inject("Growth reached 3.5 last quarter.")
    assert r.strategy == "numeric"
    assert r.perturbed == "Growth reached 103.5 last quarter."  # <100 -> +100


def test_quantifier_swaps_universal():
    # No auxiliary, no antonym, no number -> quantifier strategy fires.
    r = _inj().inject("All students passed.")
    assert r.strategy == "quantifier"
    assert r.perturbed == "no students passed."


def test_priority_negation_before_antonym():
    # "approved" is an antonym candidate, but "was" lets negation win first.
    r = _inj().inject("The plan was approved.")
    assert r.strategy == "negation"
    assert r.perturbed == "The plan was not approved."


def test_whole_word_only_no_substring_match():
    # "ball" must not match the quantifier "all"; nothing else applies.
    r = _inj().inject("The ball bounced softly.")
    assert r.changed is False
    assert r.perturbed == "The ball bounced softly."


def test_unchanged_when_no_strategy_applies():
    r = _inj().inject("Birds sang outside quietly.")
    assert r.strategy == ""
    assert r.changed is False
    assert r.perturbed == r.original


def test_custom_strategy_order_restricts_to_numeric():
    inj = ContradictionInjector(strategies=("numeric",))
    # Has a copula and an antonym, but only numeric is enabled and there is no
    # number -> unchanged.
    r = inj.inject("The deal was approved.")
    assert r.changed is False


def test_inject_batch():
    results = _inj().inject_batch(["Revenue rose sharply.", "Birds sang quietly."])
    assert [r.changed for r in results] == [True, False]


def test_inject_all_returns_one_variant_per_applicable_strategy():
    # "All sales rose in 2014" supports negation? no copula/aux -> skip; antonym
    # (rose->fell), numeric (2014->2004), quantifier (all->no).
    variants = _inj().inject_all("All sales rose 2014")
    strategies = {v.strategy for v in variants}
    assert strategies == {"antonym", "numeric", "quantifier"}
    assert all(v.changed for v in variants)


def test_inject_all_empty_when_nothing_applies():
    assert _inj().inject_all("Birds sang quietly") == []
