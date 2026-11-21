# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — AtlasKV comparison corpus tests

"""The comparison corpus is deterministic and has its preregistered shape."""

from __future__ import annotations

from collections import Counter

import pytest

from benchmarks.atlaskv_corpus import (
    QUESTION_TEMPLATES,
    RELATIONS,
    TENANTS,
    Corpus,
    build_corpus,
)


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    """Build the full preregistered corpus once."""
    return build_corpus()


def test_the_corpus_is_deterministic(corpus: Corpus) -> None:
    """The same seed gives the same snapshots and queries."""
    again = build_corpus()
    assert again.snapshot.snapshot_id == corpus.snapshot.snapshot_id
    assert again.swapped.snapshot_id == corpus.swapped.snapshot_id
    assert again.queries == corpus.queries
    assert build_corpus(seed=7).snapshot.snapshot_id != corpus.snapshot.snapshot_id


def test_the_corpus_has_its_preregistered_proportions(corpus: Corpus) -> None:
    """Entities, contradictions, retractions and tenants are near their shares."""
    facts = corpus.snapshot.facts
    roles = Counter(fact.role for fact in facts)
    primaries = roles["primary"]
    assert len({fact.entity_id for fact in facts}) == 400
    assert 0.07 < roles["contradiction"] / primaries < 0.13
    assert 0.03 < roles["retracted"] / primaries < 0.07
    assert {fact.tenant_id for fact in facts} == set(TENANTS)
    assert len(corpus.snapshot.evidence) == 3 * len(facts)
    assert sum(1 for q in corpus.queries if q.has_similar_name) > 0


def test_every_fact_has_three_paraphrases_with_its_value(corpus: Corpus) -> None:
    """Each paraphrase names the entity and states the fact's value."""
    names = {fact.fact_id: fact.entity_name for fact in corpus.snapshot.facts}
    for item in corpus.snapshot.evidence[:300]:
        assert item.value in item.text
        assert names[item.fact_id] in item.text


def test_splits_keep_entity_groups_whole(corpus: Corpus) -> None:
    """Similar-name partners share a split, and queries follow their entity."""
    split = corpus.entity_split
    for fact in corpus.snapshot.facts:
        if fact.entity_name.endswith(" North"):
            base = next(
                f.entity_id
                for f in corpus.snapshot.facts
                if f.entity_name == fact.entity_name.removesuffix(" North")
            )
            assert split[base] == split[fact.entity_id]
    assert all(q.split == split[q.entity_id] for q in corpus.queries)
    assert 0.15 < sum(1 for s in split.values() if s == "dev") / len(split) < 0.25


def test_no_question_copies_its_key_or_evidence_phrase() -> None:
    """Questions never contain the relation's key noun or evidence verb verbatim."""
    for relations in RELATIONS.values():
        for relation, noun, verb, _ in relations:
            for template in QUESTION_TEMPLATES[relation]:
                assert noun.lower() not in template.lower()
                assert verb.lower() not in template.lower()


def test_groups_follow_the_split(corpus: Corpus) -> None:
    """Every member of a group has the group's split."""
    for entity, group in corpus.entity_group.items():
        assert corpus.entity_split[entity] == corpus.entity_split[group]


def test_queries_carry_their_gold_and_claims(corpus: Corpus) -> None:
    """Answers have gold support; negatives have none; claims differ."""
    kinds = Counter(q.kind for q in corpus.queries)
    assert set(kinds) == {"answer", "foreign_tenant", "unknown_relation"}
    for query in corpus.queries:
        assert query.true_value != query.false_value
        assert query.text not in {item.text for item in corpus.snapshot.evidence[:50]}
        if query.kind == "answer":
            assert len(query.gold_support) == 3
        else:
            assert query.gold_support == () and query.gold_contradict == ()
    retracted = {
        (f.entity_id, f.relation): f.value for f in corpus.snapshot.facts if f.retracted
    }
    assert any(
        q.false_value == retracted[(q.entity_id, q.relation)]
        for q in corpus.queries
        if (q.entity_id, q.relation) in retracted
    )


def test_false_values_come_from_retrieval_errors_in_order(corpus: Corpus) -> None:
    """Each false claim takes the first error class that applies to its fact."""
    facts = corpus.snapshot.facts
    rival = {
        (f.entity_id, f.relation): f.value for f in facts if f.role == "contradiction"
    }
    stale = {(f.entity_id, f.relation): f.value for f in facts if f.role == "retracted"}
    names = {f.entity_id: f.entity_name for f in facts}
    by_name = {f.entity_name: f.entity_id for f in facts}
    primary = {(f.entity_id, f.relation): f.value for f in facts if f.role == "primary"}
    classes = {q.false_value_class for q in corpus.queries}
    assert {
        "contradiction",
        "retracted",
        "partner",
        "other_entity",
        "negative",
    } <= classes
    for query in corpus.queries:
        key = (query.entity_id, query.relation)
        assert query.entity_name == names[query.entity_id]
        if query.false_value_class == "contradiction":
            assert query.false_value == rival[key]
        elif query.false_value_class == "retracted":
            assert key not in rival and query.false_value == stale[key]
        elif query.false_value_class == "partner":
            name = query.entity_name
            partner = (
                by_name[name.removesuffix(" North")]
                if name.endswith(" North")
                else by_name[f"{name} North"]
            )
            assert query.false_value == primary[(partner, query.relation)]
    assert {q.false_value_class for q in corpus.swap_queries} == {"stale_swap"}


def test_the_swap_replaces_values_and_its_queries_expect_them(corpus: Corpus) -> None:
    """Swap queries expect the new value and treat the old one as false."""
    old = {f.fact_id: f.value for f in corpus.snapshot.facts}
    new = {f.fact_id: f.value for f in corpus.swapped.facts}
    changed = {fid for fid in old if old[fid] != new[fid]}
    assert changed and len(corpus.swap_queries) == len(changed)
    for query in corpus.swap_queries:
        fact_id = query.query_id[2:]
        assert query.true_value == new[fact_id] and query.false_value == old[fact_id]
        assert query.kind == "swap"
