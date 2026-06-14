# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — contradiction dataset builder tests

from __future__ import annotations

from benchmarks.contradiction_injection import ContradictionInjector
from training.build_contradiction_dataset import (
    LABEL_CONTRADICTION,
    LABEL_ENTAILMENT,
    LABEL_NEUTRAL,
    build_examples,
)


def test_supported_row_yields_entailment_plus_contradictions():
    rows = [
        {
            "doc": "The merger was approved by the board in 2014.",
            "claim": "The merger was approved by the board in 2014.",
            "label": "1",
        }
    ]
    ex = build_examples(rows, ContradictionInjector())
    labels = [e["label"] for e in ex]
    assert labels.count(LABEL_ENTAILMENT) == 1
    # negation, antonym (approved->rejected), numeric (2014) -> >= 2 contradictions
    assert labels.count(LABEL_CONTRADICTION) >= 2
    assert all(e["premise"] for e in ex)
    # the contradiction hypotheses differ from the entailment hypothesis
    entail = next(e for e in ex if e["label"] == LABEL_ENTAILMENT)
    contras = [e for e in ex if e["label"] == LABEL_CONTRADICTION]
    assert all(c["hypothesis"] != entail["hypothesis"] for c in contras)
    assert all(c["premise"] == entail["premise"] for c in contras)


def test_unsupported_row_yields_neutral():
    rows = [
        {
            "doc": "Some grounding text here today.",
            "claim": "Unrelated stuff.",
            "label": "0",
        }
    ]
    ex = build_examples(rows, ContradictionInjector())
    assert len(ex) == 1
    assert ex[0]["label"] == LABEL_NEUTRAL
    assert ex[0]["source"] == "aggrefact_unsupported"


def test_neutral_cap_limits_unsupported():
    rows = [
        {"doc": f"Doc number {i} content.", "claim": f"Claim {i}.", "label": "0"}
        for i in range(5)
    ]
    ex = build_examples(rows, ContradictionInjector(), neutral_cap=2)
    assert sum(1 for e in ex if e["label"] == LABEL_NEUTRAL) == 2


def test_injected_source_tags_carry_strategy():
    rows = [
        {
            "doc": "Revenue rose sharply in spring.",
            "claim": "Revenue rose sharply in spring.",
            "label": "1",
        }
    ]
    ex = build_examples(rows, ContradictionInjector())
    contra_sources = {e["source"] for e in ex if e["label"] == LABEL_CONTRADICTION}
    assert any(s.startswith("aggrefact_injected_") for s in contra_sources)


def test_cross_doc_neutral_pairs_claims_with_other_documents():
    rows = [
        {
            "doc": f"Document {i} discusses subject {i} thoroughly this year.",
            "claim": f"Subject {i} grew considerably overall.",
            "label": "1",
        }
        for i in range(6)
    ]
    ex = build_examples(rows, ContradictionInjector(), cross_doc_neutral=3)
    cross = [e for e in ex if e["source"] == "cross_doc_neutral"]
    assert len(cross) == 3
    assert all(e["label"] == LABEL_NEUTRAL for e in cross)
    assert all(e["premise"] and e["hypothesis"] for e in cross)


def test_cross_doc_neutral_zero_when_too_few_supported():
    rows = [
        {"doc": "Only one supported doc here.", "claim": "One claim.", "label": "1"}
    ]
    ex = build_examples(rows, ContradictionInjector(), cross_doc_neutral=5)
    assert not [e for e in ex if e["source"] == "cross_doc_neutral"]
