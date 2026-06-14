# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — passage-level contradiction benchmark tests

from __future__ import annotations

from benchmarks._common import select_relevant_passages, split_passages
from benchmarks.contradiction_aggrefact import _score_rows


def test_split_passages_basic():
    assert split_passages("The deal closed. It failed later. End.") == [
        "The deal closed.",
        "It failed later.",
    ]  # "End." has < 3 words and is dropped


def test_split_passages_protects_abbreviations_and_decimals():
    text = "Dr. Smith joined the U.S. team for 3.5 million dollars in total."
    # No spurious split after Dr., U.S., or 3.5
    assert split_passages(text) == [text]


def test_split_passages_drops_short_fragments():
    # "Yes." and "No." are below the 3-word floor; only the long one survives.
    assert split_passages("Yes. No. The contract was duly signed today.") == [
        "The contract was duly signed today.",
    ]


def test_split_passages_min_words_override():
    assert split_passages("Go now. Stay here.", min_words=2) == [
        "Go now.",
        "Stay here.",
    ]


def test_select_relevant_passages_returns_all_when_few():
    doc = "The sky is blue today. The grass is green now."
    out = select_relevant_passages(doc, "what colour is the sky", top_k=5)
    assert len(out) == 2


def test_select_relevant_passages_ranks_by_overlap():
    doc = (
        "Quarterly revenue rose sharply this year. "
        "The office cafeteria serves lunch daily. "
        "Marketing hired three new staff members."
    )
    out = select_relevant_passages(doc, "revenue rose this year", top_k=1)
    assert out == ["Quarterly revenue rose sharply this year."]


class _KeywordScorer:
    """Stub contradiction scorer: P(contra) = max keyword score per premise."""

    def __init__(self, contra_by_premise: dict[str, float]) -> None:
        self._by_premise = contra_by_premise

    def contradiction_batch(self, pairs):
        return [self._by_premise.get(premise, 0.0) for premise, _ in pairs]


def test_score_rows_document_uses_whole_doc():
    scorer = _KeywordScorer({"full document text here": 0.7})
    rows = [{"doc": "full document text here", "claim": "c", "label": "0"}]
    scores = _score_rows(scorer, rows, granularity="document", top_k=5, batch_size=8)
    assert scores == [0.7]


def test_score_rows_passage_takes_max_over_passages():
    # Two passages; the stronger contradiction must drive the row score.
    doc = "Profit increased in spring. Profit collapsed in autumn."
    scorer = _KeywordScorer(
        {
            "Profit increased in spring.": 0.1,
            "Profit collapsed in autumn.": 0.85,
        }
    )
    rows = [{"doc": doc, "claim": "profit grew all year", "label": "0"}]
    scores = _score_rows(scorer, rows, granularity="passage", top_k=5, batch_size=8)
    assert scores == [0.85]


def test_score_rows_passage_ungrounded_scores_zero():
    # No usable passage (all fragments below the word floor) -> 0.0, scorer
    # is never consulted for that row.
    scorer = _KeywordScorer({"x": 0.99})
    rows = [{"doc": "Hi. Ok.", "claim": "anything", "label": "0"}]
    scores = _score_rows(scorer, rows, granularity="passage", top_k=5, batch_size=8)
    assert scores == [0.0]
