# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Vector Backend Result-Contract Tests
"""Real-surface tests that a loosely-typed backend result never crashes retrieval.

The ``VectorBackend.query`` contract is ``list[dict[str, Any]]`` with no enforced
keys, so a third-party backend (e.g. ColBERT) may omit ``id`` or ``text``. These
tests prove the store skips such results safely (BUG-1) instead of raising
``KeyError`` mid-review, using a concrete backend subclass — no mocking.
"""

from __future__ import annotations

from typing import Any

import pytest

from director_ai.core.retrieval.vector_store.store import (
    _result_evidence_text,
    _result_source,
)
from director_ai.core.vector_store import VectorBackend, VectorGroundTruthStore


class _StaticBackend(VectorBackend):
    """Backend that returns a fixed, possibly malformed, result list verbatim."""

    def __init__(self, results: list[dict[str, Any]]) -> None:
        self._results = results

    def add(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._results.append({"id": doc_id, "text": text, "metadata": metadata or {}})

    def query(
        self,
        text: str,
        n_results: int = 3,
        tenant_id: str = "",
    ) -> list[dict[str, Any]]:
        return list(self._results[:n_results])

    def count(self) -> int:
        return len(self._results)


@pytest.mark.consumer
class TestResultEvidenceTextHelper:
    def test_returns_text_when_present(self):
        assert _result_evidence_text({"text": "Paris is the capital"}) == (
            "Paris is the capital"
        )

    @pytest.mark.parametrize(
        "result",
        [
            {},  # no text key at all
            {"text": ""},  # empty string carries no evidence
            {"text": None},  # non-string
            {"text": 42},  # non-string
            {"id": "doc1"},  # id but no text
        ],
    )
    def test_returns_none_for_unusable_text(self, result):
        assert _result_evidence_text(result) is None

    def test_source_tolerates_missing_id(self):
        assert _result_source({"text": "x"}) == "vector:"

    def test_source_uses_id_when_present(self):
        assert _result_source({"id": "doc7", "text": "x"}) == "vector:doc7"


@pytest.mark.consumer
class TestChunksPathNeverKeyErrors:
    def test_result_missing_id_still_yields_chunk(self):
        store = VectorGroundTruthStore(
            backend=_StaticBackend([{"text": "The sky is blue", "distance": 0.1}]),
        )
        chunks = store.retrieve_context_with_chunks("colour of sky", top_k=1)
        assert len(chunks) == 1
        assert chunks[0].text == "The sky is blue"
        assert chunks[0].source == "vector:"

    def test_result_missing_text_is_skipped_not_raised(self):
        store = VectorGroundTruthStore(
            backend=_StaticBackend(
                [
                    {"id": "bad", "distance": 0.2},  # no text → skip
                    {"id": "good", "text": "Water is wet", "distance": 0.1},
                ],
            ),
        )
        chunks = store.retrieve_context_with_chunks("state of water", top_k=2)
        assert [c.text for c in chunks] == ["Water is wet"]
        assert chunks[0].source == "vector:good"

    def test_all_malformed_results_fall_back_to_parent(self):
        # No usable text anywhere → chunks empty → parent keyword store answers.
        store = VectorGroundTruthStore(
            backend=_StaticBackend([{"id": "a"}, {"id": "b"}]),
        )
        chunks = store.retrieve_context_with_chunks("anything", top_k=2)
        assert chunks == []  # fresh parent store has no facts either


@pytest.mark.consumer
class TestStringPathNeverKeyErrors:
    def test_string_context_skips_textless_results(self):
        store = VectorGroundTruthStore(
            backend=_StaticBackend(
                [
                    {"id": "x"},  # no text → skip
                    {"id": "y", "text": "Fire is hot"},
                    {"id": "z", "text": "Ice is cold"},
                ],
            ),
        )
        context = store.retrieve_context("temperature facts", top_k=3)
        assert "Fire is hot" in context
        assert "Ice is cold" in context

    def test_string_context_all_textless_falls_back(self):
        store = VectorGroundTruthStore(
            backend=_StaticBackend([{"id": "x"}, {"id": "y"}]),
        )
        # Parent contract is ``str | None``; an empty fact store returns None.
        assert store.retrieve_context("anything", top_k=2) is None
