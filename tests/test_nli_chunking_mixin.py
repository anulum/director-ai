# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ChunkingMixin composition and floor contracts

"""Contract tests for the chunked-scoring mixin behind NLIScorer.

The long-input machinery lives in
``director_ai.core.scoring._nli_chunking`` and composes into
:class:`NLIScorer` as a mixin. These tests pin the composition and the
pure-Python floor semantics of the sentence splitter and chunk builder;
the full chunked-scoring behaviour matrix stays in
``tests/test_chunked_nli.py``.
"""

from __future__ import annotations

import pytest

import director_ai.core.scoring._nli_accel as nli_accel
from director_ai.core.scoring._nli_chunking import ChunkingMixin
from director_ai.core.scoring.nli import NLIScorer


class TestChunkingComposition:
    def test_nli_scorer_composes_the_chunking_mixin(self):
        assert issubclass(NLIScorer, ChunkingMixin)
        for name in (
            "_split_sentences",
            "_estimate_tokens",
            "_build_chunks",
            "_build_chunks_overlap",
            "_score_chunked_with_counts",
            "score_chunked",
            "score_chunked_confidence_weighted",
        ):
            assert getattr(NLIScorer, name) is getattr(ChunkingMixin, name)

    def test_estimate_tokens_uses_the_four_chars_per_token_rule(self):
        assert ChunkingMixin._estimate_tokens("") == 1
        assert ChunkingMixin._estimate_tokens("abcd" * 10) == 11


class TestPythonFloorContracts:
    @pytest.fixture(autouse=True)
    def _force_python_floor(self, monkeypatch):
        monkeypatch.setattr(nli_accel, "_RUST_NLI", False)

    def test_split_sentences_protects_abbreviations_and_decimals(self):
        sentences = NLIScorer._split_sentences(
            "Dr. Smith measured 2.5 mm. The probe worked."
        )
        assert sentences == ["Dr. Smith measured 2.5 mm.", "The probe worked."]

    def test_build_chunks_respects_the_token_budget(self):
        scorer = NLIScorer(use_model=False)
        sentences = ["alpha beta gamma delta." for _ in range(6)]
        chunks = scorer._build_chunks(sentences, budget=12)
        assert len(chunks) > 1
        assert all(chunk for chunk in chunks)

    def test_build_chunks_overlap_strides_forward(self):
        scorer = NLIScorer(use_model=False)
        sentences = [f"sentence number {i} runs here." for i in range(8)]
        chunks = scorer._build_chunks_overlap(sentences, budget=16, overlap_ratio=0.5)
        assert len(chunks) >= 2
        assert chunks[0] != chunks[-1]
