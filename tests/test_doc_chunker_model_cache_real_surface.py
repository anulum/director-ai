# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — doc chunker model cache real-surface tests
"""Real semantic chunking coverage for the sentence-transformer model cache."""

from __future__ import annotations

import sys
import types

import numpy as np
import numpy.typing as npt
import pytest

from director_ai.core.retrieval import doc_chunker
from director_ai.core.retrieval.doc_chunker import ChunkConfig, split
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


def test_doc_chunker_model_cache_unit_guard_has_real_surface_companion() -> None:
    """Doc chunker cache guard should be backed by public semantic splitting."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_doc_chunker_model_cache.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_doc_chunker_model_cache_real_surface.py" in category


def test_semantic_split_reuses_cached_sentence_transformer_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated public semantic splits should reuse one loaded model instance."""
    initialisations: list[tuple[str, str | None]] = []
    encoded_batches: list[tuple[list[str], bool]] = []

    class FakeSentenceTransformer:
        """Sentence-transformers-compatible fake with observable construction."""

        def __init__(self, model_name: str, device: str | None = None) -> None:
            initialisations.append((model_name, device))

        def encode(
            self,
            sentences: list[str],
            *,
            show_progress_bar: bool,
        ) -> npt.NDArray[np.float64]:
            encoded_batches.append((list(sentences), show_progress_bar))
            return np.eye(len(sentences), dtype=np.float64)

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.__dict__["SentenceTransformer"] = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    doc_chunker._sentence_transformer_model.cache_clear()

    text = (
        "Alpha approval evidence must remain attached. "
        "Beta retrieval controls must preserve tenant scope. "
        "Gamma rollback records must stay auditable."
    )
    config = ChunkConfig(
        chunk_size=64,
        overlap=0,
        semantic=True,
        similarity_threshold=0.5,
    )
    try:
        first_chunks = split(text, config)
        second_chunks = split(text, config)
    finally:
        doc_chunker._sentence_transformer_model.cache_clear()

    assert initialisations == [("all-MiniLM-L6-v2", "cpu")]
    assert len(encoded_batches) == 2
    assert all(show_progress is False for _sentences, show_progress in encoded_batches)
    assert first_chunks == second_chunks
    assert len(first_chunks) >= 2
    assert any("tenant scope" in chunk for chunk in first_chunks)
