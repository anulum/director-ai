# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — doc chunker model cache regression tests
"""Unit guard for the document chunker sentence-transformer model cache."""

from __future__ import annotations

import sys
import types

import pytest

from director_ai.core.retrieval import doc_chunker


def test_sentence_transformer_model_is_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The private loader should keep one model instance per process cache."""
    init_calls = 0

    class FakeModel:
        """Minimal sentence-transformer stand-in for the cache unit guard."""

        def __init__(self, _name: str, device: str | None = None) -> None:
            nonlocal init_calls
            assert device == "cpu"
            init_calls += 1

        def encode(
            self,
            _sentences: list[str],
            *,
            show_progress_bar: bool = False,
        ) -> list[list[float]]:
            assert show_progress_bar is False
            return [[1.0, 0.0]]

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.__dict__["SentenceTransformer"] = FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    doc_chunker._sentence_transformer_model.cache_clear()

    try:
        model_1 = doc_chunker._sentence_transformer_model()
        model_2 = doc_chunker._sentence_transformer_model()
    finally:
        doc_chunker._sentence_transformer_model.cache_clear()

    assert model_1 is model_2
    assert init_calls == 1
