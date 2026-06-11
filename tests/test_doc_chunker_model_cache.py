# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — doc chunker model cache regression tests

from __future__ import annotations

import sys
import types

from director_ai.core.retrieval import doc_chunker


def test_sentence_transformer_model_is_cached(monkeypatch) -> None:
    init_calls = 0

    class FakeModel:
        def __init__(self, _name: str) -> None:
            nonlocal init_calls
            init_calls += 1

        def encode(self, _sentences, show_progress_bar=False):  # noqa: ARG002
            return [[1.0, 0.0]]

    fake_module = types.SimpleNamespace(SentenceTransformer=FakeModel)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    doc_chunker._sentence_transformer_model.cache_clear()

    model_1 = doc_chunker._sentence_transformer_model()
    model_2 = doc_chunker._sentence_transformer_model()

    assert model_1 is model_2
    assert init_calls == 1
