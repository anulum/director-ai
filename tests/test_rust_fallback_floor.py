# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Rust-accelerator Python-floor tests
"""The pure-Python floor must run when the Rust accelerator is unavailable.

These functions previously routed through ``mandatory_execution`` whenever the
``_RUST_*`` flag was set, and the flag was set ``True`` even in the
``except ImportError`` branch — so without ``backfire_kernel`` they raised
``RuntimeError`` instead of using the Python implementation sitting right beside
them. The flag now defaults ``False`` when the import fails; these tests force
the no-Rust path and assert the Python floor produces correct results
(especially the large-input branches that were the actual break).
"""

from __future__ import annotations

import numpy as np


def test_knowledge_word_overlap_python_floor(monkeypatch) -> None:
    from director_ai.core.retrieval import knowledge

    monkeypatch.setattr(knowledge, "_RUST_KNOWLEDGE", False)
    # Jaccard of {"a","b","c"} vs {"b","c","d"} = 2/4 = 0.5.
    assert knowledge._word_overlap("a b c", "b c d") == 0.5
    assert knowledge._word_overlap("", "x") == 0.0


def test_nli_softmax_python_floor_large_input(monkeypatch) -> None:
    from director_ai.core.scoring import nli

    monkeypatch.setattr(nli, "_RUST_NLI", False)
    # 60x2 = 120 elements → size >= 100, the branch that previously demanded Rust.
    x = np.linspace(-3.0, 3.0, 120, dtype=np.float64).reshape(60, 2)
    out = nli._softmax_np(x)
    assert out.shape == (60, 2)
    # Each row is a valid probability distribution.
    np.testing.assert_allclose(out.sum(axis=1), np.ones(60), rtol=1e-9)
    # Matches a reference softmax.
    ref = np.exp(x - x.max(axis=1, keepdims=True))
    ref = ref / ref.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(out, ref, rtol=1e-9)


def test_nli_divergence_python_floor_large_batch(monkeypatch) -> None:
    from director_ai.core.scoring import nli

    monkeypatch.setattr(nli, "_RUST_NLI", False)
    # 12 rows >= 10 → the Rust-preferring branch; 2-class divergence = 1 - P(sup).
    probs = np.tile(np.array([0.3, 0.7]), (12, 1))
    out = nli._probs_to_divergence(probs)
    assert len(out) == 12
    np.testing.assert_allclose(out, [0.3] * 12, rtol=1e-9)


def test_nli_confidence_python_floor_large_batch(monkeypatch) -> None:
    from director_ai.core.scoring import nli

    monkeypatch.setattr(nli, "_RUST_NLI", False)
    probs = np.tile(np.array([0.25, 0.75]), (15, 1))  # 15 rows >= 10
    out = nli._probs_to_confidence(probs)
    assert len(out) == 15
    assert all(0.0 <= v <= 1.0 for v in out)


def test_doc_chunker_sum_python_floor(monkeypatch) -> None:
    from director_ai.core.retrieval import doc_chunker

    monkeypatch.setattr(doc_chunker, "_RUST_DOC_CHUNKER", False)
    assert doc_chunker._sum_float_list([1.5, 2.5, 3.0]) == 7.0
    assert doc_chunker._sum_float_list([]) == 0.0


def test_rust_flags_are_boolean() -> None:
    # The flags must always be plain booleans so the no-Rust floor is reachable.
    from director_ai.core.retrieval import doc_chunker, knowledge
    from director_ai.core.scoring import nli

    assert isinstance(knowledge._RUST_KNOWLEDGE, bool)
    assert isinstance(nli._RUST_NLI, bool)
    assert isinstance(doc_chunker._RUST_DOC_CHUNKER, bool)
