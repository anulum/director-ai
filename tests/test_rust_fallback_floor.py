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

import importlib.util

import numpy as np
import pytest

_KERNEL_ABSENT = importlib.util.find_spec("backfire_kernel") is None


def test_knowledge_word_overlap_python_floor() -> None:
    from director_ai.core.retrieval import knowledge

    # _word_overlap delegates to the shared text_overlap helper, which runs the
    # pure-Python Jaccard for inputs below the large-input threshold — no kernel
    # flag to force. Jaccard of {"a","b","c"} vs {"b","c","d"} = 2/4 = 0.5.
    assert knowledge._word_overlap("a b c", "b c d") == 0.5
    assert knowledge._word_overlap("", "x") == 0.0


def test_nli_softmax_python_floor_large_input(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_nli_divergence_python_floor_large_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from director_ai.core.scoring import nli

    monkeypatch.setattr(nli, "_RUST_NLI", False)
    # 12 rows >= 10 → the Rust-preferring branch; 2-class divergence = 1 - P(sup).
    probs = np.tile(np.array([0.3, 0.7]), (12, 1))
    out = nli._probs_to_divergence(probs)
    assert len(out) == 12
    np.testing.assert_allclose(out, [0.3] * 12, rtol=1e-9)


def test_nli_confidence_python_floor_large_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from director_ai.core.scoring import nli

    monkeypatch.setattr(nli, "_RUST_NLI", False)
    probs = np.tile(np.array([0.25, 0.75]), (15, 1))  # 15 rows >= 10
    out = nli._probs_to_confidence(probs)
    assert len(out) == 15
    assert all(0.0 <= v <= 1.0 for v in out)


def test_doc_chunker_sum_python_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    from director_ai.core.retrieval import doc_chunker

    monkeypatch.setattr(doc_chunker, "_RUST_DOC_CHUNKER", False)
    assert doc_chunker._sum_float_list([1.5, 2.5, 3.0]) == 7.0
    assert doc_chunker._sum_float_list([]) == 0.0


def test_rust_flags_are_boolean() -> None:
    # The flags must always be plain booleans so the no-Rust floor is reachable.
    from director_ai.core.retrieval import adaptive_router, doc_chunker
    from director_ai.core.scoring import nli, rules_scorer
    from director_ai.core.verification import numeric_verifier

    assert isinstance(nli._RUST_NLI, bool)
    assert isinstance(doc_chunker._RUST_DOC_CHUNKER, bool)
    assert isinstance(numeric_verifier._RUST_NUMERIC, bool)
    assert isinstance(rules_scorer._RUST_AVAILABLE, bool)
    assert isinstance(adaptive_router._RUST_AVAILABLE, bool)


@pytest.mark.skipif(
    not _KERNEL_ABSENT,
    reason="genuine floor path — only meaningful in a base install without the kernel",
)
def test_core_review_runs_on_the_pure_python_floor() -> None:
    """A base ``pip install director-ai`` (no ``[rust]``) must still review.

    This is the end-to-end floor guarantee: with the compiled kernel genuinely
    absent, ``CoherenceScorer.review()`` runs the pure-Python task classifier and
    scoring path instead of raising ``RuntimeError``. It runs only in the ``floor``
    CI job (kernel uninstalled); normal kernel-present jobs skip it.
    """
    from director_ai.core import CoherenceScorer

    approved, score = CoherenceScorer(use_nli=False).review(
        "What is 2+2? Explain.", "The answer is 4 because 2+2=4."
    )
    assert isinstance(approved, bool)
    assert 0.0 <= score.score <= 1.0


@pytest.mark.skipif(
    not _KERNEL_ABSENT,
    reason="genuine floor path — only meaningful in a base install without the kernel",
)
def test_numeric_verifier_runs_on_the_pure_python_floor() -> None:
    """``verify_numeric`` must run kernel-absent via its bit-exact Python floor.

    Exercises the real ``except ImportError`` branch (``_RUST_NUMERIC`` False) in a
    base install: the arithmetic/date/probability checks run in pure Python and
    still flag the wrong percentage. Bit-exactness against the kernel is proven
    separately in ``test_numeric_verifier_parity.py`` (kernel-present).
    """
    from director_ai.core.verification.numeric_verifier import verify_numeric

    result = verify_numeric("Revenue grew 15% from $10 million to $12 million.")
    assert result.valid is False
    assert any(issue.issue_type == "arithmetic" for issue in result.issues)


@pytest.mark.skipif(
    not _KERNEL_ABSENT,
    reason="genuine floor path — only meaningful in a base install without the kernel",
)
def test_rules_scorer_runs_on_the_pure_python_floor() -> None:
    """The rule scorer must score kernel-absent via its bit-exact lexical ports.

    Exercises the real ``except ImportError`` branch (``_RUST_AVAILABLE`` False) —
    the entity/numeric/negation/word-overlap signals run in pure Python. This is
    the "Guardrails-competitive tier that ships in the base install" the module
    docstring promises. Bit-exactness against the kernel is proven separately in
    ``test_lexical_signals_parity.py`` (kernel-present).
    """
    from director_ai.core.scoring.rules_scorer import RulesBackend

    backend = RulesBackend()
    grounded = backend.score(
        "Water boils at 100 degrees.", "Water boils at 100 degrees."
    )
    flagged = backend.score(
        "Water boils at 100 degrees.", "Water boils at 500 degrees."
    )
    assert 0.0 <= flagged <= grounded <= 1.0
    assert grounded > flagged


@pytest.mark.skipif(
    not _KERNEL_ABSENT,
    reason="genuine floor path — only meaningful in a base install without the kernel",
)
def test_adaptive_router_runs_on_the_pure_python_floor() -> None:
    """The adaptive router must classify kernel-absent via ``detect_task_type``.

    With the kernel genuinely absent (``_RUST_AVAILABLE`` False) the router falls
    back to the pure-Python task classifier — proven bit-exact with
    ``rust_detect_task_type`` in ``test_task_scoring_paths`` — instead of raising.
    """
    from director_ai.core.retrieval.adaptive_router import AdaptiveRouter

    factual = AdaptiveRouter().should_retrieve("What is the refund policy?")
    creative = AdaptiveRouter().should_retrieve("Write me a poem about spring")
    assert factual.retrieve is True
    assert creative.retrieve is False
