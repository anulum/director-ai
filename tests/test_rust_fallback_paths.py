# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Mandatory Rust Path Tests
"""Tests proving mandatory Rust accelerator failures propagate."""

from __future__ import annotations

import numpy as np
import pytest


def _raise_kernel_unavailable(*_args, **_kwargs):
    raise RuntimeError("kernel unavailable")


def test_lite_scorer_single_rust_failure_raises(monkeypatch):
    """Lite single-pair scoring must fail when the mandatory kernel fails."""
    import director_ai.core.scoring.lite_scorer as mod

    monkeypatch.setattr(mod, "_RUST_LITE", True)
    monkeypatch.setattr(mod, "rust_lite_score", _raise_kernel_unavailable)

    with pytest.raises(RuntimeError, match="kernel unavailable"):
        mod.LiteScorer().score("The sky is blue.", "The sky is green.")


def test_lite_scorer_batch_rust_failure_raises(monkeypatch):
    """Lite batch scoring must fail when the mandatory kernel fails."""
    import director_ai.core.scoring.lite_scorer as mod

    monkeypatch.setattr(mod, "_RUST_LITE", True)
    monkeypatch.setattr(mod, "rust_lite_score_batch", _raise_kernel_unavailable)

    with pytest.raises(RuntimeError, match="kernel unavailable"):
        mod.LiteScorer().score_batch([("The sky is blue.", "The sky is green.")])


def test_nli_softmax_rust_failure_raises(monkeypatch):
    """Large NLI softmax batches must fail when the mandatory kernel fails."""
    import director_ai.core.scoring.nli as mod

    monkeypatch.setattr(mod, "_RUST_NLI", True)
    monkeypatch.setattr(mod, "rust_softmax", _raise_kernel_unavailable)

    logits = np.ones((50, 3), dtype=np.float64)
    with pytest.raises(RuntimeError, match="kernel unavailable"):
        mod._softmax_np(logits)


def test_nli_divergence_rust_failure_raises(monkeypatch):
    """Large NLI divergence batches must fail when the mandatory kernel fails."""
    import director_ai.core.scoring.nli as mod

    monkeypatch.setattr(mod, "_RUST_NLI", True)
    monkeypatch.setattr(mod, "rust_probs_to_divergence", _raise_kernel_unavailable)

    probs = np.full((15, 3), 1 / 3, dtype=np.float64)
    with pytest.raises(RuntimeError, match="kernel unavailable"):
        mod._probs_to_divergence(probs)


def test_nli_confidence_rust_failure_raises(monkeypatch):
    """Large NLI confidence batches must fail when the mandatory kernel fails."""
    import director_ai.core.scoring.nli as mod

    monkeypatch.setattr(mod, "_RUST_NLI", True)
    monkeypatch.setattr(mod, "rust_probs_to_confidence", _raise_kernel_unavailable)

    probs = np.full((15, 3), 1 / 3, dtype=np.float64)
    with pytest.raises(RuntimeError, match="kernel unavailable"):
        mod._probs_to_confidence(probs)


def test_injection_sentence_split_rust_failure_raises(monkeypatch):
    """Injection sentence splitting must fail when the mandatory kernel fails."""
    import director_ai.core.safety.injection as mod

    monkeypatch.setattr(mod, "_RUST_INJECTION", True)
    monkeypatch.setattr(mod, "rust_split_sentences", _raise_kernel_unavailable)

    with pytest.raises(RuntimeError, match="kernel unavailable"):
        mod._fallback_split("One. Two.")


def test_sanitizer_sum_rust_failure_raises(monkeypatch):
    """Sanitizer scoring helpers must fail when the mandatory kernel fails."""
    import director_ai.core.safety.sanitizer as mod

    monkeypatch.setattr(mod, "_RUST_SANITIZER", True)
    monkeypatch.setattr(mod, "rust_sum_i64", _raise_kernel_unavailable)

    with pytest.raises(RuntimeError, match="kernel unavailable"):
        mod._sum_int([1, 2, 3])


def test_agentic_loop_monitor_jaccard_drift_values():
    """Goal-drift now delegates to the shared text_overlap helper, whose Rust
    path mandatory-failure is enforced in test_text_overlap. Here we pin the
    drift arithmetic: 1.0 - Jaccard similarity."""
    import director_ai.agentic.loop_monitor as mod

    # No shared words -> 0.0 similarity -> maximal 1.0 drift.
    assert mod.LoopMonitor._jaccard_drift("write report", "open calculator") == 1.0
    # Identical tokens -> full similarity -> 0.0 drift.
    assert mod.LoopMonitor._jaccard_drift("write report", "write report") == 0.0
