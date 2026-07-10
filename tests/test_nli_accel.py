# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — NLI accelerator binding contracts

"""Contract tests for the canonical Rust NLI accelerator binding.

``director_ai.core.scoring._nli_accel`` is the single module through which
every NLI compute path resolves its Rust fast lane. These tests pin that
contract: the flag and all nine accelerator names live there, and patching
the binding module — one place — switches the consumers in ``nli.py``
between the accelerated branch and the pure-Python floor.
"""

from __future__ import annotations

import numpy as np
import pytest

import director_ai.core.scoring._nli_accel as nli_accel
import director_ai.core.scoring.nli as nli_mod
from director_ai.core.scoring.nli import NLIScorer

_ACCEL_NAMES = (
    "rust_aggregate_chunk_scores",
    "rust_aggregate_chunk_scores_confidence_weighted",
    "rust_build_chunks",
    "rust_coverage_from_divergences",
    "rust_probs_to_confidence",
    "rust_probs_to_divergence",
    "rust_reduce_claim_attribution",
    "rust_softmax",
    "rust_split_sentences",
)


class TestBindingSurface:
    def test_flag_is_boolean_and_all_names_are_callable(self):
        assert isinstance(nli_accel._RUST_NLI, bool)
        for name in _ACCEL_NAMES:
            assert callable(getattr(nli_accel, name))

    def test_scoring_modules_consume_the_binding_module(self):
        import director_ai.core.scoring._nli_chunking as nli_chunking
        import director_ai.core.scoring._nli_claims as nli_claims
        import director_ai.core.scoring._nli_numeric as nli_numeric

        assert nli_numeric._nli_accel is nli_accel
        assert nli_chunking._nli_accel is nli_accel
        assert nli_claims._nli_accel is nli_accel


class TestCanonicalPatchPoint:
    def test_disabling_the_flag_routes_softmax_to_the_python_floor(self, monkeypatch):
        def _explode(_flat, _cols):
            raise AssertionError("accelerated branch must not run")

        monkeypatch.setattr(nli_accel, "_RUST_NLI", False)
        monkeypatch.setattr(nli_accel, "rust_softmax", _explode)
        x = np.linspace(-2.0, 2.0, 300, dtype=np.float64).reshape(100, 3)
        out = nli_mod._softmax_np(x)
        np.testing.assert_allclose(out.sum(axis=1), np.ones(100), rtol=1e-9)

    def test_enabling_the_flag_dispatches_through_the_binding(self, monkeypatch):
        monkeypatch.setattr(nli_accel, "_RUST_NLI", True)
        monkeypatch.setattr(
            nli_accel,
            "rust_split_sentences",
            lambda _text: ["patched sentence."],
        )
        assert NLIScorer._split_sentences("One. Two.") == ["patched sentence."]

    def test_binding_failure_propagates_as_mandatory(self, monkeypatch):
        def _boom(_flat, _ncols):
            raise RuntimeError("kernel unavailable")

        monkeypatch.setattr(nli_accel, "_RUST_NLI", True)
        monkeypatch.setattr(nli_accel, "rust_probs_to_confidence", _boom)
        probs = np.full((15, 3), 1 / 3, dtype=np.float64)
        with pytest.raises(RuntimeError, match="kernel unavailable"):
            nli_mod._probs_to_confidence(probs)
