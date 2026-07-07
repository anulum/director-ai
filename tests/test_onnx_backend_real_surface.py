# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - ONNX backend real-surface tests
"""Real-surface coverage for public ONNX backend wiring."""

from __future__ import annotations

import pytest

from director_ai.core import CoherenceScorer
from director_ai.core import export_onnx as core_export_onnx
from director_ai.core.nli import NLIScorer as PublicNLIScorer
from director_ai.core.nli import export_onnx as compatibility_export_onnx
from director_ai.core.scoring.nli import NLIScorer as RuntimeNLIScorer
from director_ai.core.scoring.nli import export_onnx as runtime_export_onnx
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


def test_onnx_backend_unit_guard_declares_real_surface_companion() -> None:
    """The ONNX backend unit guard should name its public companion surface."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_onnx_backend.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_onnx_backend_real_surface.py" in reason


def test_public_onnx_backend_paths_share_runtime_scorer() -> None:
    """Public compatibility NLI imports should resolve to the runtime scorer."""
    assert PublicNLIScorer is RuntimeNLIScorer
    assert core_export_onnx is runtime_export_onnx
    assert compatibility_export_onnx is runtime_export_onnx


def test_public_onnx_backend_falls_back_without_artifact_directory() -> None:
    """The public ONNX backend should fall back safely without an artifact dir."""
    scorer = PublicNLIScorer(use_model=True, backend="onnx")

    assert scorer.backend == "onnx"
    assert scorer.model_available is False
    assert 0.0 <= scorer.score("premise", "hypothesis") <= 1.0
    assert all(
        0.0 <= score <= 1.0
        for score in scorer.score_batch([("same", "same"), ("left", "right")])
    )


def test_public_coherence_scorer_accepts_onnx_backend_contract() -> None:
    """CoherenceScorer should route ONNX through the public NLI contract."""
    scorer = CoherenceScorer(use_nli=False, scorer_backend="onnx")

    approved, score = scorer.review("premise", "hypothesis")

    assert isinstance(approved, bool)
    assert 0.0 <= score.score <= 1.0


@pytest.mark.parametrize("quantize", ["nf4", "awq"])
def test_public_onnx_export_rejects_unknown_quantize_mode(quantize: str) -> None:
    """Public export wiring should reject unsupported quantization modes."""
    with pytest.raises(ValueError, match="quantize"):
        core_export_onnx(output_dir="unused", quantize=quantize)
