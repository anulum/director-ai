# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — NLI Scorer Real Surface Tests
"""Real-surface coverage for the public NLI scorer compatibility path."""

from __future__ import annotations

import importlib

import pytest

from director_ai.core.nli import NLIScorer as PublicNLIScorer
from director_ai.core.scoring.nli import NLIScorer as RuntimeNLIScorer
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


def test_nli_scorer_unit_guard_declares_real_surface_companion() -> None:
    """The NLI scorer unit guard should name its public companion surface."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_nli_scorer.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_nli_scorer_real_surface.py" in reason


def test_public_nli_compatibility_path_uses_runtime_scorer() -> None:
    """The public compatibility import should expose the runtime scorer."""
    public_module = importlib.import_module("director_ai.core.nli")
    runtime_module = importlib.import_module("director_ai.core.scoring.nli")

    assert public_module is runtime_module
    assert PublicNLIScorer is RuntimeNLIScorer

    scorer = PublicNLIScorer(use_model=False)
    aligned = scorer.score(
        "Refund approval requires a signature.",
        "This is consistent with reality.",
    )
    contradicted = scorer.score(
        "Refund approval requires a signature.",
        "The opposite is true.",
    )
    batch = scorer.score_batch(
        [
            (
                "Refund approval requires a signature.",
                "This is consistent with reality.",
            ),
            ("Refund approval requires a signature.", "The opposite is true."),
        ],
    )

    assert aligned == pytest.approx(0.1)
    assert contradicted == pytest.approx(0.9)
    assert batch == [aligned, contradicted]
