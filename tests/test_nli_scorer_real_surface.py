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
from dataclasses import dataclass
from typing import Protocol, cast

import pytest

from director_ai.core.nli import NLIScorer as PublicNLIScorer
from director_ai.core.scoring.nli import NLIScorer as RuntimeNLIScorer
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


@dataclass(frozen=True)
class _MiniCheckCall:
    """Captured MiniCheck package scoring call."""

    docs: list[str]
    claims: list[str]


class _MiniCheckProtocolScorer:
    """Deterministic MiniCheck package protocol for public NLI calls."""

    def __init__(self, scores_by_claim: dict[str, float]) -> None:
        self._scores_by_claim = scores_by_claim
        self.calls: list[_MiniCheckCall] = []

    def score(self, *, docs: list[str], claims: list[str]) -> list[float]:
        """Return support probabilities for the requested claims."""
        assert len(docs) == len(claims)
        self.calls.append(_MiniCheckCall(docs=docs, claims=claims))
        return [self._scores_by_claim[claim] for claim in claims]


class _MiniCheckState(Protocol):
    """Mutable MiniCheck state exposed by NLIScorer instances."""

    _minicheck: _MiniCheckProtocolScorer
    _minicheck_loaded: bool


def _install_minicheck_protocol(
    scorer: PublicNLIScorer,
    minicheck: _MiniCheckProtocolScorer,
) -> None:
    """Install a local MiniCheck protocol object without loading a checkpoint."""
    state = cast(_MiniCheckState, scorer)
    state._minicheck = minicheck
    state._minicheck_loaded = True


def test_nli_scorer_unit_guard_declares_real_surface_companion() -> None:
    """The NLI scorer unit guard should name its public companion surface."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_nli_scorer.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_nli_scorer_real_surface.py" in reason


def test_nli_minicheck_unit_guard_declares_real_surface_companion() -> None:
    """The MiniCheck unit guard should name its public companion surface."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_nli_minicheck.py"
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


def test_public_minicheck_scorer_uses_package_score_contract() -> None:
    """MiniCheck scoring should call the package with docs and claims."""
    minicheck = _MiniCheckProtocolScorer(
        {
            "Refund approval requires a signed receipt.": 0.93,
            "Refund approval can ignore signed receipts.": 0.18,
        }
    )
    scorer = PublicNLIScorer(backend="minicheck")
    _install_minicheck_protocol(scorer, minicheck)

    aligned = scorer.score(
        "Refund approvals require signed receipts.",
        "Refund approval requires a signed receipt.",
    )
    batch = scorer.score_batch(
        [
            (
                "Refund approvals require signed receipts.",
                "Refund approval requires a signed receipt.",
            ),
            (
                "Refund approvals require signed receipts.",
                "Refund approval can ignore signed receipts.",
            ),
        ]
    )

    assert aligned == pytest.approx(0.07)
    assert batch == pytest.approx([0.07, 0.82])
    assert minicheck.calls == [
        _MiniCheckCall(
            docs=["Refund approvals require signed receipts."],
            claims=["Refund approval requires a signed receipt."],
        ),
        _MiniCheckCall(
            docs=[
                "Refund approvals require signed receipts.",
                "Refund approvals require signed receipts.",
            ],
            claims=[
                "Refund approval requires a signed receipt.",
                "Refund approval can ignore signed receipts.",
            ],
        ),
    ]
