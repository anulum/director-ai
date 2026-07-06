# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - backend registry real-surface tests
"""Real public-surface coverage for scorer backend registry wiring."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType

import director_ai
from director_ai.core.config import DirectorConfig
from director_ai.core.scoring import backends as scoring_backends
from director_ai.core.scoring.backends import (
    LiteBackend,
    ScorerBackend,
    get_backend,
    list_backends,
    register_backend,
)
from director_ai.core.scoring.scorer import CoherenceScorer
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


class _LengthDeltaBackend(ScorerBackend):
    """Deterministic backend registered through the public registry API."""

    def score(self, premise: str, hypothesis: str) -> float:
        """Return a bounded divergence based on text length delta."""
        return min(abs(len(hypothesis) - len(premise)) / 100.0, 1.0)

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score each pair with the same production registry contract."""
        return [self.score(premise, hypothesis) for premise, hypothesis in pairs]


def _module(name: str) -> ModuleType:
    """Import *name* and return the runtime module object."""
    return importlib.import_module(name)


def _runtime_symbol(module: ModuleType, name: str) -> object:
    """Resolve a symbol through the runtime module surface."""
    return getattr(module, name)


def test_legacy_core_backend_import_path_is_live_registry_module() -> None:
    """The compatibility import path should expose the same registry module."""
    compat_module = _module("director_ai.core.backends")

    assert compat_module is scoring_backends
    assert _runtime_symbol(compat_module, "LiteBackend") is LiteBackend
    assert _runtime_symbol(director_ai, "ScorerBackend") is ScorerBackend
    assert _runtime_symbol(director_ai, "list_backends") is list_backends
    assert _runtime_symbol(director_ai, "register_backend") is register_backend


def test_scorer_backend_unit_guard_declares_this_companion() -> None:
    """The scorer backend unit guard should be backed by this real surface."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_scorer_backend.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_backends_real_surface.py" in reason


def test_builtin_lite_backend_is_available_through_public_registry() -> None:
    """The zero-dependency Lite backend should score real text via registry."""
    registry = list_backends()

    assert registry["lite"] is LiteBackend
    assert issubclass(registry["lite"], ScorerBackend)

    backend_class = get_backend("lite")
    backend = backend_class()
    aligned = backend.score("Saturn has visible rings.", "Saturn has visible rings.")
    divergent = backend.score(
        "Saturn has visible rings.",
        "The invoice was approved yesterday.",
    )
    batch = backend.score_batch(
        [
            ("Water freezes at 0C.", "Water freezes at 0C."),
            ("Water freezes at 0C.", "Coffee grows on Mars."),
        ]
    )

    assert 0.0 <= aligned <= 1.0
    assert 0.0 <= divergent <= 1.0
    assert aligned < divergent
    assert len(batch) == 2
    assert all(0.0 <= score <= 1.0 for score in batch)


def test_config_file_lite_backend_drives_public_scorer_review(tmp_path: Path) -> None:
    """A real config file should build a working Lite backend scorer."""
    config_path = tmp_path / "director-scorer.json"
    config_path.write_text(
        json.dumps(
            {
                "mode": "general",
                "coherence_threshold": 0.2,
                "hard_limit": 0.1,
                "soft_limit": 0.3,
                "scorer_backend": "lite",
                "use_nli": False,
                "vector_backend": "memory",
                "hybrid_retrieval": False,
                "reranker_enabled": False,
            }
        ),
        encoding="utf-8",
    )

    config = DirectorConfig.from_yaml(str(config_path))
    scorer = config.build_scorer()

    approved, score = scorer.review(
        "Saturn has visible rings.",
        "Saturn has visible rings.",
    )

    assert scorer.scorer_backend == "lite"
    assert approved is True
    assert score.approved is True
    assert score.score >= config.coherence_threshold


def test_custom_backend_registered_through_public_api_scores_batches() -> None:
    """A third-party backend should round-trip through registry APIs."""
    register_backend("test_real_surface_length_delta", _LengthDeltaBackend)

    backend_class = get_backend("test_real_surface_length_delta")
    backend = backend_class()

    assert isinstance(backend, _LengthDeltaBackend)
    assert backend.score("abcd", "abcd") == 0.0
    assert backend.score("abcd", "abcdefghij") == 0.06
    assert backend.score_batch([("same", "same"), ("short", "much longer")]) == [
        0.0,
        0.06,
    ]


def test_lite_registry_backend_drives_coherence_scorer_review() -> None:
    """CoherenceScorer should consume the real Lite registry backend."""
    scorer = CoherenceScorer(scorer_backend="lite", use_nli=False)

    approved, score = scorer.review(
        "The verification receipt was signed.",
        "The verification receipt was signed.",
    )

    assert approved is True
    assert score.approved is True
    assert score.score > 0.5
