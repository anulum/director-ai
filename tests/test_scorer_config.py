# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ScorerConfig / from_config tests

from __future__ import annotations

import pytest

from director_ai.core.retrieval.knowledge import GroundTruthStore
from director_ai.core.scoring.scorer import CoherenceScorer
from director_ai.core.scoring.scorer_config import ScorerConfig


def test_defaults_match_constructor():
    # ScorerConfig defaults must mirror the per-argument constructor defaults,
    # so from_config(ScorerConfig()) is equivalent to CoherenceScorer().
    cfg = ScorerConfig()
    a = CoherenceScorer.from_config(cfg)
    b = CoherenceScorer()
    assert a.threshold == b.threshold == 0.5
    assert a.W_LOGIC == b.W_LOGIC
    assert a.W_FACT == b.W_FACT
    assert a.strict_mode == b.strict_mode is False
    assert a.scorer_backend == b.scorer_backend == "deberta"


def test_from_config_applies_values():
    cfg = ScorerConfig(
        threshold=0.7,
        soft_limit=0.85,
        w_logic=0.5,
        w_fact=0.5,
        strict_mode=True,
        use_nli=False,
    )
    scorer = CoherenceScorer.from_config(cfg)
    assert scorer.threshold == 0.7
    assert scorer.soft_limit == 0.85
    assert scorer.W_LOGIC == 0.5
    assert scorer.W_FACT == 0.5
    assert scorer.strict_mode is True


def test_from_config_injects_dependencies():
    store = GroundTruthStore()
    store.add("sky", "the sky is blue")
    scorer = CoherenceScorer.from_config(
        ScorerConfig(use_nli=False), ground_truth_store=store
    )
    assert scorer.ground_truth_store is store


def test_from_config_scores_like_constructor():
    store = GroundTruthStore()
    store.add("boiling", "water boils at 100 C")
    cfg = ScorerConfig(threshold=0.6, use_nli=False)
    via_config = CoherenceScorer.from_config(cfg, ground_truth_store=store)
    direct = CoherenceScorer(threshold=0.6, use_nli=False, ground_truth_store=store)
    p, r = "boiling point?", "Water boils at 100 C."
    assert via_config.review(p, r)[1].score == direct.review(p, r)[1].score


def test_to_kwargs_round_trips():
    cfg = ScorerConfig(threshold=0.65, strict_mode=True)
    kwargs = cfg.to_kwargs()
    assert kwargs["threshold"] == 0.65
    assert kwargs["strict_mode"] is True
    # every key must be a real constructor parameter (no drift)
    import inspect

    params = set(inspect.signature(CoherenceScorer.__init__).parameters)
    assert set(kwargs).issubset(params)


@pytest.mark.parametrize(
    "bad",
    [
        {"threshold": 1.5},
        {"soft_limit": 0.2, "threshold": 0.5},  # soft_limit < threshold
        {"w_logic": 2.0},
        {"history_window": 0},
    ],
)
def test_validation_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        ScorerConfig(**bad)


def test_frozen():
    cfg = ScorerConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.threshold = 0.9  # type: ignore[misc]
