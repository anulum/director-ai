# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Defense Update Pipeline Edge Tests
"""Module-specific edge tests for reviewed defense promotion."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from director_ai.core.defense_genome import DefenseRegistry, DefenseUpdatePipeline


class _StaticDefense:
    def score(self, prompt: str) -> float:
        _ = prompt
        return 0.9


def test_pipeline_rejects_invalid_gate_configuration() -> None:
    registry = DefenseRegistry()

    with pytest.raises(ValueError, match="min_adversarial_cases"):
        DefenseUpdatePipeline(registry=registry, min_adversarial_cases=0)
    with pytest.raises(ValueError, match="min_holdout_improvement"):
        DefenseUpdatePipeline(registry=registry, min_holdout_improvement=float("nan"))
    with pytest.raises(ValueError, match="min_holdout_improvement"):
        DefenseUpdatePipeline(registry=registry, min_holdout_improvement=-0.1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"version": 0}, "version"),
        ({"label": " "}, "label"),
        ({"baseline_score": -0.1}, "baseline_score"),
        ({"candidate_score": 1.1}, "candidate_score"),
    ],
)
def test_review_and_promote_rejects_invalid_request_fields(kwargs, message) -> None:
    params = _promotion_kwargs() | kwargs

    with pytest.raises(ValueError, match=message):
        DefenseUpdatePipeline(registry=DefenseRegistry()).review_and_promote(**params)


def test_review_and_promote_rejects_non_allow_guard_decision() -> None:
    params = _promotion_kwargs()
    params["proposal"] = _proposal(decision="block")

    with pytest.raises(PermissionError, match="guard decision"):
        DefenseUpdatePipeline(registry=DefenseRegistry()).review_and_promote(**params)


def test_review_and_promote_rejects_blank_approval_id() -> None:
    params = _promotion_kwargs()
    params["proposal"] = _proposal(approval_id=" ")

    with pytest.raises(PermissionError, match="approved"):
        DefenseUpdatePipeline(registry=DefenseRegistry()).review_and_promote(**params)


def test_review_and_promote_rejects_insufficient_holdout_delta() -> None:
    pipeline = DefenseUpdatePipeline(
        registry=DefenseRegistry(),
        min_holdout_improvement=0.2,
    )
    params = _promotion_kwargs(candidate_score=0.75, baseline_score=0.70)

    with pytest.raises(ValueError, match="holdout score"):
        pipeline.review_and_promote(**params)


def _promotion_kwargs(**overrides):
    params = {
        "proposal": _proposal(),
        "evolve_report": _evolve_report(),
        "defense": _StaticDefense(),
        "version": 1,
        "label": "defense-v1",
        "baseline_score": 0.70,
        "candidate_score": 0.90,
    }
    params.update(overrides)
    return params


def _proposal(*, decision: str = "allow", approval_id: str = "approval-1"):
    return SimpleNamespace(
        proposal_id="proposal-1",
        proposal_type="lora_training_job",
        guard_decision=SimpleNamespace(decision=decision),
        approved=True,
        approval_id=approval_id,
        manifest=SimpleNamespace(manifest_id="manifest-1", event_count=4),
        rollback_id="defense-v0",
    )


def _evolve_report():
    return SimpleNamespace(
        version=SimpleNamespace(version=7),
        adversarial_case_count=3,
        mined_pattern_count=2,
        promotion_reason="candidate passed adversarial replay",
    )
