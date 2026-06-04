# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — auto-redteam defence loop tests

from __future__ import annotations

import json
from collections.abc import Iterable

import pytest

from director_ai.core.continual_adversarial import FailureEvent
from director_ai.core.defense_genome import (
    AutoRedteamCycleInput,
    AutoRedteamDefenceLoop,
    DefenseRegistry,
)
from director_ai.core.guard_control import GuardDecision, RiskEnvelope
from director_ai.core.self_evolving import (
    GuardLoopProposal,
    ReviewedFeedbackManifest,
)


class _KeywordDefence:
    def __init__(self, markers: Iterable[str]) -> None:
        self._markers = tuple(marker.lower() for marker in markers)

    def score(self, prompt: str) -> float:
        text = prompt.lower()
        return 0.1 if any(marker in text for marker in self._markers) else 0.9


def _risk_envelope() -> RiskEnvelope:
    return RiskEnvelope(
        action_category="training",
        reversibility="costly",
        domain="security",
        calibrated_threshold=0.5,
        no_go_threshold=0.8,
    )


def _approved_proposal(proposal_id: str = "proposal-redteam-v2") -> GuardLoopProposal:
    event_refs = tuple(f"feedback-{index}" for index in range(8))
    manifest = ReviewedFeedbackManifest(
        manifest_id=f"manifest-{proposal_id}",
        source_ref="feedback://reviewed-redteam-window",
        event_count=len(event_refs),
        label_counts={"unsafe": len(event_refs)},
        reviewer_ids=("reviewer-passport-a",),
        event_refs=event_refs,
    )
    decision = GuardDecision(
        decision="allow",
        risk_score=0.1,
        confidence_low=0.72,
        confidence_high=0.84,
        policy_id="policy.auto-redteam",
        reason="self_improvement_training_ready",
        tenant_safe_explanation="Reviewed feedback supports promotion.",
        evidence_refs=event_refs,
        verifier_signals=(),
        risk_envelope=_risk_envelope(),
        attributes={
            "proposal_type": "lora_training_job",
            "manifest_id": manifest.manifest_id,
        },
    )
    return GuardLoopProposal(
        proposal_id=proposal_id,
        proposal_type="lora_training_job",
        manifest=manifest,
        rollback_id="defence-v1",
        guard_decision=decision,
        payload={
            "dataset_uri": "s3://tenant-safe-artifacts/defence-v2.jsonl",
            "base_model_ref": "registry://guard/base@sha256:abc123",
        },
        approved=True,
        approval_id=f"approval-{proposal_id}",
    )


def _failures(marker: str, count: int = 8) -> tuple[FailureEvent, ...]:
    return tuple(
        FailureEvent(
            prompt=marker,
            label="unsafe",
            timestamp=float(index),
            metadata={"source": "redteam-fixture"},
        )
        for index in range(count)
    )


def _cycle(
    *,
    marker: str,
    proposal_id: str,
    defence: _KeywordDefence,
    version: int,
    label: str,
) -> AutoRedteamCycleInput:
    return AutoRedteamCycleInput(
        failures=_failures(marker),
        safe_corpus=("normal grounded request", "summarise approved policy"),
        proposal=_approved_proposal(proposal_id),
        candidate_defence=defence,
        version=version,
        label=label,
        baseline_score=0.72,
        candidate_score=0.84,
    )


def test_auto_redteam_loop_promotes_candidate_after_detection_uplift() -> None:
    registry = DefenseRegistry()
    baseline = registry.promote(
        defense=_KeywordDefence(()),
        version=1,
        label="defence-v1",
    )
    loop = AutoRedteamDefenceLoop(
        registry=registry,
        min_failures=6,
        min_detection_uplift=0.5,
    )

    report = loop.run_cycle(
        _cycle(
            marker="bypass alpha guard",
            proposal_id="proposal-redteam-v2",
            defence=_KeywordDefence(("bypass alpha guard",)),
            version=2,
            label="defence-v2",
        )
    )

    active = registry.active()

    assert active is not None
    assert active.version == 2
    assert registry.history() == (baseline,)
    assert report.promoted is True
    assert report.promoted_version == 2
    assert report.baseline_detection_rate == 0.0
    assert report.candidate_detection_rate == 1.0
    assert report.detection_uplift == 1.0
    assert report.adversarial_case_count >= 1
    assert report.metadata["proposal_id"] == "proposal-redteam-v2"
    assert report.metadata["redteam_detection_uplift"] == "1.000000"


def test_auto_redteam_loop_runs_repeated_cycles_against_current_baseline() -> None:
    registry = DefenseRegistry()
    registry.promote(defense=_KeywordDefence(()), version=1, label="defence-v1")
    loop = AutoRedteamDefenceLoop(
        registry=registry,
        min_failures=6,
        min_detection_uplift=0.5,
    )

    reports = loop.run(
        (
            _cycle(
                marker="bypass alpha guard",
                proposal_id="proposal-redteam-v2",
                defence=_KeywordDefence(("bypass alpha guard",)),
                version=2,
                label="defence-v2",
            ),
            _cycle(
                marker="exfiltrate beta policy",
                proposal_id="proposal-redteam-v3",
                defence=_KeywordDefence(
                    ("bypass alpha guard", "exfiltrate beta policy")
                ),
                version=3,
                label="defence-v3",
            ),
        )
    )

    active = registry.active()

    assert [report.promoted_version for report in reports] == [2, 3]
    assert active is not None
    assert active.version == 3
    assert [snapshot.version for snapshot in registry.history()] == [1, 2]
    assert reports[1].baseline_detection_rate == 0.0
    assert reports[1].candidate_detection_rate == 1.0


def test_auto_redteam_loop_rejects_no_uplift_before_registry_mutation() -> None:
    registry = DefenseRegistry()
    baseline = registry.promote(
        defense=_KeywordDefence(()),
        version=1,
        label="defence-v1",
    )
    loop = AutoRedteamDefenceLoop(
        registry=registry,
        min_failures=6,
        min_detection_uplift=0.5,
    )

    with pytest.raises(ValueError, match="detection uplift"):
        loop.run_cycle(
            _cycle(
                marker="bypass alpha guard",
                proposal_id="proposal-redteam-v2",
                defence=_KeywordDefence(("unrelated marker",)),
                version=2,
                label="defence-v2",
            )
        )

    assert registry.active() is baseline
    assert registry.history() == ()


def test_auto_redteam_loop_requires_active_baseline_defence() -> None:
    loop = AutoRedteamDefenceLoop(
        registry=DefenseRegistry(),
        min_failures=6,
        min_detection_uplift=0.5,
    )

    with pytest.raises(ValueError, match="active baseline"):
        loop.run_cycle(
            _cycle(
                marker="bypass alpha guard",
                proposal_id="proposal-redteam-v2",
                defence=_KeywordDefence(("bypass alpha guard",)),
                version=2,
                label="defence-v2",
            )
        )


def test_auto_redteam_report_serialises_without_raw_prompts_or_defence_objects() -> None:
    registry = DefenseRegistry()
    registry.promote(defense=_KeywordDefence(()), version=1, label="defence-v1")
    loop = AutoRedteamDefenceLoop(
        registry=registry,
        min_failures=6,
        min_detection_uplift=0.5,
    )

    report = loop.run_cycle(
        _cycle(
            marker="bypass alpha guard",
            proposal_id="proposal-redteam-v2",
            defence=_KeywordDefence(("bypass alpha guard",)),
            version=2,
            label="defence-v2",
        )
    )
    serialised = json.dumps(report.to_dict(), sort_keys=True)

    assert "bypass alpha guard" not in serialised
    assert "_KeywordDefence" not in serialised
    assert report.pattern_digest
