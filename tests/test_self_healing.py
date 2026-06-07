# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Self-Healing Threshold Controller Tests
"""Multi-angle tests for holdout-validated threshold adaptation + rollback."""

from __future__ import annotations

import math

import pytest

from director_ai.core.self_healing import (
    ACCEPT,
    INSUFFICIENT_DATA,
    NO_PRIOR,
    REJECT,
    ROLLBACK,
    STABLE,
    LabelledOutcome,
    SelfHealingThresholdController,
    TuningConfig,
)


def _drift_batch(n: int = 18) -> list[LabelledOutcome]:
    """Grounded answers score ~0.65, hallucinations ~0.2 → optimum well below 0.9."""
    out = []
    for i in range(n):
        if i % 2 == 0:
            out.append(LabelledOutcome(score=0.65, grounded=True))
        else:
            out.append(LabelledOutcome(score=0.2, grounded=False))
    return out


class TestValidation:
    def test_outcome_score_range(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            LabelledOutcome(score=1.5, grounded=True)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            LabelledOutcome(score=math.nan, grounded=True)

    def test_config_holdout_fraction(self):
        with pytest.raises(ValueError, match="holdout_fraction"):
            TuningConfig(holdout_fraction=0.0)
        with pytest.raises(ValueError, match="holdout_fraction"):
            TuningConfig(holdout_fraction=1.0)

    def test_config_min_samples(self):
        with pytest.raises(ValueError, match="min_samples"):
            TuningConfig(min_samples=1)

    def test_config_regression_tolerance(self):
        with pytest.raises(ValueError, match="regression_tolerance"):
            TuningConfig(regression_tolerance=-0.1)

    def test_config_threshold_bounds(self):
        with pytest.raises(ValueError, match="threshold_min"):
            TuningConfig(threshold_min=0.8, threshold_max=0.5)

    def test_initial_threshold_must_be_in_bounds(self):
        with pytest.raises(ValueError, match="initial_threshold"):
            SelfHealingThresholdController(1.5)


class TestPropose:
    def test_insufficient_data_keeps_buffer(self):
        c = SelfHealingThresholdController(0.6)
        c.observe(LabelledOutcome(0.7, True))
        update = c.propose()
        assert update.action == INSUFFICIENT_DATA
        assert update.changed is False
        assert c.pending == 1  # buffer NOT consumed
        assert c.threshold == 0.6

    def test_drift_accepts_and_moves_threshold(self):
        c = SelfHealingThresholdController(0.9)
        c.observe_many(_drift_batch())
        update = c.propose()
        assert update.action == ACCEPT
        assert update.changed is True
        assert update.new_threshold < 0.9
        assert update.holdout_error_new < update.holdout_error_old
        assert c.threshold == update.new_threshold
        assert c.previous_threshold == 0.9
        assert c.pending == 0  # window consumed

    def test_no_separation_is_rejected(self):
        c = SelfHealingThresholdController(0.5)
        # Identical score for both labels — no threshold can separate them.
        c.observe_many([LabelledOutcome(0.5, grounded=(i % 2 == 0)) for i in range(18)])
        update = c.propose()
        assert update.action == REJECT
        assert c.threshold == 0.5
        assert c.previous_threshold is None
        assert c.pending == 0

    def test_holdout_gate_blocks_support_overfit(self):
        # Split stride for holdout_fraction 0.34 is 3 → holdout = indices 0,3,6,...
        # Holdout favours the current 0.5; support favours ~0.7. The candidate that
        # fits support worsens the holdout, so it must be rejected.
        c = SelfHealingThresholdController(0.5)
        batch = []
        for i in range(18):
            if i % 3 == 0:  # holdout
                batch.append(
                    LabelledOutcome(0.6, grounded=True)
                    if (i // 3) % 2 == 0
                    else LabelledOutcome(0.4, grounded=False)
                )
            else:  # support → cleanly separable at ~0.7
                batch.append(
                    LabelledOutcome(0.8, grounded=True)
                    if i % 2 == 0
                    else LabelledOutcome(0.65, grounded=False)
                )
        c.observe_many(batch)
        update = c.propose()
        assert update.action == REJECT
        assert update.holdout_error_new >= update.holdout_error_old
        assert c.threshold == 0.5

    def test_propose_is_deterministic(self):
        a = SelfHealingThresholdController(0.9)
        b = SelfHealingThresholdController(0.9)
        a.observe_many(_drift_batch())
        b.observe_many(_drift_batch())
        ua, ub = a.propose(), b.propose()
        assert ua.action == ub.action
        assert ua.new_threshold == ub.new_threshold


class TestRollback:
    def _accepted_controller(self) -> SelfHealingThresholdController:
        c = SelfHealingThresholdController(0.9)
        c.observe_many(_drift_batch())
        assert c.propose().action == ACCEPT
        return c

    def test_regression_rolls_back_to_previous(self):
        c = self._accepted_controller()
        moved_to = c.threshold
        prior = c.previous_threshold
        # Fresh data where the same 0.65-scoring answers are now hallucinations:
        # the lowered threshold approves them (bad), the prior 0.9 would not.
        fresh = [LabelledOutcome(0.65, grounded=False) for _ in range(8)]
        update = c.evaluate_regression(fresh)
        assert update.action == ROLLBACK
        assert update.changed is True
        assert c.threshold == prior
        assert c.threshold != moved_to
        assert c.previous_threshold is None  # cannot roll back twice

    def test_stable_update_is_kept(self):
        c = self._accepted_controller()
        moved_to = c.threshold
        # Fresh data consistent with the accepted policy → no rollback.
        fresh = _drift_batch(8)
        update = c.evaluate_regression(fresh)
        assert update.action == STABLE
        assert c.threshold == moved_to

    def test_no_prior_when_nothing_deployed(self):
        c = SelfHealingThresholdController(0.6)
        update = c.evaluate_regression([LabelledOutcome(0.5, True)])
        assert update.action == NO_PRIOR
        assert c.threshold == 0.6

    def test_empty_regression_batch_is_stable(self):
        c = self._accepted_controller()
        moved_to = c.threshold
        # No fresh evidence → zero error both ways → no rollback.
        update = c.evaluate_regression([])
        assert update.action == STABLE
        assert c.threshold == moved_to


class TestAuditAndWeights:
    def test_history_records_every_decision(self):
        c = SelfHealingThresholdController(0.9)
        c.propose()  # insufficient
        c.observe_many(_drift_batch())
        c.propose()  # accept
        actions = [u.action for u in c.history]
        assert actions == [INSUFFICIENT_DATA, ACCEPT]
        audit = c.audit()
        assert audit[-1]["action"] == ACCEPT
        assert set(audit[-1]) == {
            "action",
            "old_threshold",
            "new_threshold",
            "holdout_error_old",
            "holdout_error_new",
            "sample_count",
            "reason",
        }

    def test_missed_hallucination_weight_shifts_optimum(self):
        # Heavily penalising missed hallucinations pushes the threshold up so the
        # ambiguous mid-score not-grounded answers are rejected.
        cfg = TuningConfig(missed_hallucination_weight=5.0)
        c = SelfHealingThresholdController(0.3, cfg)
        batch = []
        for i in range(18):
            if i % 2 == 0:
                batch.append(LabelledOutcome(0.9, grounded=True))
            else:
                batch.append(LabelledOutcome(0.55, grounded=False))
        c.observe_many(batch)
        update = c.propose()
        assert update.action == ACCEPT
        assert c.threshold > 0.55  # rejects the 0.55 hallucinations


class TestGuardWiring:
    def test_guard_self_healing_persists(self):
        from director_ai.core.config import DirectorConfig
        from director_ai.guard import ProductionGuard

        guard = ProductionGuard(DirectorConfig(use_nli=False, llm_provider="mock"))
        controller = guard.self_healing
        assert controller.threshold == DirectorConfig().coherence_threshold
        assert guard.self_healing is controller  # persists across calls
        controller.observe_many(_drift_batch())
        # The seeded threshold (0.6) already separates the 0.65/0.2 batch well, so
        # the controller may keep or refine it — either way it audits the decision.
        update = controller.propose()
        assert update.action in (ACCEPT, REJECT)
        assert len(controller.history) == 1
