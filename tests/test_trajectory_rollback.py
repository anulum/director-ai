# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — trajectory rollback hook tests

from __future__ import annotations

from types import SimpleNamespace

import pytest

from director_ai.core.trajectory import (
    PreflightVerdict,
    RollbackHandle,
    RollbackOutcome,
    TrajectoryResult,
    TrajectoryRollbackManager,
)


def _trajectory(idx: int, *, approved: bool) -> TrajectoryResult:
    return TrajectoryResult(
        trajectory_id=idx,
        seed=17 + idx,
        tokens=("tenant-safe",),
        final_coherence=0.9 if approved else 0.1,
        approved=approved,
    )


def _verdict(recommended: str, approvals: list[bool]) -> PreflightVerdict:
    trajectories = tuple(
        _trajectory(idx, approved=approved) for idx, approved in enumerate(approvals)
    )
    halt_rate = sum(not approved for approved in approvals) / len(approvals)
    return PreflightVerdict(
        n_simulations=len(approvals),
        halt_rate=halt_rate,
        mean_coherence=0.8,
        std_coherence=0.1,
        ci_low=0.7,
        ci_high=0.9,
        recommended=recommended,  # type: ignore[arg-type]
        reason="test verdict",
        trajectories=trajectories,
    )


def _register(
    manager: TrajectoryRollbackManager,
    calls: list[tuple[str, str]],
    *,
    rollback_id: str = "rb-1",
) -> RollbackHandle:
    def hook(handle: RollbackHandle, reason: str) -> dict[str, str]:
        calls.append((handle.rollback_id, reason))
        return {"rollback_store": "audit-log"}

    return manager.register(
        rollback_id=rollback_id,
        action_id="deploy-policy",
        tenant_id="tenant-a",
        hook=hook,
        evidence_refs=("change:42",),
        metadata={"owner": "safety"},
    )


def test_register_requires_stable_ids() -> None:
    manager = TrajectoryRollbackManager()

    with pytest.raises(ValueError, match="rollback_id"):
        manager.register(rollback_id="", action_id="a", hook=lambda _h, _r: {})
    with pytest.raises(ValueError, match="action_id"):
        manager.register(rollback_id="rb", action_id="", hook=lambda _h, _r: {})


def test_duplicate_rollback_id_is_rejected() -> None:
    manager = TrajectoryRollbackManager()
    calls: list[tuple[str, str]] = []
    _register(manager, calls)

    with pytest.raises(ValueError, match="already registered"):
        _register(manager, calls)


def test_metadata_rejects_raw_or_secret_keys() -> None:
    manager = TrajectoryRollbackManager()

    with pytest.raises(ValueError, match="tenant-safe"):
        manager.register(
            rollback_id="rb",
            action_id="deploy-policy",
            hook=lambda _h, _r: {},
            metadata={"raw_prompt": "blocked"},
        )


def test_proceed_verdict_keeps_rollback_not_required() -> None:
    manager = TrajectoryRollbackManager()
    calls: list[tuple[str, str]] = []
    handle = _register(manager, calls)

    outcome = manager.evaluate_preflight(
        handle.rollback_id, _verdict("proceed", [True])
    )

    assert outcome.status == "not_required"
    assert outcome.executed is False
    assert calls == []


def test_warn_verdict_arms_without_executing_hook() -> None:
    manager = TrajectoryRollbackManager()
    calls: list[tuple[str, str]] = []
    handle = _register(manager, calls)

    outcome = manager.evaluate_preflight(
        handle.rollback_id,
        _verdict("warn", [True, False, True, True]),
    )

    assert outcome.status == "armed"
    assert outcome.reason == "trajectory_preflight_uncertain"
    assert outcome.executed is False
    assert outcome.evidence_refs == ("trajectory:1",)
    assert calls == []


def test_halt_verdict_executes_hook_and_merges_evidence_refs() -> None:
    manager = TrajectoryRollbackManager()
    calls: list[tuple[str, str]] = []
    handle = _register(manager, calls)

    outcome = manager.evaluate_preflight(
        handle.rollback_id,
        _verdict("halt", [False, False, True]),
    )

    assert outcome.status == "executed"
    assert outcome.executed is True
    assert outcome.evidence_refs == ("change:42", "trajectory:0", "trajectory:1")
    assert outcome.metadata == {"owner": "safety", "rollback_store": "audit-log"}
    assert calls == [(handle.rollback_id, "trajectory_preflight_halt")]


def test_steering_halt_executes_even_when_verdict_warns() -> None:
    manager = TrajectoryRollbackManager()
    calls: list[tuple[str, str]] = []
    handle = _register(manager, calls)
    steering = SimpleNamespace(action="halt", evidence_refs=("steering:1",))

    outcome = manager.evaluate_preflight(
        handle.rollback_id,
        _verdict("warn", [True, False]),
        steering_decision=steering,
    )

    assert outcome.status == "executed"
    assert outcome.evidence_refs == ("change:42", "steering:1", "trajectory:1")


def test_execute_is_idempotent_after_success() -> None:
    manager = TrajectoryRollbackManager()
    calls: list[tuple[str, str]] = []
    handle = _register(manager, calls)

    first = manager.execute(handle.rollback_id, reason="manual_halt")
    second = manager.execute(handle.rollback_id, reason="manual_halt")

    assert first.status == "executed"
    assert first.executed is True
    assert second.status == "already_executed"
    assert second.executed is False
    assert calls == [(handle.rollback_id, "manual_halt")]


def test_hook_failure_reports_error_type_without_marking_executed() -> None:
    manager = TrajectoryRollbackManager()

    def hook(_handle: RollbackHandle, _reason: str) -> None:
        raise RuntimeError("raw backend message should not be serialised")

    handle = manager.register(
        rollback_id="rb",
        action_id="deploy-policy",
        hook=hook,
    )

    outcome = manager.execute(handle.rollback_id, reason="manual_halt")
    payload = outcome.to_dict()

    assert outcome.status == "failed"
    assert outcome.executed is False
    assert outcome.error_type == "RuntimeError"
    assert "raw backend message" not in str(payload)


def test_outcome_to_dict_is_json_safe() -> None:
    outcome = RollbackOutcome(
        rollback_id="rb",
        action_id="deploy-policy",
        status="armed",
        reason="trajectory_preflight_uncertain",
        executed=False,
        tenant_id="tenant-a",
        evidence_refs=("trajectory:1",),
        metadata={"owner": "safety"},
    )

    assert outcome.to_dict() == {
        "rollback_id": "rb",
        "action_id": "deploy-policy",
        "status": "armed",
        "reason": "trajectory_preflight_uncertain",
        "executed": False,
        "tenant_id": "tenant-a",
        "evidence_refs": ["trajectory:1"],
        "metadata": {"owner": "safety"},
    }


def test_unknown_rollback_id_raises_key_error() -> None:
    manager = TrajectoryRollbackManager()

    with pytest.raises(KeyError, match="unknown rollback_id"):
        manager.execute("missing", reason="manual_halt")
