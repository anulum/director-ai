# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — standalone interlock tests

from __future__ import annotations

import pytest

from director_ai.interlock import InterlockDecision, InterlockKernel, InterlockPolicy


def test_interlock_policy_validates_thresholds_and_identifiers() -> None:
    with pytest.raises(ValueError, match=r"hard_limit must be finite and in \[0, 1\]"):
        InterlockPolicy(hard_limit=float("nan"))
    with pytest.raises(ValueError, match="window_size must be >= 1"):
        InterlockPolicy(window_size=0)
    with pytest.raises(ValueError, match="trend_window must be >= 0"):
        InterlockPolicy(trend_window=-1)
    with pytest.raises(ValueError, match="hook_id is required"):
        InterlockPolicy(hook_id=" ")
    with pytest.raises(ValueError, match="policy_id is required"):
        InterlockPolicy(policy_id=" ")
    with pytest.raises(ValueError, match="tenant_safe_explanation is required"):
        InterlockPolicy(tenant_safe_explanation=" ")


def test_interlock_decision_validates_decision_scores_and_evidence_refs() -> None:
    with pytest.raises(ValueError, match="unsupported decision"):
        InterlockDecision(decision="block", output="", scores=())
    with pytest.raises(ValueError, match=r"score must be finite and in \[0, 1\]"):
        InterlockDecision(decision="allow", output="", scores=(1.2,))

    decision = InterlockDecision(
        decision="warn",
        output="accepted",
        scores=(0.7,),
        evidence_refs=(123,),
    )

    assert decision.evidence_refs == ("123",)
    assert decision.to_dict()["halt_event"] is None


def test_interlock_allows_stream_with_bring_your_own_scorer() -> None:
    kernel = InterlockKernel(InterlockPolicy(hard_limit=0.4))

    result = kernel.run(["safe", " text"], scorer=lambda text: 0.91)

    assert result.decision == "allow"
    assert result.output == "safe text"
    assert result.halt_event is None
    assert result.scores == (0.91, 0.91)


def test_interlock_halts_before_appending_low_score_token() -> None:
    scores = iter([0.9, 0.2])
    kernel = InterlockKernel(
        InterlockPolicy(
            hard_limit=0.5,
            hook_id="standalone.test",
            policy_id="policy.interlock.regulated",
        )
    )

    result = kernel.run(["safe", " unsafe"], scorer=lambda text: next(scores))

    assert result.decision == "halt"
    assert result.output == "safe"
    assert result.halt_index == 1
    assert result.halt_reason == "hard_limit"
    assert result.halt_event is not None
    assert result.halt_event.policy_decision == "halt"
    assert result.halt_event.evidence_refs == ("interlock://token/1",)
    assert result.halt_event.attributes["policy_id"] == "policy.interlock.regulated"
    assert result.halt_event.request_id == ""
    assert result.halt_event.tenant_id == ""
    assert "unsafe" not in str(result.to_dict())


def test_interlock_downward_trend_halts_with_request_context() -> None:
    scores = iter([0.9, 0.87, 0.5])
    kernel = InterlockKernel(
        InterlockPolicy(
            hard_limit=0.2,
            window_size=4,
            trend_window=1,
            trend_threshold=0.2,
            hook_id="trend.test",
            hook_scope="agent",
        )
    )

    result = kernel.run(
        ["a", "b", "c"],
        scorer=lambda text: next(scores),
        request_id="req-1",
        tenant_id="tenant-1",
    )

    assert result.decision == "halt"
    assert result.output == "ab"
    assert result.halt_index == 2
    assert result.halt_reason == "downward_trend"
    assert result.halt_event is not None
    assert result.halt_event.request_id == "req-1"
    assert result.halt_event.tenant_id == "tenant-1"
    assert result.halt_event.hook_scope == "agent"


def test_interlock_window_average_can_warn_without_halting() -> None:
    scores = iter([0.9, 0.42, 0.43])
    kernel = InterlockKernel(
        InterlockPolicy(
            hard_limit=0.2,
            window_size=2,
            window_threshold=0.5,
            warn_only=True,
        )
    )

    result = kernel.run(["a", "b", "c"], scorer=lambda text: next(scores))

    assert result.decision == "warn"
    assert result.output == "abc"
    assert result.halt_reason == "window_average"
    assert result.halt_event is not None
    assert result.halt_event.policy_decision == "warn"
    assert result.halt_event.attributes["warn_only"] == "true"


def test_interlock_warn_only_hard_limit_appends_flagged_token() -> None:
    scores = iter([0.9, 0.2])
    kernel = InterlockKernel(
        InterlockPolicy(
            hard_limit=0.5,
            warn_only=True,
        )
    )

    result = kernel.run(["safe", " flagged"], scorer=lambda text: next(scores))

    assert result.decision == "warn"
    assert result.output == "safe flagged"
    assert result.halt_index == 1
    assert result.halt_reason == "hard_limit"
    assert result.evidence_refs == ("interlock://token/1",)


def test_interlock_rejects_invalid_scorer_output() -> None:
    kernel = InterlockKernel(InterlockPolicy())

    with pytest.raises(ValueError, match="score"):
        kernel.run(["x"], scorer=lambda text: 1.5)

    with pytest.raises(ValueError, match="score"):
        kernel.run(["x"], scorer=lambda text: float("inf"))


def test_interlock_uses_score_attribute_results() -> None:
    class ScoreResult:
        score = 0.82

    kernel = InterlockKernel(InterlockPolicy(hard_limit=0.5))

    result = kernel.run(["x"], scorer=lambda text: ScoreResult())

    assert result.decision == "allow"
    assert result.scores == (0.82,)


def test_root_lazy_import_export() -> None:
    from director_ai import InterlockKernel as RootInterlockKernel

    assert RootInterlockKernel is InterlockKernel
