# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the agent / MCP preflight guard.

Covers decision and policy validation and all five hook points
(before/after tool call, before final answer, before handoff, before
irreversible action), plus metrics and the ProductionGuard integration.
"""

from __future__ import annotations

import pytest

from director_ai.core.agent_preflight import (
    AgentPreflightGuard,
    PreflightDecision,
    PreflightPolicy,
)
from director_ai.core.metrics import metrics

_MANIFEST = {
    "pay_invoice": {
        "description": "Pay an invoice",
        "parameters": {"invoice_id": {"type": "string", "required": True}},
        "returns": "a payment confirmation id",
    }
}


# ── PreflightDecision ───────────────────────────────────────────────────


class TestDecision:
    def test_bad_decision_rejected(self):
        with pytest.raises(ValueError, match="unsupported decision"):
            PreflightDecision(hook="h", decision="maybe", reason="r")

    def test_empty_hook_rejected(self):
        with pytest.raises(ValueError, match="hook is required"):
            PreflightDecision(hook=" ", decision="allow", reason="r")

    def test_empty_reason_rejected(self):
        with pytest.raises(ValueError, match="reason is required"):
            PreflightDecision(hook="h", decision="allow", reason=" ")

    def test_allowed_and_blocked_properties(self):
        assert PreflightDecision("h", "allow", "ok").allowed
        assert not PreflightDecision("h", "allow", "ok").blocked
        assert PreflightDecision("h", "block", "x").blocked
        assert PreflightDecision("h", "escalate", "x").blocked
        assert not PreflightDecision("h", "warn", "x").allowed

    def test_metadata_stringified(self):
        decision = PreflightDecision("h", "allow", "ok", metadata={"k": 1})
        assert decision.metadata == {"k": "1"}

    def test_to_dict(self):
        payload = PreflightDecision(
            "h", "block", "unknown_tool", evidence_refs=("e1",)
        ).to_dict()
        assert payload["decision"] == "block"
        assert payload["evidence_refs"] == ["e1"]

    def test_allow_factory(self):
        decision = PreflightDecision.allow("h", evidence_refs=("e1",))
        assert decision.allowed
        assert decision.evidence_refs == ("e1",)


# ── PreflightPolicy ─────────────────────────────────────────────────────


class TestPolicy:
    def test_threshold_range(self):
        with pytest.raises(ValueError, match="reversibility_threshold"):
            PreflightPolicy(reversibility_threshold=1.5)

    def test_handoff_targets_normalised(self):
        policy = PreflightPolicy(allowed_handoff_targets=frozenset({" billing ", ""}))
        assert policy.allowed_handoff_targets == frozenset({"billing"})

    def test_handoff_unrestricted_when_empty(self):
        assert PreflightPolicy().handoff_allowed("anything") is True

    def test_handoff_restricted(self):
        policy = PreflightPolicy(allowed_handoff_targets=frozenset({"billing"}))
        assert policy.handoff_allowed("billing")
        assert not policy.handoff_allowed("attacker")


# ── before_tool_call ────────────────────────────────────────────────────


class TestBeforeToolCall:
    def test_policy_denied(self):
        decision = AgentPreflightGuard().before_tool_call(
            "pay_invoice", {}, policy_allows=False
        )
        assert decision.reason == "policy_denied"
        assert decision.blocked

    def test_evidence_missing(self):
        decision = AgentPreflightGuard().before_tool_call(
            "pay_invoice", {}, evidence_ok=False
        )
        assert decision.reason == "evidence_missing"

    def test_unknown_tool(self):
        decision = AgentPreflightGuard().before_tool_call(
            "wire_funds", {"invoice_id": "x"}, manifest=_MANIFEST
        )
        assert decision.reason == "unknown_tool"

    def test_invalid_arguments(self):
        decision = AgentPreflightGuard().before_tool_call(
            "pay_invoice", {}, manifest=_MANIFEST
        )
        assert decision.reason == "invalid_arguments"

    def test_allow_with_valid_call(self):
        decision = AgentPreflightGuard().before_tool_call(
            "pay_invoice", {"invoice_id": "INV-1"}, manifest=_MANIFEST
        )
        assert decision.allowed

    def test_allow_without_manifest(self):
        decision = AgentPreflightGuard().before_tool_call("anything", {"a": 1})
        assert decision.allowed


# ── after_tool_result ───────────────────────────────────────────────────


class TestAfterToolResult:
    def test_fabricated_result(self):
        decision = AgentPreflightGuard().after_tool_result(
            "pay_invoice",
            {"invoice_id": "INV-1"},
            "PAID-999",
            execution_log=[{"function": "other", "arguments": {}, "result": "x"}],
        )
        assert decision.reason == "fabricated_result"
        assert decision.blocked

    def test_implausible_result(self):
        guard = AgentPreflightGuard(score_fn=lambda _p, _h: 0.9)
        decision = guard.after_tool_result(
            "pay_invoice",
            {"invoice_id": "INV-1"},
            "The moon is made of cheese.",
            manifest=_MANIFEST,
        )
        assert decision.decision == "warn"
        assert decision.reason == "implausible_result"

    def test_allow_plausible_result(self):
        guard = AgentPreflightGuard(score_fn=lambda _p, _h: 0.1)
        decision = guard.after_tool_result(
            "pay_invoice",
            {"invoice_id": "INV-1"},
            "PAID-123",
            manifest=_MANIFEST,
        )
        assert decision.allowed


# ── before_final_answer ─────────────────────────────────────────────────


class TestBeforeFinalAnswer:
    def test_unsupported_blocked(self):
        decision = AgentPreflightGuard().before_final_answer(evidence_ok=False)
        assert decision.reason == "answer_unsupported"

    def test_canary_blocked(self):
        decision = AgentPreflightGuard().before_final_answer(canary_tripped=True)
        assert decision.reason == "canary_tripped"

    def test_allow_with_evidence(self):
        decision = AgentPreflightGuard().before_final_answer(
            evidence_ok=True, evidence_refs=("vector:d1",)
        )
        assert decision.allowed
        assert decision.evidence_refs == ("vector:d1",)

    def test_evidence_requirement_can_be_disabled(self):
        guard = AgentPreflightGuard(PreflightPolicy(require_evidence_for_answer=False))
        assert guard.before_final_answer(evidence_ok=False).allowed

    def test_canary_block_can_be_disabled(self):
        guard = AgentPreflightGuard(PreflightPolicy(block_on_canary=False))
        assert guard.before_final_answer(canary_tripped=True).allowed


# ── before_handoff ──────────────────────────────────────────────────────


class TestBeforeHandoff:
    def test_target_not_allowed(self):
        guard = AgentPreflightGuard(
            PreflightPolicy(allowed_handoff_targets=frozenset({"billing"}))
        )
        assert guard.before_handoff("attacker").reason == "handoff_target_not_allowed"

    def test_unsafe_payload(self):
        decision = AgentPreflightGuard().before_handoff("billing", payload_safe=False)
        assert decision.reason == "unsafe_handoff_payload"

    def test_allow_unrestricted(self):
        assert AgentPreflightGuard().before_handoff("anyone").allowed

    def test_allow_listed_target(self):
        guard = AgentPreflightGuard(
            PreflightPolicy(allowed_handoff_targets=frozenset({"billing"}))
        )
        assert guard.before_handoff("billing").allowed


# ── before_irreversible_action ──────────────────────────────────────────


class TestBeforeIrreversibleAction:
    def test_reversible_allowed(self):
        decision = AgentPreflightGuard().before_irreversible_action(
            "draft an email for review"
        )
        assert decision.allowed
        assert "reversibility" in decision.metadata

    def test_irreversible_escalated(self):
        decision = AgentPreflightGuard().before_irreversible_action(
            "permanently delete the production database"
        )
        assert decision.decision == "escalate"
        assert decision.reason == "irreversible_no_safeguard"

    def test_irreversible_with_rollback_warns(self):
        decision = AgentPreflightGuard().before_irreversible_action(
            "permanently delete the production database",
            rollback_registered=True,
        )
        assert decision.decision == "warn"
        assert decision.reason == "irreversible_rollback_armed"

    def test_irreversible_with_human_ack_warns(self):
        decision = AgentPreflightGuard().before_irreversible_action(
            "permanently delete the production database",
            human_acknowledged=True,
        )
        assert decision.decision == "warn"
        assert decision.reason == "irreversible_acknowledged"

    def test_human_ack_not_required_policy(self):
        guard = AgentPreflightGuard(
            PreflightPolicy(require_human_ack_for_irreversible=False)
        )
        decision = guard.before_irreversible_action(
            "permanently delete the production database"
        )
        assert decision.decision == "warn"

    def test_injected_reversibility_estimator(self):
        class AlwaysReversible:
            def score(self, action, *, context=None):
                from director_ai.core.irreversibility.reversibility import (
                    ReversibilityScore,
                )

                return ReversibilityScore(score=1.0, reason="stub")

        guard = AgentPreflightGuard(reversibility=AlwaysReversible())
        assert guard.before_irreversible_action("delete everything").allowed


# ── metrics + integration ───────────────────────────────────────────────


class TestMetricsAndIntegration:
    def test_metrics_counted(self):
        metrics.reset()
        AgentPreflightGuard().before_handoff("anyone")
        snapshot = metrics.get_metrics()
        counter = snapshot["counters"]["agent_preflight_decisions_total"]
        assert (
            counter["multi_labels"].get('decision="allow",hook="before_handoff"') == 1.0
        )

    def test_guard_preflight_property(self):
        from director_ai.guard import ProductionGuard

        guard = ProductionGuard()
        preflight = guard.preflight
        assert isinstance(preflight, AgentPreflightGuard)
        # Cached: same instance on second access.
        assert guard.preflight is preflight
        decision = preflight.before_tool_call(
            "pay_invoice", {"invoice_id": "INV-1"}, manifest=_MANIFEST
        )
        assert decision.allowed
