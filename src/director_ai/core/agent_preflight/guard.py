# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Agent preflight guard

"""Five preflight gates for an agent / MCP loop, each tied to evidence + policy.

An agent that can call tools, hand off to other agents, and take real-world
actions needs a guard at the seams, not just on the final answer.
:class:`AgentPreflightGuard` provides one decision per seam:

* :meth:`before_tool_call` — the call must name a known tool with valid
  arguments, be allowed by policy, and rest on present evidence.
* :meth:`after_tool_result` — the claimed result must not be fabricated or
  implausible against the manifest.
* :meth:`before_final_answer` — the answer must be evidence-backed and must not
  have tripped a canary.
* :meth:`before_handoff` — the target agent must be permitted and the payload
  safe.
* :meth:`before_irreversible_action` — an irreversible action needs a safeguard
  (a registered rollback or a human acknowledgement) before it proceeds.

It composes the existing tool-call verifier and reversibility estimator and is
dependency-injected, so it carries no opinion about a particular model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..irreversibility.reversibility import ReversibilityEstimator, RuleReversibility
from ..metrics import metrics
from ..verification.tool_call_verifier import verify_tool_call
from .decision import PreflightDecision
from .policy import PreflightPolicy

__all__ = ["AgentPreflightGuard"]

_PREFLIGHT_DECISIONS = "agent_preflight_decisions_total"

_BEFORE_TOOL = "before_tool_call"
_AFTER_TOOL = "after_tool_result"
_BEFORE_ANSWER = "before_final_answer"
_BEFORE_HANDOFF = "before_handoff"
_BEFORE_IRREVERSIBLE = "before_irreversible_action"


class AgentPreflightGuard:
    """Evidence- and policy-tied gates for the seams of an agent loop.

    Parameters
    ----------
    policy:
        The :class:`PreflightPolicy`; defaults to the fail-closed defaults.
    reversibility:
        Estimator for the irreversible-action gate; defaults to
        :class:`RuleReversibility`.
    score_fn:
        Optional ``score_fn(premise, hypothesis) -> float`` passed to the
        tool-call verifier to score result plausibility after a call.
    """

    def __init__(
        self,
        policy: PreflightPolicy | None = None,
        *,
        reversibility: ReversibilityEstimator | None = None,
        score_fn: Callable[[str, str], float] | None = None,
    ) -> None:
        self.policy = policy or PreflightPolicy()
        self._reversibility = reversibility or RuleReversibility()
        self._score_fn = score_fn

    def before_tool_call(
        self,
        function_name: str,
        arguments: dict[str, Any],
        *,
        manifest: dict[str, Any] | None = None,
        policy_allows: bool = True,
        evidence_ok: bool = True,
    ) -> PreflightDecision:
        """Gate a tool call before it executes."""
        if not policy_allows:
            return self._decide(_BEFORE_TOOL, "block", "policy_denied")
        if not evidence_ok:
            return self._decide(_BEFORE_TOOL, "block", "evidence_missing")
        if manifest is not None:
            result = verify_tool_call(function_name, arguments, manifest=manifest)
            if not result.function_exists:
                return self._decide(_BEFORE_TOOL, "block", "unknown_tool")
            if not result.arguments_valid:
                return self._decide(_BEFORE_TOOL, "block", "invalid_arguments")
        return self._decide(_BEFORE_TOOL, "allow", "ok")

    def after_tool_result(
        self,
        function_name: str,
        arguments: dict[str, Any],
        result: str,
        *,
        manifest: dict[str, Any] | None = None,
        execution_log: list[dict[str, Any]] | None = None,
    ) -> PreflightDecision:
        """Gate a tool result before the agent consumes it.

        With an ``execution_log`` the result is checked against what was actually
        run, catching a fabricated or mismatched result; with a ``score_fn`` and
        a manifest the result is also checked for plausibility against the tool's
        declared return.
        """
        verdict = verify_tool_call(
            function_name,
            arguments,
            claimed_result=result,
            manifest=manifest,
            execution_log=execution_log,
            score_fn=self._score_fn,
        )
        if verdict.fabrication_suspected:
            return self._decide(_AFTER_TOOL, "block", "fabricated_result")
        if not verdict.result_plausible:
            return self._decide(_AFTER_TOOL, "warn", "implausible_result")
        return self._decide(_AFTER_TOOL, "allow", "ok")

    def before_final_answer(
        self,
        *,
        evidence_ok: bool = True,
        canary_tripped: bool = False,
        evidence_refs: tuple[str, ...] = (),
    ) -> PreflightDecision:
        """Gate the final answer before it is returned to the user."""
        if self.policy.require_evidence_for_answer and not evidence_ok:
            return self._decide(_BEFORE_ANSWER, "block", "answer_unsupported")
        if self.policy.block_on_canary and canary_tripped:
            return self._decide(_BEFORE_ANSWER, "block", "canary_tripped")
        return self._decide(_BEFORE_ANSWER, "allow", "ok", evidence_refs=evidence_refs)

    def before_handoff(
        self,
        target_agent: str,
        *,
        payload_safe: bool = True,
    ) -> PreflightDecision:
        """Gate a cross-agent handoff before control transfers."""
        if not self.policy.handoff_allowed(target_agent):
            return self._decide(_BEFORE_HANDOFF, "block", "handoff_target_not_allowed")
        if not payload_safe:
            return self._decide(_BEFORE_HANDOFF, "block", "unsafe_handoff_payload")
        return self._decide(_BEFORE_HANDOFF, "allow", "ok")

    def before_irreversible_action(
        self,
        action: str,
        *,
        human_acknowledged: bool = False,
        rollback_registered: bool = False,
        context: Mapping[str, object] | None = None,
    ) -> PreflightDecision:
        """Gate a real-world action by its reversibility and safeguards.

        A reversible action is allowed. An irreversible one is allowed only with
        a registered rollback (which the trajectory rollback manager can arm) or
        a human acknowledgement; otherwise it is escalated for a human to decide.
        """
        score = self._reversibility.score(action, context=context)
        meta = {"reversibility": f"{score.score:.4f}"}
        if score.score >= self.policy.reversibility_threshold:
            return self._decide(
                _BEFORE_IRREVERSIBLE, "allow", "reversible", metadata=meta
            )
        if rollback_registered:
            return self._decide(
                _BEFORE_IRREVERSIBLE,
                "warn",
                "irreversible_rollback_armed",
                metadata=meta,
            )
        if human_acknowledged or not self.policy.require_human_ack_for_irreversible:
            return self._decide(
                _BEFORE_IRREVERSIBLE,
                "warn",
                "irreversible_acknowledged",
                metadata=meta,
            )
        return self._decide(
            _BEFORE_IRREVERSIBLE, "escalate", "irreversible_no_safeguard", metadata=meta
        )

    def _decide(
        self,
        hook: str,
        decision: str,
        reason: str,
        *,
        evidence_refs: tuple[str, ...] = (),
        metadata: dict[str, str] | None = None,
    ) -> PreflightDecision:
        metrics.inc_labeled(_PREFLIGHT_DECISIONS, {"hook": hook, "decision": decision})
        return PreflightDecision(
            hook=hook,
            decision=decision,
            reason=reason,
            evidence_refs=evidence_refs,
            metadata=metadata or {},
        )
