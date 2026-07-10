# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Production Guard Response Defence
"""Input/output defence surface of the production guard.

:class:`ResponseDefenceMixin` carries the response-side defences of
:class:`~director_ai.guard.ProductionGuard`: prompt-injection detection,
content moderation, the combined :meth:`~ResponseDefenceMixin.firewall`
pass (whose verdict is the :class:`FirewallDecision` dataclass defined
here), agent tool-call verification, and streaming repair of unsupported
clauses. Detectors are built lazily on first use and persist on the
guard; the coherence check itself stays on the facade and is consumed
through the ``check`` contract.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from director_ai.core.license import enforce_capability_tier
from director_ai.core.types import CoherenceScore, InjectionResult

if TYPE_CHECKING:
    from director_ai.core import CoherenceScorer, GroundTruthStore
    from director_ai.core.config import DirectorConfig
    from director_ai.core.safety.injection import InjectionDetector
    from director_ai.core.streaming_repair import RepairResult
    from director_ai.core.verification.tool_call_verifier import ToolCallResult
    from director_ai.guard import GuardResult

logger = logging.getLogger("DirectorAI.Guard")

__all__ = ["FirewallDecision", "ResponseDefenceMixin"]


@dataclass(frozen=True)
class FirewallDecision:
    """Unified output of :meth:`ResponseDefenceMixin.firewall`.

    One pass over a response runs every enabled guard — hallucination
    (coherence), prompt injection, and content moderation (PII + toxicity) — and
    folds them into a single block/allow decision. ``blocked`` is ``True`` when
    any guard fires; ``reasons`` lists why, and the per-guard fields carry the
    detail for an audit trail.
    """

    blocked: bool
    approved: bool
    coherence: CoherenceScore
    injection_detected: bool
    injection_risk: float
    moderation_flags: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Tenant-safe summary (no raw response text)."""
        return {
            "blocked": self.blocked,
            "approved": self.approved,
            "coherence_score": round(self.coherence.score, 4),
            "injection_detected": self.injection_detected,
            "injection_risk": round(self.injection_risk, 4),
            "moderation_flags": list(self.moderation_flags),
            "reasons": list(self.reasons),
        }


class ResponseDefenceMixin:
    """Injection detection, moderation, firewall, tool verification, repair.

    All state is initialised by :class:`~director_ai.guard.ProductionGuard`'s
    ``__init__``; the scorer, configuration, knowledge base, and the coherence
    ``check`` service come from the composing guard through the contracts
    declared below.
    """

    _injection_detector: InjectionDetector | None
    _moderation_detectors: Any

    if TYPE_CHECKING:
        # Provided by the composing ProductionGuard.
        _scorer: CoherenceScorer
        _config: DirectorConfig
        _store: GroundTruthStore

        def check(self, prompt: str, response: str) -> GuardResult: ...

    def check_injection(
        self,
        intent: str,
        response: str,
        user_query: str = "",
        system_prompt: str = "",
    ) -> InjectionResult:
        """Detect prompt injection effects in a response via NLI divergence.

        Lazily initialises InjectionDetector on first call using config
        thresholds.  Reuses the scorer's NLI model when available.
        """
        if self._injection_detector is None:
            from director_ai.core.safety.injection import InjectionDetector

            nli = getattr(self._scorer, "_nli", None)
            cfg = self._config
            self._injection_detector = InjectionDetector(
                nli_scorer=nli,
                injection_threshold=cfg.injection_threshold,
                drift_threshold=cfg.injection_drift_threshold,
                injection_claim_threshold=cfg.injection_claim_threshold,
                baseline_divergence=cfg.injection_baseline_divergence,
                stage1_weight=cfg.injection_stage1_weight,
                require_model_backed_nli=getattr(
                    cfg,
                    "injection_require_model_backed_nli",
                    False,
                ),
            )
            logger.info(
                "Injection detector initialised (threshold=%.2f)",
                cfg.injection_threshold,
            )

        return self._injection_detector.detect(
            intent=intent,
            response=response,
            user_query=user_query,
            system_prompt=system_prompt,
        )

    def set_moderation_detectors(self, detectors: list[Any]) -> None:
        """Override the moderation detectors used by :meth:`firewall`.

        Each detector must implement ``analyse(text) -> ModerationResult``. The
        default is the dependency-free regex PII + keyword toxicity pair; swap in
        ``PresidioPIIDetector`` / ``DetoxifyDetector`` for stronger coverage.
        """
        self._moderation_detectors = list(detectors)

    def _ensure_moderation(self) -> list[Any]:
        """Lazily build the default dependency-free moderation detectors."""
        if self._moderation_detectors is None:
            from director_ai.core.safety.moderation import (
                KeywordToxicityDetector,
                RegexPIIDetector,
            )

            self._moderation_detectors = [
                RegexPIIDetector(),
                KeywordToxicityDetector(),
            ]
        detectors: list[Any] = list(self._moderation_detectors)
        return detectors

    def firewall(
        self,
        prompt: str,
        response: str,
        *,
        system_prompt: str = "",
        check_injection: bool = True,
        moderate: bool = True,
    ) -> FirewallDecision:
        """Run every enabled guard in one pass and fold them into one decision.

        Composes the hallucination guard (coherence), the prompt-injection
        detector, and the content-moderation detectors (PII + toxicity) over a
        single ``(prompt, response)`` pair. ``blocked`` is ``True`` when any
        guard fires; ``reasons`` explains which. Injection and moderation can be
        toggled off for latency-sensitive paths.
        """
        result = self.check(prompt, response)
        reasons: list[str] = []
        if not result.approved:
            reasons.append(
                f"hallucination: coherence {result.score:.3f} below threshold"
            )

        injection_detected = False
        injection_risk = 0.0
        if check_injection:
            inj = self.check_injection(
                intent=prompt,
                response=response,
                user_query=prompt,
                system_prompt=system_prompt,
            )
            injection_detected = inj.injection_detected
            injection_risk = inj.injection_risk
            if injection_detected:
                reasons.append(f"prompt injection: risk {injection_risk:.3f}")

        moderation_flags: list[str] = []
        if moderate:
            for detector in self._ensure_moderation():
                mod = detector.analyse(response)
                if mod.flagged:
                    moderation_flags.append(mod.detector)
                    reasons.append(
                        f"moderation: {mod.detector} flagged {len(mod.matches)} match(es)"
                    )

        blocked = bool(reasons)
        return FirewallDecision(
            blocked=blocked,
            approved=result.approved,
            coherence=result.coherence,
            injection_detected=injection_detected,
            injection_risk=injection_risk,
            moderation_flags=tuple(moderation_flags),
            reasons=tuple(reasons),
        )

    def verify_tool(
        self,
        function_name: str,
        arguments: dict[str, Any],
        claimed_result: str = "",
        manifest: dict[str, Any] | None = None,
        execution_log: list[dict[str, Any]] | None = None,
    ) -> ToolCallResult:
        """Verify an agent tool/function call against a manifest."""
        from director_ai.core.verification.tool_call_verifier import verify_tool_call

        return verify_tool_call(
            function_name=function_name,
            arguments=arguments,
            claimed_result=claimed_result,
            manifest=manifest,
            execution_log=execution_log,
        )

    def repair_stream(
        self,
        prompt: str,
        response: str,
        *,
        tenant_id: str = "",
        request_id: str = "",
        rewrite_fn: Callable[[str, list[str]], str] | None = None,
        threshold: float | None = None,
    ) -> RepairResult:
        """Repair unsupported clauses in a generated response.

        Turns a coherence halt into a corrective pass: each clause is scored
        against the knowledge base, and an unsupported clause is rewritten from
        retrieved corrective evidence (when ``rewrite_fn`` is supplied and
        evidence is found) or redacted, leaving the supported clauses intact.
        Returns a :class:`RepairResult` with the corrected text, per-clause
        actions, and a tenant-safe repair event per fix.
        """
        enforce_capability_tier("repair_stream")
        try:
            from director_ai.core.streaming_repair import StreamingRepairer
        except ModuleNotFoundError as exc:  # pragma: no cover - advanced tier only
            raise RuntimeError(
                "repair_stream requires the advanced tier "
                "(director_ai.core.streaming_repair is not installed)."
            ) from exc

        def _score_clause(clause: str) -> float:
            return self._scorer.review(prompt, clause, tenant_id=tenant_id)[1].score

        def _retrieve(clause: str) -> list[Any]:
            getter = getattr(self._store, "retrieve_context_with_chunks", None)
            if getter is None:
                return []
            return list(getter(clause, tenant_id=tenant_id))

        repairer = StreamingRepairer(
            _score_clause,
            threshold=(
                threshold if threshold is not None else self._config.coherence_threshold
            ),
            retrieve_fn=_retrieve,
            rewrite_fn=rewrite_fn,
        )
        return repairer.repair(response, tenant_id=tenant_id, request_id=request_id)
