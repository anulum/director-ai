# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Intent-Grounded Prompt Injection Detection
"""Output-side prompt injection detection via bidirectional NLI.

Instead of pattern-matching known attacks in the input, this module
detects the *effect* of injection by measuring whether the LLM response
diverges from the original intent (system prompt + user query).

Two-stage pipeline:
  Stage 1 — InputSanitizer (regex/pattern, fast, catches encoding tricks)
  Stage 2 — InjectionDetector (NLI bidirectional, catches semantic injection)

Any successful injection MUST change the response away from the original
intent.  NLI measures this drift regardless of how the injection was
encoded.  Per-claim attribution provides explainability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..scoring.verified_scorer import _entity_overlap, _traceability
from ..types import InjectedClaim, InjectionResult

logger = logging.getLogger("DirectorAI.Injection")

_VERDICTS = frozenset({"grounded", "drifted", "injected"})


@dataclass
class _DetectorConfig:
    injection_threshold: float = 0.7
    drift_threshold: float = 0.6
    injection_claim_threshold: float = 0.75
    baseline_divergence: float = 0.4
    stage1_weight: float = 0.3
    traceability_floor: float = 0.15


class InjectionDetector:
    """Intent-grounded prompt injection detection via output-side NLI.

    Measures whether an LLM response diverges from the original intent
    (system prompt + user query).  Unlike pattern-matching approaches,
    detects the *effect* of injection rather than the injection itself.

    Parameters
    ----------
    nli_scorer : NLIScorer | None
        Existing NLI scorer instance (reuses loaded model).
    sanitizer : InputSanitizer | None
        Stage 1 pattern-based detector (optional).
    injection_threshold : float
        Combined score above which injection is flagged (default 0.7).
    drift_threshold : float
        Per-claim calibrated divergence above which = "drifted" (default 0.6).
    injection_claim_threshold : float
        Calibrated divergence + low traceability = "injected" (default 0.75).
    baseline_divergence : float
        Expected normal intent divergence for on-topic responses (default 0.4).
    stage1_weight : float
        Weight of InputSanitizer score in combined score (default 0.3).
    """

    def __init__(
        self,
        nli_scorer=None,
        sanitizer=None,
        injection_threshold: float = 0.7,
        drift_threshold: float = 0.6,
        injection_claim_threshold: float = 0.75,
        baseline_divergence: float = 0.4,
        stage1_weight: float = 0.3,
    ) -> None:
        self._nli = nli_scorer
        self._sanitizer = sanitizer
        self._cfg = _DetectorConfig(
            injection_threshold=injection_threshold,
            drift_threshold=drift_threshold,
            injection_claim_threshold=injection_claim_threshold,
            baseline_divergence=baseline_divergence,
            stage1_weight=stage1_weight,
        )

    # -- Public API -----------------------------------------------------------

    def detect(
        self,
        intent: str,
        response: str,
        user_query: str = "",
        system_prompt: str = "",
    ) -> InjectionResult:
        """Run the full two-stage injection detection pipeline.

        Parameters
        ----------
        intent : str
            Direct intent string.  Ignored if *system_prompt* or
            *user_query* are provided (they take precedence).
        response : str
            The LLM-generated response to analyse.
        user_query : str
            The user's original query (optional).
        system_prompt : str
            The system prompt / task description (optional).

        Returns
        -------
        InjectionResult
            Structured result with per-claim attribution.
        """
        effective_intent = self._build_intent(intent, user_query, system_prompt)

        # Stage 1: fast pattern-based check
        sanitizer_score = self._stage1_score(user_query or intent)

        # Short-circuit on empty response
        if not response or not response.strip():
            return InjectionResult(
                injection_detected=False,
                injection_risk=0.0,
                intent_coverage=1.0,
                total_claims=0,
                grounded_claims=0,
                drifted_claims=0,
                injected_claims=0,
                claims=[],
                input_sanitizer_score=sanitizer_score,
                combined_score=self._cfg.stage1_weight * sanitizer_score,
            )

        # Stage 2: NLI-based intent divergence
        claims_text = self._decompose(response)
        if not claims_text:
            return InjectionResult(
                injection_detected=False,
                injection_risk=0.0,
                intent_coverage=1.0,
                total_claims=0,
                grounded_claims=0,
                drifted_claims=0,
                injected_claims=0,
                claims=[],
                input_sanitizer_score=sanitizer_score,
                combined_score=self._cfg.stage1_weight * sanitizer_score,
            )

        scored_claims = self._score_claims_against_intent(claims_text, effective_intent)

        injection_risk, combined = self._compute_injection_risk(
            scored_claims, sanitizer_score
        )

        n_grounded = sum(1 for c in scored_claims if c.verdict == "grounded")
        n_drifted = sum(1 for c in scored_claims if c.verdict == "drifted")
        n_injected = sum(1 for c in scored_claims if c.verdict == "injected")
        total = len(scored_claims)
        coverage = n_grounded / total if total else 1.0

        return InjectionResult(
            injection_detected=combined >= self._cfg.injection_threshold,
            injection_risk=injection_risk,
            intent_coverage=coverage,
            total_claims=total,
            grounded_claims=n_grounded,
            drifted_claims=n_drifted,
            injected_claims=n_injected,
            claims=scored_claims,
            input_sanitizer_score=sanitizer_score,
            combined_score=combined,
        )

    # -- Intent construction --------------------------------------------------

    @staticmethod
    def _build_intent(intent: str, user_query: str, system_prompt: str) -> str:
        """Compose effective intent from available inputs.

        Priority: system_prompt + user_query > user_query > intent.
        """
        if system_prompt and user_query:
            return system_prompt + "\n\n" + user_query
        if system_prompt:
            return system_prompt
        if user_query:
            return user_query
        return intent

    # -- Stage 1: InputSanitizer ----------------------------------------------

    def _stage1_score(self, text: str) -> float:
        if self._sanitizer is None:
            return 0.0
        result = self._sanitizer.score(text)
        return float(result.suspicion_score)

    # -- Claim decomposition --------------------------------------------------

    def _decompose(self, response: str) -> list[str]:
        """Split response into atomic claims."""
        if self._nli is not None:
            return list(self._nli.decompose_claims(response))
        # Fallback: sentence splitting without NLI scorer
        return _fallback_split(response)

    # -- Bidirectional NLI scoring per claim ----------------------------------

    def _score_claims_against_intent(
        self,
        claims: list[str],
        intent: str,
    ) -> list[InjectedClaim]:
        fwd_scores, rev_scores = self._bidirectional_nli_batch(intent, claims)

        result: list[InjectedClaim] = []
        for i, claim in enumerate(claims):
            fwd = fwd_scores[i]
            rev = rev_scores[i]
            bidir = min(fwd, rev)

            trace = _traceability(claim, intent)
            entity = _entity_overlap(claim, intent)

            calibrated = self._baseline_calibrate(bidir)
            verdict, confidence = self._claim_verdict(calibrated, trace, entity)

            result.append(
                InjectedClaim(
                    claim=claim,
                    claim_index=i,
                    intent_divergence=fwd,
                    reverse_divergence=rev,
                    bidirectional_divergence=bidir,
                    traceability=trace,
                    entity_match=entity,
                    verdict=verdict,
                    confidence=confidence,
                )
            )
        return result

    def _bidirectional_nli_batch(
        self, intent: str, claims: list[str]
    ) -> tuple[list[float], list[float]]:
        """Two batched NLI passes: forward (intent→claim) and reverse."""
        if self._nli is None or not self._nli.model_available:
            # Heuristic fallback
            fwd = (
                [self._nli.score(intent, c) for c in claims]
                if self._nli
                else [0.5] * len(claims)
            )
            rev = (
                [self._nli.score(c, intent) for c in claims]
                if self._nli
                else [0.5] * len(claims)
            )
            return fwd, rev

        fwd_pairs = [(intent, c) for c in claims]
        rev_pairs = [(c, intent) for c in claims]
        fwd_scores = self._nli.score_batch(fwd_pairs)
        rev_scores = self._nli.score_batch(rev_pairs)
        return fwd_scores, rev_scores

    # -- Baseline calibration -------------------------------------------------

    def _baseline_calibrate(self, raw_divergence: float) -> float:
        """Shift raw divergence by baseline to suppress on-topic noise.

        Matches the dialogue/summarisation calibration pattern in scorer.py.
        """
        bl = self._cfg.baseline_divergence
        if bl >= 1.0:
            return 0.0
        return max(0.0, (raw_divergence - bl) / (1.0 - bl))

    # -- Multi-signal verdict -------------------------------------------------

    def _claim_verdict(
        self,
        calibrated_div: float,
        traceability: float,
        entity_match: float,
    ) -> tuple[str, float]:
        """Determine verdict for a single claim.

        Returns (verdict, confidence) where verdict is one of
        "grounded", "drifted", "injected".
        """
        cfg = self._cfg

        # Fabrication override: content entirely absent from intent
        if traceability < cfg.traceability_floor:
            confidence = min(
                1.0,
                (cfg.traceability_floor - traceability) / cfg.traceability_floor + 0.5,
            )
            return "injected", confidence

        if calibrated_div >= cfg.injection_claim_threshold and traceability < 0.2:
            signals_agree = 1.0 if entity_match < 0.3 else 0.6
            return "injected", signals_agree

        if calibrated_div >= cfg.drift_threshold:
            if traceability >= 0.3:
                return "drifted", min(1.0, calibrated_div)
            signals_agree = 1.0 if entity_match < 0.3 else 0.7
            return "injected", signals_agree

        return "grounded", min(1.0, 1.0 - calibrated_div)

    # -- Aggregation ----------------------------------------------------------

    def _compute_injection_risk(
        self,
        claims: list[InjectedClaim],
        sanitizer_score: float,
    ) -> tuple[float, float]:
        """Aggregate per-claim verdicts into overall risk.

        Returns (injection_risk, combined_score).
        """
        if not claims:
            return 0.0, self._cfg.stage1_weight * sanitizer_score

        total = len(claims)
        weighted = sum(
            1.0 if c.verdict == "injected" else 0.4 if c.verdict == "drifted" else 0.0
            for c in claims
        )
        injection_risk = weighted / total

        w = self._cfg.stage1_weight
        combined = w * sanitizer_score + (1.0 - w) * injection_risk
        return injection_risk, combined


# -- Module-level helpers -----------------------------------------------------


def _fallback_split(text: str) -> list[str]:
    """Sentence splitting without NLI scorer — period-based fallback."""
    sentences = [s.strip() + "." for s in text.split(".") if s.strip()]
    return sentences if sentences else [text.strip()] if text.strip() else []
