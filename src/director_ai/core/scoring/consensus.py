# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Cross-model consensus scoring — multi-model factual agreement.

Queries the same prompt to multiple models, then scores pairwise
factual agreement via NLI. High disagreement → low confidence.

Usage::

    scorer = ConsensusScorer(
        models=["gpt-4o", "claude-3.5-sonnet", "gemini-1.5-pro"],
        generate_fn=my_generate_function,
    )
    result = scorer.score("What is the capital of France?")
    print(result.agreement_score)  # 0.95 = high consensus
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from director_ai.core.guard_control import (
    GuardDecision,
    NoGoPolicy,
    RiskEnvelope,
    VerifierSignal,
)

__all__ = [
    "BFTConsensusResult",
    "BFTConsensusVote",
    "ByzantineFaultTolerantConsensus",
    "ConsensusScorer",
    "ConsensusResult",
    "CrossVerifierConsensus",
    "CriticalConsensusProfile",
    "ModelResponse",
    "PairwiseAgreement",
]


@dataclass
class ModelResponse:
    """Response from a single model."""

    model: str
    response: str


@dataclass
class PairwiseAgreement:
    """Agreement between two model responses."""

    model_a: str
    model_b: str
    divergence: float  # 0 = agree, 1 = contradict
    agreed: bool


@dataclass
class ConsensusResult:
    """Result of cross-model consensus check."""

    responses: list[ModelResponse]
    pairs: list[PairwiseAgreement] = field(default_factory=list)
    agreement_score: float = 1.0  # 0 = complete disagreement, 1 = consensus
    lowest_pair_agreement: float = 1.0
    disagreement_pairs: list[PairwiseAgreement] = field(default_factory=list)
    num_models: int = 0

    @property
    def has_consensus(self) -> bool:
        return self.agreement_score > 0.7


@dataclass(frozen=True)
class BFTConsensusVote:
    """One independent verifier vote for BFT consensus."""

    verifier: str
    verdict: str
    risk_score: float
    evidence_ref: str = ""

    def __post_init__(self) -> None:
        if not self.verifier.strip():
            raise ValueError("verifier must be non-empty")
        if self.verdict not in {"allow", "warn", "halt"}:
            raise ValueError("verdict must be one of allow, warn, halt")
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError("risk_score must be in [0, 1]")


@dataclass(frozen=True)
class BFTConsensusResult:
    """PBFT-style quorum result for verifier votes."""

    decision: str
    reason: str
    policy_id: str
    fault_tolerance: int
    required_replicas: int
    quorum_size: int
    byzantine_resilient: bool
    participating_verifiers: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    risk_score: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "policy_id": self.policy_id,
            "fault_tolerance": self.fault_tolerance,
            "required_replicas": self.required_replicas,
            "quorum_size": self.quorum_size,
            "byzantine_resilient": self.byzantine_resilient,
            "participating_verifiers": self.participating_verifiers,
            "evidence_refs": tuple(_tenant_safe_ref(ref) for ref in self.evidence_refs),
            "risk_score": self.risk_score,
        }


class ByzantineFaultTolerantConsensus:
    """PBFT-style quorum over independent verifier votes.

    For fault tolerance ``f``, at least ``3f + 1`` independent votes are
    required and a decision requires a ``2f + 1`` quorum for the same verdict.
    """

    def __init__(self, *, fault_tolerance: int) -> None:
        if fault_tolerance < 0:
            raise ValueError("fault_tolerance must be non-negative")
        self.fault_tolerance = fault_tolerance
        self.required_replicas = 3 * fault_tolerance + 1
        self.quorum_size = 2 * fault_tolerance + 1

    def decide(
        self,
        votes: Sequence[BFTConsensusVote],
        *,
        policy_id: str,
    ) -> BFTConsensusResult:
        vote_tuple = tuple(votes)
        verifiers = [vote.verifier for vote in vote_tuple]
        if len(verifiers) != len(set(verifiers)):
            raise ValueError("duplicate verifier votes are not allowed")
        if len(vote_tuple) < self.required_replicas:
            return BFTConsensusResult(
                decision="warn",
                reason="bft_insufficient_replicas",
                policy_id=policy_id,
                fault_tolerance=self.fault_tolerance,
                required_replicas=self.required_replicas,
                quorum_size=self.quorum_size,
                byzantine_resilient=False,
            )
        by_verdict: dict[str, list[BFTConsensusVote]] = {
            "halt": [],
            "warn": [],
            "allow": [],
        }
        for vote in vote_tuple:
            by_verdict[vote.verdict].append(vote)
        for verdict in ("halt", "warn", "allow"):
            bucket = by_verdict[verdict]
            if len(bucket) >= self.quorum_size:
                quorum = tuple(bucket[: self.quorum_size])
                return BFTConsensusResult(
                    decision=verdict,
                    reason="bft_quorum",
                    policy_id=policy_id,
                    fault_tolerance=self.fault_tolerance,
                    required_replicas=self.required_replicas,
                    quorum_size=self.quorum_size,
                    byzantine_resilient=True,
                    participating_verifiers=tuple(v.verifier for v in quorum),
                    evidence_refs=tuple(
                        v.evidence_ref for v in quorum if v.evidence_ref
                    ),
                    risk_score=max(v.risk_score for v in quorum),
                )
        return BFTConsensusResult(
            decision="warn",
            reason="bft_no_quorum",
            policy_id=policy_id,
            fault_tolerance=self.fault_tolerance,
            required_replicas=self.required_replicas,
            quorum_size=self.quorum_size,
            byzantine_resilient=False,
            risk_score=max((vote.risk_score for vote in vote_tuple), default=0.0),
        )


@dataclass(frozen=True)
class CriticalConsensusProfile:
    """Verifier coverage and calibration settings for critical-domain consensus."""

    required_verifiers: Sequence[str]
    weights: dict[str, float] = field(default_factory=dict)
    max_interval_width: float = 0.35

    def __post_init__(self) -> None:
        required = tuple(sorted({name.strip() for name in self.required_verifiers}))
        if not required:
            raise ValueError("required_verifiers are required")
        if self.max_interval_width < 0.0 or self.max_interval_width > 1.0:
            raise ValueError("max_interval_width must be in [0, 1]")
        for name, weight in self.weights.items():
            if not name.strip():
                raise ValueError("weight verifier names must be non-empty")
            if weight < 0.0:
                raise ValueError("weights must be non-negative")
        object.__setattr__(self, "required_verifiers", required)


class ConsensusScorer:
    """Score factual agreement across multiple LLM responses.

    Parameters
    ----------
    models : list[str]
        Model identifiers to query.
    generate_fn : callable
        Function(prompt: str, model: str) -> str. Generates a response.
    score_fn : callable | None
        Function(text_a: str, text_b: str) -> float (divergence 0-1).
        If None, uses Jaccard word overlap heuristic.
    agreement_threshold : float
        Divergence below which a pair is considered in agreement.
    """

    def __init__(
        self,
        models: list[str],
        generate_fn=None,
        score_fn=None,
        agreement_threshold: float = 0.5,
    ):
        if len(models) < 2:
            raise ValueError("Need at least 2 models for consensus scoring")
        self._models = models
        self._generate = generate_fn
        self._score_fn = score_fn or self._jaccard_divergence
        self._threshold = agreement_threshold

    def score(self, prompt: str) -> ConsensusResult:
        """Query all models and compute pairwise agreement.

        Parameters
        ----------
        prompt : str
            The prompt to send to all models.

        Returns
        -------
        ConsensusResult
            Pairwise agreement scores and overall consensus.
        """
        responses = self._gather_responses(prompt)
        return self.score_responses(responses)

    def score_responses(self, responses: list[ModelResponse]) -> ConsensusResult:
        """Score agreement across pre-generated responses.

        Useful when you already have responses from multiple models
        and want to check consensus without re-generating.
        """
        if len(responses) < 2:
            return ConsensusResult(
                responses=responses,
                agreement_score=1.0,
                num_models=len(responses),
            )

        pairs: list[PairwiseAgreement] = []
        disagreements: list[PairwiseAgreement] = []

        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                div = self._score_fn(responses[i].response, responses[j].response)
                agreed = div < self._threshold
                pa = PairwiseAgreement(
                    model_a=responses[i].model,
                    model_b=responses[j].model,
                    divergence=div,
                    agreed=agreed,
                )
                pairs.append(pa)
                if not agreed:
                    disagreements.append(pa)

        if pairs:
            avg_agreement = 1.0 - sum(p.divergence for p in pairs) / len(pairs)
            lowest = 1.0 - max(p.divergence for p in pairs)
        else:
            avg_agreement = 1.0
            lowest = 1.0

        return ConsensusResult(
            responses=responses,
            pairs=pairs,
            agreement_score=max(0.0, min(1.0, avg_agreement)),
            lowest_pair_agreement=max(0.0, min(1.0, lowest)),
            disagreement_pairs=disagreements,
            num_models=len(responses),
        )

    def _gather_responses(self, prompt: str) -> list[ModelResponse]:
        if self._generate is None:
            raise ValueError("generate_fn is required for score()")
        return [
            ModelResponse(model=m, response=self._generate(prompt, m))
            for m in self._models
        ]

    @staticmethod
    def _jaccard_divergence(a: str, b: str) -> float:
        wa = set(a.lower().split())
        wb = set(b.lower().split())
        if not wa or not wb:
            return 1.0
        return 1.0 - len(wa & wb) / len(wa | wb)


class CrossVerifierConsensus:
    """Aggregate verifier signals into one tenant-safe guard decision.

    ``conservative`` mode treats high-confidence unsafe signals as decisive.
    ``weighted`` mode computes a weighted average risk score, then still lets
    the no-go policy upgrade irreversible or threshold-exceeding actions.
    """

    def __init__(
        self,
        *,
        mode: str = "conservative",
        weights: dict[str, float] | None = None,
        contradiction_threshold: float = 0.8,
        warn_threshold: float = 0.45,
        no_go_policy: NoGoPolicy | None = None,
    ) -> None:
        if mode not in {"conservative", "weighted"}:
            raise ValueError("mode must be 'conservative' or 'weighted'")
        if not 0.0 <= contradiction_threshold <= 1.0:
            raise ValueError("contradiction_threshold must be in [0, 1]")
        if not 0.0 <= warn_threshold <= 1.0:
            raise ValueError("warn_threshold must be in [0, 1]")
        self._mode = mode
        self._weights = dict(weights or {})
        self._contradiction_threshold = contradiction_threshold
        self._warn_threshold = warn_threshold
        self._no_go = no_go_policy

    def decide(
        self,
        signals: list[VerifierSignal] | tuple[VerifierSignal, ...],
        *,
        risk_envelope: RiskEnvelope,
        policy_id: str,
    ) -> GuardDecision:
        """Return one decision for a set of verifier signals."""
        signal_tuple = tuple(signals)
        if not signal_tuple:
            return self._finalize(
                GuardDecision(
                    decision="warn",
                    risk_score=0.0,
                    confidence_low=0.0,
                    confidence_high=1.0,
                    policy_id=policy_id,
                    reason="insufficient_verifier_evidence",
                    tenant_safe_explanation=(
                        "No verifier evidence was available; human review is required."
                    ),
                    evidence_refs=(),
                    verifier_signals=(),
                    risk_envelope=risk_envelope,
                    attributes={"consensus_mode": self._mode},
                )
            )
        risk_score = (
            self._weighted_risk(signal_tuple)
            if self._mode == "weighted"
            else self._conservative_risk(signal_tuple)
        )
        confidence_low = min(signal.confidence_low for signal in signal_tuple)
        confidence_high = max(signal.confidence_high for signal in signal_tuple)
        evidence_refs = _collect_evidence_refs(signal_tuple)
        contradiction = self._decisive_contradiction(signal_tuple)
        if contradiction is not None:
            decision = "halt"
            reason = "cross_verifier_contradiction"
            explanation = "At least one verifier found a high-confidence conflict."
        elif risk_score >= risk_envelope.calibrated_threshold:
            decision = "halt"
            reason = "cross_verifier_risk_threshold"
            explanation = "Verifier consensus crossed the calibrated risk threshold."
        elif risk_score >= self._warn_threshold:
            decision = "warn"
            reason = "cross_verifier_uncertain"
            explanation = "Verifier consensus is uncertain and requires review."
        else:
            decision = "allow"
            reason = "cross_verifier_supported"
            explanation = "Verifier consensus did not identify blocking risk."
        return self._finalize(
            GuardDecision(
                decision=decision,
                risk_score=risk_score,
                confidence_low=confidence_low,
                confidence_high=confidence_high,
                policy_id=policy_id,
                reason=reason,
                tenant_safe_explanation=explanation,
                evidence_refs=evidence_refs,
                verifier_signals=signal_tuple,
                risk_envelope=risk_envelope,
                attributes={"consensus_mode": self._mode},
            )
        )

    def decide_critical(
        self,
        signals: Sequence[VerifierSignal],
        *,
        profile: CriticalConsensusProfile,
        risk_envelope: RiskEnvelope,
        policy_id: str,
    ) -> GuardDecision:
        """Return a profile-gated consensus decision for critical domains."""
        signal_tuple = tuple(signals)
        decision = self.decide(
            signal_tuple,
            risk_envelope=risk_envelope,
            policy_id=policy_id,
        )
        present = tuple(sorted({signal.verifier for signal in signal_tuple}))
        missing = tuple(
            verifier
            for verifier in profile.required_verifiers
            if verifier not in present
        )
        interval_low, interval_high = _fused_interval(signal_tuple, profile.weights)
        interval_width = interval_high - interval_low
        attributes = {
            **dict(decision.attributes),
            "consensus_profile": "critical",
            "required_verifiers": ",".join(profile.required_verifiers),
            "present_verifiers": ",".join(present),
            "missing_verifiers": ",".join(missing),
            "calibrated_interval_width": f"{interval_width:.6f}",
        }
        if missing and decision.decision == "allow":
            return GuardDecision(
                decision="warn",
                risk_score=decision.risk_score,
                confidence_low=interval_low,
                confidence_high=interval_high,
                policy_id=decision.policy_id,
                reason="critical_consensus_missing_verifier",
                tenant_safe_explanation=(
                    "Critical-domain consensus is missing required verifier coverage."
                ),
                evidence_refs=decision.evidence_refs,
                verifier_signals=decision.verifier_signals,
                risk_envelope=decision.risk_envelope,
                attributes=attributes,
            )
        if interval_width > profile.max_interval_width and decision.decision == "allow":
            return GuardDecision(
                decision="warn",
                risk_score=decision.risk_score,
                confidence_low=interval_low,
                confidence_high=interval_high,
                policy_id=decision.policy_id,
                reason="critical_consensus_interval_too_wide",
                tenant_safe_explanation=(
                    "Critical-domain consensus interval is too wide for release."
                ),
                evidence_refs=decision.evidence_refs,
                verifier_signals=decision.verifier_signals,
                risk_envelope=decision.risk_envelope,
                attributes=attributes,
            )
        return GuardDecision(
            decision=decision.decision,
            risk_score=_critical_risk_score(decision, signal_tuple, profile.weights),
            confidence_low=interval_low,
            confidence_high=interval_high,
            policy_id=decision.policy_id,
            reason=decision.reason,
            tenant_safe_explanation=decision.tenant_safe_explanation,
            evidence_refs=decision.evidence_refs,
            verifier_signals=decision.verifier_signals,
            risk_envelope=decision.risk_envelope,
            attributes=attributes,
        )

    def _finalize(self, decision: GuardDecision) -> GuardDecision:
        if self._no_go is None:
            return decision
        verdict = self._no_go.evaluate(decision)
        if verdict.decision == decision.decision and verdict.reason == decision.reason:
            return decision
        return GuardDecision(
            decision=verdict.decision,
            risk_score=decision.risk_score,
            confidence_low=decision.confidence_low,
            confidence_high=decision.confidence_high,
            policy_id=decision.policy_id,
            reason=verdict.reason,
            tenant_safe_explanation=(
                "No-go policy blocked this action; human review is required."
                if verdict.requires_human_review
                else decision.tenant_safe_explanation
            ),
            evidence_refs=decision.evidence_refs,
            verifier_signals=decision.verifier_signals,
            risk_envelope=decision.risk_envelope,
            attributes={
                **dict(decision.attributes),
                "requires_human_review": str(verdict.requires_human_review).lower(),
            },
        )

    def _conservative_risk(self, signals: tuple[VerifierSignal, ...]) -> float:
        return max(signal.score for signal in signals)

    def _weighted_risk(self, signals: tuple[VerifierSignal, ...]) -> float:
        weighted_sum = 0.0
        weight_total = 0.0
        for signal in signals:
            weight = self._weights.get(signal.verifier, 1.0)
            if weight < 0.0:
                raise ValueError("weights must be non-negative")
            weighted_sum += signal.score * weight
            weight_total += weight
        if weight_total <= 0.0:
            raise ValueError("at least one verifier weight must be positive")
        return max(0.0, min(1.0, weighted_sum / weight_total))

    def _decisive_contradiction(
        self,
        signals: tuple[VerifierSignal, ...],
    ) -> VerifierSignal | None:
        unsafe_verdicts = {
            "block",
            "blocked",
            "contradiction",
            "hallucinated",
            "unsafe",
        }
        for signal in signals:
            if (
                signal.verdict.lower() in unsafe_verdicts
                and signal.score >= self._contradiction_threshold
            ):
                return signal
        return None


def _collect_evidence_refs(signals: tuple[VerifierSignal, ...]) -> tuple[str, ...]:
    refs: list[str] = []
    seen: set[str] = set()
    for signal in signals:
        for ref in signal.evidence_refs:
            if ref not in seen:
                refs.append(ref)
                seen.add(ref)
    return tuple(refs)


def _fused_interval(
    signals: tuple[VerifierSignal, ...],
    weights: dict[str, float],
) -> tuple[float, float]:
    if not signals:
        return (0.0, 1.0)
    low = _weighted_value(signals, weights, "confidence_low")
    high = _weighted_value(signals, weights, "confidence_high")
    return (low, high)


def _weighted_value(
    signals: tuple[VerifierSignal, ...],
    weights: dict[str, float],
    field_name: str,
) -> float:
    if not signals:
        return 0.0
    weighted_sum = 0.0
    weight_total = 0.0
    for signal in signals:
        weight = weights.get(signal.verifier, 1.0)
        if weight < 0.0:
            raise ValueError("weights must be non-negative")
        weighted_sum += getattr(signal, field_name) * weight
        weight_total += weight
    if weight_total <= 0.0:
        raise ValueError("at least one verifier weight must be positive")
    return max(0.0, min(1.0, weighted_sum / weight_total))


def _critical_risk_score(
    decision: GuardDecision,
    signals: tuple[VerifierSignal, ...],
    weights: dict[str, float],
) -> float:
    weighted = _weighted_value(signals, weights, "score")
    if decision.decision in {"halt", "block"}:
        return max(decision.risk_score, weighted)
    return weighted


def _tenant_safe_ref(ref: str) -> str:
    return "redacted" if ref.startswith(("secret://", "raw://", "prompt://")) else ref
