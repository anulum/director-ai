# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for Phase 5 Gem 10: Cross-Model Consensus.

Covers: identical/different responses, multi-model, single response,
generate_fn, custom score_fn, validation guards, pairwise count,
parametrised model counts, pipeline integration, and performance.
"""

from __future__ import annotations

import pytest

from director_ai.core.guard_control import (
    NoGoPolicy,
    ReviewedIrreversibilityThreshold,
    RiskEnvelope,
    VerifierSignal,
)
from director_ai.core.irreversibility import Forecast
from director_ai.core.scoring.consensus import (
    BFTConsensusVote,
    ByzantineFaultTolerantConsensus,
    ConsensusScorer,
    CriticalConsensusProfile,
    CrossVerifierConsensus,
    ModelResponse,
    _fused_interval,
    _sum_float,
    _weighted_value,
    _word_overlap,
)


class TestConsensusScorer:
    def test_identical_responses_full_agreement(self):
        responses = [
            ModelResponse(model="a", response="The capital of France is Paris."),
            ModelResponse(model="b", response="The capital of France is Paris."),
        ]
        scorer = ConsensusScorer(models=["a", "b"])
        result = scorer.score_responses(responses)
        assert result.agreement_score == 1.0
        assert result.has_consensus
        assert len(result.disagreement_pairs) == 0

    def test_completely_different_responses(self):
        responses = [
            ModelResponse(model="a", response="AAAA BBBB CCCC DDDD"),
            ModelResponse(model="b", response="xxxx yyyy zzzz wwww"),
        ]
        scorer = ConsensusScorer(models=["a", "b"])
        result = scorer.score_responses(responses)
        assert result.agreement_score < 0.3
        assert not result.has_consensus

    def test_three_models_mixed(self):
        responses = [
            ModelResponse(model="a", response="Paris is the capital of France."),
            ModelResponse(model="b", response="The capital of France is Paris."),
            ModelResponse(model="c", response="Tokyo is the capital of Japan."),
        ]
        scorer = ConsensusScorer(models=["a", "b", "c"])
        result = scorer.score_responses(responses)
        assert result.num_models == 3
        assert len(result.pairs) == 3  # C(3,2) = 3

    def test_single_response(self):
        responses = [ModelResponse(model="a", response="test")]
        scorer = ConsensusScorer(models=["a", "b"])
        result = scorer.score_responses(responses)
        assert result.agreement_score == 1.0
        assert result.num_models == 1


class TestConsensusWithGenerate:
    def test_generate_fn(self):
        def mock_gen(prompt, model):
            return f"{model} says: the answer is 42"

        scorer = ConsensusScorer(
            models=["a", "b"],
            generate_fn=mock_gen,
        )
        result = scorer.score("What is the answer?")
        assert result.num_models == 2
        assert result.agreement_score > 0.5

    def test_no_generate_fn_raises(self):
        scorer = ConsensusScorer(models=["a", "b"])
        with pytest.raises(ValueError, match="generate_fn"):
            scorer.score("test")


class TestConsensusCustomScorer:
    def test_custom_score_fn(self):
        def always_agree(a, b):
            return 0.0

        responses = [
            ModelResponse(model="a", response="anything"),
            ModelResponse(model="b", response="completely different"),
        ]
        scorer = ConsensusScorer(models=["a", "b"], score_fn=always_agree)
        result = scorer.score_responses(responses)
        assert result.agreement_score == 1.0

    def test_custom_score_fn_disagree(self):
        def always_disagree(a, b):
            return 1.0

        responses = [
            ModelResponse(model="a", response="same text"),
            ModelResponse(model="b", response="same text"),
        ]
        scorer = ConsensusScorer(models=["a", "b"], score_fn=always_disagree)
        result = scorer.score_responses(responses)
        assert result.agreement_score == 0.0


class TestValidation:
    def test_needs_two_models(self):
        with pytest.raises(ValueError, match="at least 2"):
            ConsensusScorer(models=["only_one"])

    def test_pairwise_count(self):
        responses = [
            ModelResponse(model="a", response="x"),
            ModelResponse(model="b", response="x"),
            ModelResponse(model="c", response="x"),
            ModelResponse(model="d", response="x"),
        ]
        scorer = ConsensusScorer(models=["a", "b", "c", "d"])
        result = scorer.score_responses(responses)
        assert len(result.pairs) == 6  # C(4,2)


class TestConsensusParametrised:
    """Parametrised consensus tests."""

    @pytest.mark.parametrize("n_models", [2, 3, 4, 5])
    def test_pairwise_formula(self, n_models):
        models = [f"m{i}" for i in range(n_models)]
        responses = [ModelResponse(model=m, response="same text") for m in models]
        scorer = ConsensusScorer(models=models)
        result = scorer.score_responses(responses)
        expected_pairs = n_models * (n_models - 1) // 2
        assert len(result.pairs) == expected_pairs

    @pytest.mark.parametrize(
        "agreement,expected_consensus",
        [(1.0, True), (0.9, True), (0.5, False), (0.0, False)],
    )
    def test_consensus_thresholds(self, agreement, expected_consensus):
        def score_fn(a, b):
            return 1.0 - agreement

        responses = [
            ModelResponse(model="a", response="x"),
            ModelResponse(model="b", response="y"),
        ]
        scorer = ConsensusScorer(models=["a", "b"], score_fn=score_fn)
        result = scorer.score_responses(responses)
        assert result.has_consensus == expected_consensus


class TestConsensusPerformanceDoc:
    """Document consensus pipeline performance."""

    def test_result_has_all_fields(self):
        responses = [
            ModelResponse(model="a", response="test"),
            ModelResponse(model="b", response="test"),
        ]
        scorer = ConsensusScorer(models=["a", "b"])
        result = scorer.score_responses(responses)
        assert hasattr(result, "agreement_score")
        assert hasattr(result, "has_consensus")
        assert hasattr(result, "num_models")
        assert hasattr(result, "pairs")
        assert hasattr(result, "disagreement_pairs")

    def test_scoring_fast(self):
        import time

        responses = [
            ModelResponse(model=f"m{i}", response=f"Response from model {i}")
            for i in range(5)
        ]
        scorer = ConsensusScorer(models=[f"m{i}" for i in range(5)])
        t0 = time.perf_counter()
        scorer.score_responses(responses)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 1000, f"Consensus scoring took {elapsed_ms:.0f}ms"


class TestConsensusRustDelegation:
    # _word_overlap now delegates to the shared director_ai.core.text_overlap
    # helper; the Python/Rust dispatch and mandatory-failure behaviour are covered
    # by test_text_overlap. These pin the values consensus scoring relies on.
    def test_word_overlap_values(self):
        assert _word_overlap("alpha beta", "alpha gamma") == pytest.approx(1.0 / 3.0)
        assert _word_overlap("", "alpha gamma") == 0.0
        assert _word_overlap("alpha beta", "alpha beta") == 1.0

    def test_jaccard_divergence_uses_overlap(self):
        assert ConsensusScorer._jaccard_divergence("x y", "x y") == pytest.approx(0.0)
        assert ConsensusScorer._jaccard_divergence("x", "y") == pytest.approx(1.0)

    def test_sum_float_delegates_to_rust_and_python_paths(self, monkeypatch):
        from director_ai.core.scoring import consensus as consensus_mod

        monkeypatch.setattr(consensus_mod, "_RUST_CONSENSUS", True)
        calls = []

        def _sum(values):
            calls.append(list(values))
            return 2.5

        monkeypatch.setattr(consensus_mod, "rust_sum_f64", _sum, raising=False)
        assert _sum_float([1.0, 1.5]) == pytest.approx(2.5)
        assert calls == [[1.0, 1.5]]

        monkeypatch.setattr(consensus_mod, "_RUST_CONSENSUS", False)
        assert _sum_float([1.0, 1.5]) == pytest.approx(2.5)

    def test_sum_float_exception_is_mandatory_failure(self, monkeypatch):
        from director_ai.core.scoring import consensus as consensus_mod

        monkeypatch.setattr(consensus_mod, "_RUST_CONSENSUS", True)

        def _boom(values):
            raise RuntimeError("ffi fail")

        monkeypatch.setattr(consensus_mod, "rust_sum_f64", _boom, raising=False)
        with pytest.raises(RuntimeError, match="ffi fail"):
            _sum_float([1.0, 2.0])


class TestByzantineFaultTolerantConsensus:
    def test_vote_validation_rejects_invalid_inputs(self):
        with pytest.raises(ValueError, match="verifier must be non-empty"):
            BFTConsensusVote(verifier=" ", verdict="allow", risk_score=0.1)
        with pytest.raises(ValueError, match="verdict must be one of"):
            BFTConsensusVote(verifier="nli", verdict="block", risk_score=0.1)
        with pytest.raises(ValueError, match=r"risk_score must be in \[0, 1\]"):
            BFTConsensusVote(verifier="nli", verdict="allow", risk_score=1.1)

    def test_fault_tolerance_validation_and_insufficient_replicas(self):
        with pytest.raises(ValueError, match="fault_tolerance must be non-negative"):
            ByzantineFaultTolerantConsensus(fault_tolerance=-1)

        result = ByzantineFaultTolerantConsensus(fault_tolerance=1).decide(
            (
                BFTConsensusVote("nli", "allow", 0.1),
                BFTConsensusVote("policy", "allow", 0.2),
                BFTConsensusVote("numeric", "allow", 0.3),
            ),
            policy_id="policy.bft",
        )

        assert result.decision == "warn"
        assert result.reason == "bft_insufficient_replicas"
        assert result.required_replicas == 4
        assert result.quorum_size == 3
        assert not result.byzantine_resilient

    def test_quorum_prefers_halt_and_redacts_unsafe_evidence_refs(self):
        result = ByzantineFaultTolerantConsensus(fault_tolerance=1).decide(
            (
                BFTConsensusVote("nli", "halt", 0.8, "secret://raw-claim"),
                BFTConsensusVote("policy", "halt", 0.7, "raw://frame"),
                BFTConsensusVote("symbolic", "halt", 0.6, "proof://claim"),
                BFTConsensusVote("numeric", "allow", 0.1, "calc://claim"),
            ),
            policy_id="policy.bft",
        )

        assert result.decision == "halt"
        assert result.reason == "bft_quorum"
        assert result.participating_verifiers == ("nli", "policy", "symbolic")
        assert result.risk_score == pytest.approx(0.8)
        assert result.to_dict()["evidence_refs"] == (
            "redacted",
            "redacted",
            "proof://claim",
        )

    def test_no_quorum_and_duplicate_verifiers_are_rejected(self):
        consensus = ByzantineFaultTolerantConsensus(fault_tolerance=1)
        with pytest.raises(ValueError, match="duplicate verifier votes"):
            consensus.decide(
                (
                    BFTConsensusVote("nli", "allow", 0.1),
                    BFTConsensusVote("nli", "warn", 0.4),
                    BFTConsensusVote("policy", "halt", 0.9),
                    BFTConsensusVote("numeric", "allow", 0.2),
                ),
                policy_id="policy.bft",
            )

        result = consensus.decide(
            (
                BFTConsensusVote("nli", "allow", 0.1),
                BFTConsensusVote("policy", "warn", 0.4),
                BFTConsensusVote("numeric", "halt", 0.9),
                BFTConsensusVote("symbolic", "allow", 0.2),
            ),
            policy_id="policy.bft",
        )

        assert result.decision == "warn"
        assert result.reason == "bft_no_quorum"
        assert result.risk_score == pytest.approx(0.9)


class TestCrossVerifierConsensus:
    def test_profile_validation_rejects_empty_bad_interval_and_weights(self):
        with pytest.raises(ValueError, match="required_verifiers are required"):
            CriticalConsensusProfile(required_verifiers=())
        with pytest.raises(ValueError, match=r"max_interval_width must be in \[0, 1\]"):
            CriticalConsensusProfile(
                required_verifiers=("nli",),
                max_interval_width=1.5,
            )
        with pytest.raises(ValueError, match="weight verifier names must be non-empty"):
            CriticalConsensusProfile(
                required_verifiers=("nli",),
                weights={" ": 1.0},
            )
        with pytest.raises(ValueError, match="weights must be non-negative"):
            CriticalConsensusProfile(
                required_verifiers=("nli",),
                weights={"nli": -1.0},
            )

    def test_constructor_and_weight_validation_rejects_invalid_thresholds(self):
        with pytest.raises(ValueError, match="mode must be"):
            CrossVerifierConsensus(mode="average")
        with pytest.raises(
            ValueError, match=r"contradiction_threshold must be in \[0, 1\]"
        ):
            CrossVerifierConsensus(contradiction_threshold=1.1)
        with pytest.raises(ValueError, match=r"warn_threshold must be in \[0, 1\]"):
            CrossVerifierConsensus(warn_threshold=-0.1)

        consensus = CrossVerifierConsensus(
            mode="weighted",
            weights={"nli": -1.0},
        )
        with pytest.raises(ValueError, match="weights must be non-negative"):
            consensus.decide(
                (
                    VerifierSignal(
                        verifier="nli",
                        modality="text",
                        score=0.1,
                        verdict="supported",
                        confidence_low=0.0,
                        confidence_high=0.2,
                    ),
                ),
                risk_envelope=RiskEnvelope(
                    action_category="text",
                    reversibility="reversible",
                    domain="regulated",
                    calibrated_threshold=0.65,
                    no_go_threshold=0.9,
                ),
                policy_id="policy.weighted",
            )

    def test_weighted_consensus_rejects_zero_total_weight(self):
        consensus = CrossVerifierConsensus(
            mode="weighted",
            weights={"nli": 0.0},
        )

        with pytest.raises(ValueError, match="at least one verifier weight"):
            consensus.decide(
                (
                    VerifierSignal(
                        verifier="nli",
                        modality="text",
                        score=0.1,
                        verdict="supported",
                        confidence_low=0.0,
                        confidence_high=0.2,
                    ),
                ),
                risk_envelope=RiskEnvelope(
                    action_category="text",
                    reversibility="reversible",
                    domain="regulated",
                    calibrated_threshold=0.65,
                    no_go_threshold=0.9,
                ),
                policy_id="policy.weighted",
            )

    def test_critical_consensus_combines_required_verifiers_into_interval(self):
        envelope = RiskEnvelope(
            action_category="text",
            reversibility="reversible",
            domain="regulated",
            calibrated_threshold=0.65,
            no_go_threshold=0.9,
        )
        profile = CriticalConsensusProfile(
            required_verifiers=("nli", "policy", "temporal", "numeric", "symbolic"),
            weights={
                "nli": 2.0,
                "policy": 1.5,
                "temporal": 1.0,
                "numeric": 1.0,
                "symbolic": 1.0,
            },
            max_interval_width=0.35,
        )

        decision = CrossVerifierConsensus().decide_critical(
            (
                VerifierSignal(
                    verifier="nli",
                    modality="text",
                    score=0.12,
                    verdict="supported",
                    confidence_low=0.08,
                    confidence_high=0.18,
                    evidence_refs=("kb://claim",),
                ),
                VerifierSignal(
                    verifier="policy",
                    modality="policy",
                    score=0.18,
                    verdict="allowed",
                    confidence_low=0.12,
                    confidence_high=0.25,
                    evidence_refs=("policy://regulated",),
                ),
                VerifierSignal(
                    verifier="temporal",
                    modality="text",
                    score=0.21,
                    verdict="fresh",
                    confidence_low=0.14,
                    confidence_high=0.31,
                    evidence_refs=("time://observed",),
                ),
                VerifierSignal(
                    verifier="numeric",
                    modality="code",
                    score=0.09,
                    verdict="consistent",
                    confidence_low=0.05,
                    confidence_high=0.16,
                    evidence_refs=("calc://claim",),
                ),
                VerifierSignal(
                    verifier="symbolic",
                    modality="code",
                    score=0.16,
                    verdict="valid",
                    confidence_low=0.1,
                    confidence_high=0.22,
                    evidence_refs=("proof://claim",),
                ),
            ),
            profile=profile,
            risk_envelope=envelope,
            policy_id="policy.critical.regulated",
        )

        assert decision.decision == "allow"
        assert decision.reason == "cross_verifier_supported"
        assert decision.risk_score == pytest.approx(0.14923076923076922)
        assert decision.confidence_low == pytest.approx(0.09692307692307692)
        assert decision.confidence_high == pytest.approx(0.21923076923076923)
        assert decision.attributes["consensus_profile"] == "critical"
        assert decision.attributes["missing_verifiers"] == ""
        assert decision.attributes["present_verifiers"] == (
            "nli,numeric,policy,symbolic,temporal"
        )
        assert decision.attributes["calibrated_interval_width"] == "0.122308"

    def test_critical_consensus_warns_when_required_verifier_is_missing(self):
        envelope = RiskEnvelope(
            action_category="text",
            reversibility="reversible",
            domain="regulated",
            calibrated_threshold=0.65,
            no_go_threshold=0.9,
        )
        profile = CriticalConsensusProfile(
            required_verifiers=("nli", "policy", "temporal", "numeric", "symbolic"),
        )

        decision = CrossVerifierConsensus().decide_critical(
            (
                VerifierSignal(
                    verifier="nli",
                    modality="text",
                    score=0.12,
                    verdict="supported",
                    confidence_low=0.08,
                    confidence_high=0.18,
                ),
                VerifierSignal(
                    verifier="policy",
                    modality="policy",
                    score=0.1,
                    verdict="allowed",
                    confidence_low=0.06,
                    confidence_high=0.18,
                ),
            ),
            profile=profile,
            risk_envelope=envelope,
            policy_id="policy.critical.regulated",
        )

        assert decision.decision == "warn"
        assert decision.reason == "critical_consensus_missing_verifier"
        assert decision.attributes["missing_verifiers"] == "numeric,symbolic,temporal"

    def test_critical_consensus_warns_when_interval_is_too_wide(self):
        envelope = RiskEnvelope(
            action_category="text",
            reversibility="reversible",
            domain="regulated",
            calibrated_threshold=0.8,
            no_go_threshold=0.95,
        )
        profile = CriticalConsensusProfile(
            required_verifiers=("nli", "policy"),
            max_interval_width=0.1,
        )

        decision = CrossVerifierConsensus().decide_critical(
            (
                VerifierSignal(
                    verifier="nli",
                    modality="text",
                    score=0.1,
                    verdict="supported",
                    confidence_low=0.0,
                    confidence_high=0.4,
                ),
                VerifierSignal(
                    verifier="policy",
                    modality="policy",
                    score=0.1,
                    verdict="allowed",
                    confidence_low=0.0,
                    confidence_high=0.4,
                ),
            ),
            profile=profile,
            risk_envelope=envelope,
            policy_id="policy.critical.regulated",
        )

        assert decision.decision == "warn"
        assert decision.reason == "critical_consensus_interval_too_wide"
        assert decision.confidence_low == pytest.approx(0.0)
        assert decision.confidence_high == pytest.approx(0.4)

    def test_critical_consensus_preserves_decisive_contradiction_risk(self):
        envelope = RiskEnvelope(
            action_category="text",
            reversibility="reversible",
            domain="regulated",
            calibrated_threshold=0.65,
            no_go_threshold=0.9,
        )
        profile = CriticalConsensusProfile(
            required_verifiers=("nli", "policy", "temporal", "numeric", "symbolic"),
            weights={"nli": 0.1, "policy": 1.0, "temporal": 1.0, "numeric": 1.0},
        )

        decision = CrossVerifierConsensus().decide_critical(
            (
                VerifierSignal(
                    verifier="nli",
                    modality="text",
                    score=0.94,
                    verdict="contradiction",
                    confidence_low=0.9,
                    confidence_high=0.98,
                ),
                VerifierSignal(
                    verifier="policy",
                    modality="policy",
                    score=0.05,
                    verdict="allowed",
                    confidence_low=0.02,
                    confidence_high=0.08,
                ),
                VerifierSignal(
                    verifier="temporal",
                    modality="text",
                    score=0.04,
                    verdict="fresh",
                    confidence_low=0.01,
                    confidence_high=0.07,
                ),
                VerifierSignal(
                    verifier="numeric",
                    modality="code",
                    score=0.03,
                    verdict="consistent",
                    confidence_low=0.01,
                    confidence_high=0.05,
                ),
                VerifierSignal(
                    verifier="symbolic",
                    modality="code",
                    score=0.02,
                    verdict="valid",
                    confidence_low=0.01,
                    confidence_high=0.04,
                ),
            ),
            profile=profile,
            risk_envelope=envelope,
            policy_id="policy.critical.regulated",
        )

        assert decision.decision == "halt"
        assert decision.reason == "cross_verifier_contradiction"
        assert decision.risk_score == pytest.approx(0.94)

    def test_conservative_consensus_halts_on_high_confidence_contradiction(self):
        envelope = RiskEnvelope(
            action_category="text",
            reversibility="reversible",
            domain="regulated",
            calibrated_threshold=0.65,
            no_go_threshold=0.9,
        )
        signals = [
            VerifierSignal(
                verifier="nli",
                modality="text",
                score=0.91,
                verdict="contradiction",
                confidence_low=0.86,
                confidence_high=0.97,
                evidence_refs=("kb://claim-a",),
                latency_ms=4.0,
            ),
            VerifierSignal(
                verifier="numeric",
                modality="text",
                score=0.21,
                verdict="supported",
                confidence_low=0.18,
                confidence_high=0.34,
                evidence_refs=("calc://n-1",),
                latency_ms=1.0,
            ),
        ]

        decision = CrossVerifierConsensus().decide(
            signals,
            risk_envelope=envelope,
            policy_id="policy.critical.regulated",
        )

        assert decision.decision == "halt"
        assert decision.reason == "cross_verifier_contradiction"
        assert decision.risk_score == pytest.approx(0.91)
        assert decision.confidence_low == pytest.approx(0.18)
        assert decision.confidence_high == pytest.approx(0.97)
        assert decision.evidence_refs == ("kb://claim-a", "calc://n-1")
        assert (
            decision.to_safety_event(
                hook_id="consensus", hook_scope="agent"
            ).policy_decision
            == "halt"
        )

    def test_empty_consensus_warns_instead_of_allowing(self):
        envelope = RiskEnvelope(
            action_category="code",
            reversibility="costly",
            domain="security",
            calibrated_threshold=0.6,
            no_go_threshold=0.8,
        )

        decision = CrossVerifierConsensus().decide(
            [],
            risk_envelope=envelope,
            policy_id="policy.code",
        )

        assert decision.decision == "warn"
        assert decision.reason == "insufficient_verifier_evidence"
        assert decision.risk_score == 0.0
        assert decision.verifier_signals == ()

    def test_no_go_policy_preserves_matching_empty_consensus_decision(self):
        envelope = RiskEnvelope(
            action_category="code",
            reversibility="costly",
            domain="security",
            calibrated_threshold=0.6,
            no_go_threshold=0.8,
        )

        decision = CrossVerifierConsensus(
            no_go_policy=NoGoPolicy(enable_irreversibility_forecast=False)
        ).decide(
            [],
            risk_envelope=envelope,
            policy_id="policy.code",
        )

        assert decision.decision == "warn"
        assert decision.reason == "insufficient_verifier_evidence"
        assert "requires_human_review" not in decision.attributes

    def test_uncertain_consensus_warns_below_calibrated_halt_threshold(self):
        envelope = RiskEnvelope(
            action_category="text",
            reversibility="reversible",
            domain="regulated",
            calibrated_threshold=0.8,
            no_go_threshold=0.95,
        )
        signal = VerifierSignal(
            verifier="policy",
            modality="policy",
            score=0.5,
            verdict="uncertain",
            confidence_low=0.4,
            confidence_high=0.6,
            evidence_refs=("policy:review",),
        )

        decision = CrossVerifierConsensus(warn_threshold=0.45).decide(
            (signal,),
            risk_envelope=envelope,
            policy_id="policy.warn",
        )

        assert decision.decision == "warn"
        assert decision.reason == "cross_verifier_uncertain"
        assert decision.tenant_safe_explanation == (
            "Verifier consensus is uncertain and requires review."
        )
        assert decision.evidence_refs == ("policy:review",)

    def test_consensus_deduplicates_evidence_refs_in_signal_order(self):
        envelope = RiskEnvelope(
            action_category="text",
            reversibility="reversible",
            domain="regulated",
            calibrated_threshold=0.8,
            no_go_threshold=0.95,
        )

        decision = CrossVerifierConsensus(warn_threshold=0.45).decide(
            (
                VerifierSignal(
                    verifier="nli",
                    modality="text",
                    score=0.5,
                    verdict="uncertain",
                    confidence_low=0.4,
                    confidence_high=0.6,
                    evidence_refs=("kb://claim", "policy://rule"),
                ),
                VerifierSignal(
                    verifier="policy",
                    modality="policy",
                    score=0.4,
                    verdict="allowed",
                    confidence_low=0.3,
                    confidence_high=0.5,
                    evidence_refs=("policy://rule", "calc://claim"),
                ),
            ),
            risk_envelope=envelope,
            policy_id="policy.warn",
        )

        assert decision.evidence_refs == (
            "kb://claim",
            "policy://rule",
            "calc://claim",
        )

    def test_blank_action_sequence_is_not_serialised_for_forecast(self):
        envelope = RiskEnvelope(
            action_category="tool",
            reversibility="costly",
            domain="financial",
            calibrated_threshold=0.8,
            no_go_threshold=0.95,
        )
        signal = VerifierSignal(
            verifier="policy",
            modality="policy",
            score=0.61,
            verdict="uncertain",
            confidence_low=0.54,
            confidence_high=0.73,
            evidence_refs=("policy:change-risk",),
        )

        decision = CrossVerifierConsensus(no_go_policy=NoGoPolicy()).decide(
            (signal,),
            risk_envelope=envelope,
            policy_id="policy.finance.ops",
            action_sequence=(" ", ""),
        )

        assert decision.decision == "warn"
        assert decision.reason == "cross_verifier_uncertain"
        assert "action_sequence" not in decision.attributes

    def test_no_go_policy_blocks_irreversible_weighted_consensus(self):
        envelope = RiskEnvelope(
            action_category="physical",
            reversibility="irreversible",
            domain="physical",
            calibrated_threshold=0.5,
            no_go_threshold=0.7,
        )
        signals = [
            VerifierSignal(
                verifier="trajectory",
                modality="physical",
                score=0.62,
                verdict="uncertain",
                confidence_low=0.55,
                confidence_high=0.72,
                evidence_refs=("physical:trajectory",),
            ),
            VerifierSignal(
                verifier="policy",
                modality="policy",
                score=0.88,
                verdict="unsafe",
                confidence_low=0.88,
                confidence_high=0.88,
                evidence_refs=("policy:no-go",),
            ),
        ]
        consensus = CrossVerifierConsensus(
            mode="weighted",
            weights={"trajectory": 0.25, "policy": 0.75},
            no_go_policy=NoGoPolicy(irreversible_threshold=0.5),
        )

        decision = consensus.decide(
            signals,
            risk_envelope=envelope,
            policy_id="policy.physical",
        )

        assert decision.decision == "block"
        assert decision.reason == "no_go_irreversible_risk"
        assert decision.attributes["consensus_mode"] == "weighted"

    def test_no_go_policy_blocks_forecasted_irreversibility_from_consensus(self):
        class AlwaysIrreversibleForecaster:
            def forecast(self, actions, *, seed=0):
                assert tuple(actions) == ("stage plan", "transfer funds")
                return Forecast(
                    p_irreversible=0.87,
                    ci_low=0.78,
                    ci_high=0.93,
                    crossed=87,
                    samples=100,
                )

        envelope = RiskEnvelope(
            action_category="tool",
            reversibility="costly",
            domain="financial",
            calibrated_threshold=0.5,
            no_go_threshold=0.95,
        )
        signal = VerifierSignal(
            verifier="policy",
            modality="policy",
            score=0.61,
            verdict="uncertain",
            confidence_low=0.54,
            confidence_high=0.73,
            evidence_refs=("policy:change-risk",),
        )
        consensus = CrossVerifierConsensus(
            no_go_policy=NoGoPolicy(
                irreversible_threshold=0.95,
                irreversibility_forecaster=AlwaysIrreversibleForecaster(),
                reviewed_irreversibility_threshold=ReviewedIrreversibilityThreshold(
                    threshold=0.7,
                    source_ref="calibration://irreversibility/2026-05-13",
                    reviewer_id="reviewer-passport-a",
                    calibration_size=128,
                    coverage=0.95,
                ),
            )
        )

        decision = consensus.decide(
            (signal,),
            risk_envelope=envelope,
            policy_id="policy.finance.ops",
            action_sequence=("stage plan", "transfer funds"),
        )

        assert decision.decision == "block"
        assert decision.reason == "no_go_reviewed_irreversibility_forecast"
        assert decision.attributes["requires_human_review"] == "true"
        assert decision.attributes["irreversibility_forecast_ci_low"] == "0.780000"
        assert decision.attributes["reviewed_threshold"] == "0.700000"
        assert decision.attributes["reviewed_threshold_calibration_size"] == "128"

    def test_interval_fusion_handles_empty_signals_and_validates_weights(self):
        assert _fused_interval((), {}) == (0.0, 1.0)
        assert _weighted_value((), {}, "score") == 0.0

        signal = VerifierSignal(
            verifier="nli",
            modality="text",
            score=0.2,
            verdict="supported",
            confidence_low=0.1,
            confidence_high=0.3,
        )

        with pytest.raises(ValueError, match="weights must be non-negative"):
            _weighted_value((signal,), {"nli": -1.0}, "score")
        with pytest.raises(ValueError, match="at least one verifier weight"):
            _weighted_value((signal,), {"nli": 0.0}, "score")
