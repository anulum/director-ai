# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the Answer Bill of Materials.

Covers claim-record and manifest validation, JSON round-tripping, the builder's
mapping from scorer claim-level provenance (supported/unsupported/contradicted,
support strength, evidence-id resolution), and the ProductionGuard integration.
"""

from __future__ import annotations

import pytest

from director_ai.core.answer_bom import (
    ANSWER_BOM_SCHEMA_VERSION,
    AnswerBOM,
    ClaimRecord,
    build_answer_bom,
    new_answer_id,
    utc_timestamp,
)
from director_ai.core.types import (
    ClaimAttribution,
    CoherenceScore,
    EvidenceChunk,
    ScoringEvidence,
)


def _attr(
    claim: str,
    *,
    divergence: float,
    supported: bool,
    source_index: int = 0,
    source: str = "matched source sentence",
) -> ClaimAttribution:
    return ClaimAttribution(
        claim=claim,
        claim_index=0,
        source_sentence=source,
        source_index=source_index,
        divergence=divergence,
        supported=supported,
    )


def _score(
    attributions: list[ClaimAttribution],
    chunks: list[EvidenceChunk] | None = None,
) -> CoherenceScore:
    evidence = ScoringEvidence(
        chunks=chunks or [],
        nli_premise="",
        nli_hypothesis="",
        nli_score=0.0,
        attributions=attributions,
        claims=[a.claim for a in attributions],
    )
    return CoherenceScore(
        score=0.7,
        approved=True,
        h_logical=0.1,
        h_factual=0.1,
        evidence=evidence,
    )


# ── ids / timestamps ────────────────────────────────────────────────────


class TestIdentifiers:
    def test_answer_id_prefix(self):
        assert new_answer_id().startswith("abom_")

    def test_answer_id_unique(self):
        assert new_answer_id() != new_answer_id()

    def test_timestamp_is_zulu(self):
        assert utc_timestamp().endswith("Z")


# ── ClaimRecord ─────────────────────────────────────────────────────────


class TestClaimRecord:
    def test_bad_verdict_rejected(self):
        with pytest.raises(ValueError, match="unsupported claim verdict"):
            ClaimRecord(claim="c", verdict="maybe", support=0.5)

    def test_support_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="support must be"):
            ClaimRecord(claim="c", verdict="supported", support=1.5)

    def test_support_nan_rejected(self):
        with pytest.raises(ValueError, match="support must be"):
            ClaimRecord(claim="c", verdict="supported", support=float("nan"))

    def test_supported_property(self):
        assert ClaimRecord(claim="c", verdict="supported", support=1.0).supported
        assert not ClaimRecord(claim="c", verdict="unsupported", support=0.0).supported

    def test_evidence_and_policy_coerced_to_tuple(self):
        record = ClaimRecord(
            claim="c",
            verdict="supported",
            support=1.0,
            evidence_ids=["a", "b"],
            policy_refs=["p"],
        )
        assert record.evidence_ids == ("a", "b")
        assert record.policy_refs == ("p",)

    def test_to_from_dict_roundtrip(self):
        record = ClaimRecord(
            claim="Paris is the capital of France.",
            verdict="supported",
            support=0.9,
            evidence_ids=("vector:doc1",),
            freshness="2026-01-01",
            tenant="acme",
            policy_refs=("kyc-1",),
        )
        assert ClaimRecord.from_dict(record.to_dict()) == record


# ── AnswerBOM ───────────────────────────────────────────────────────────


class TestAnswerBOM:
    def test_bad_schema_version_rejected(self):
        with pytest.raises(ValueError, match="schema_version"):
            AnswerBOM(
                answer_id="a",
                model="m",
                scorer="s",
                threshold=0.5,
                schema_version="bad",
            )

    def test_empty_answer_id_rejected(self):
        with pytest.raises(ValueError, match="answer_id is required"):
            AnswerBOM(answer_id="  ", model="m", scorer="s", threshold=0.5)

    def test_threshold_range_enforced(self):
        with pytest.raises(ValueError, match="threshold must be"):
            AnswerBOM(answer_id="a", model="m", scorer="s", threshold=2.0)

    def test_timestamp_autofilled(self):
        bom = AnswerBOM(answer_id="a", model="m", scorer="s", threshold=0.5)
        assert bom.timestamp.endswith("Z")

    def test_schema_version_default(self):
        bom = AnswerBOM(answer_id="a", model="m", scorer="s", threshold=0.5)
        assert bom.schema_version == ANSWER_BOM_SCHEMA_VERSION

    def test_unsupported_claims(self):
        bom = AnswerBOM(
            answer_id="a",
            model="m",
            scorer="s",
            threshold=0.5,
            claims=(
                ClaimRecord(claim="ok", verdict="supported", support=1.0),
                ClaimRecord(claim="bad", verdict="unsupported", support=0.0),
                ClaimRecord(claim="wrong", verdict="contradicted", support=0.0),
            ),
        )
        assert [c.claim for c in bom.unsupported_claims] == ["bad", "wrong"]

    def test_support_coverage_empty_is_one(self):
        bom = AnswerBOM(answer_id="a", model="m", scorer="s", threshold=0.5)
        assert bom.support_coverage == 1.0

    def test_support_coverage_mixed(self):
        bom = AnswerBOM(
            answer_id="a",
            model="m",
            scorer="s",
            threshold=0.5,
            claims=(
                ClaimRecord(claim="ok", verdict="supported", support=1.0),
                ClaimRecord(claim="bad", verdict="unsupported", support=0.0),
            ),
        )
        assert bom.support_coverage == 0.5

    def test_to_dict_shape(self):
        bom = AnswerBOM(
            answer_id="a",
            model="m",
            scorer="s",
            threshold=0.5,
            tenant="acme",
            claims=(ClaimRecord(claim="ok", verdict="supported", support=1.0),),
        )
        payload = bom.to_dict()
        assert payload["answer_id"] == "a"
        assert payload["tenant"] == "acme"
        assert payload["unsupported_claims"] == []
        assert payload["claims"][0]["verdict"] == "supported"

    def test_to_json_parses(self):
        import json

        bom = AnswerBOM(answer_id="a", model="m", scorer="s", threshold=0.5)
        assert json.loads(bom.to_json())["answer_id"] == "a"

    def test_from_dict_roundtrip(self):
        bom = AnswerBOM(
            answer_id="a",
            model="gpt",
            scorer="rust",
            threshold=0.6,
            tenant="acme",
            claims=(
                ClaimRecord(
                    claim="ok",
                    verdict="supported",
                    support=1.0,
                    evidence_ids=("vector:doc1",),
                ),
            ),
        )
        assert AnswerBOM.from_dict(bom.to_dict()).to_dict() == bom.to_dict()

    def test_from_dict_ignores_derived_fields(self):
        payload = {
            "answer_id": "a",
            "model": "m",
            "scorer": "s",
            "threshold": 0.5,
            "claims": [],
            "unsupported_claims": ["stale"],
            "support_coverage": 0.0,
        }
        bom = AnswerBOM.from_dict(payload)
        assert bom.unsupported_claims == ()
        assert bom.support_coverage == 1.0

    def test_from_dict_rejects_non_list_claims(self):
        with pytest.raises(ValueError, match="claims must be an array"):
            AnswerBOM.from_dict({"answer_id": "a", "threshold": 0.5, "claims": "nope"})


# ── builder ─────────────────────────────────────────────────────────────


class TestBuilder:
    def test_supported_claim_maps_to_evidence(self):
        chunks = [EvidenceChunk(text="src", distance=0.1, source="vector:doc1")]
        score = _score([_attr("c", divergence=0.1, supported=True)], chunks)
        bom = build_answer_bom(score, model="m", scorer="s", threshold=0.6)
        record = bom.claims[0]
        assert record.verdict == "supported"
        assert record.support == 0.9
        assert record.evidence_ids == ("vector:doc1",)

    def test_contradicted_claim(self):
        score = _score([_attr("c", divergence=0.8, supported=False)])
        bom = build_answer_bom(score, model="m", scorer="s", threshold=0.6)
        record = bom.claims[0]
        assert record.verdict == "contradicted"
        assert record.evidence_ids == ()

    def test_unsupported_below_contradiction_threshold(self):
        score = _score([_attr("c", divergence=0.3, supported=False)])
        bom = build_answer_bom(score, model="m", scorer="s", threshold=0.6)
        assert bom.claims[0].verdict == "unsupported"

    def test_contradiction_threshold_is_configurable(self):
        score = _score([_attr("c", divergence=0.4, supported=False)])
        bom = build_answer_bom(
            score, model="m", scorer="s", threshold=0.6, contradiction_threshold=0.3
        )
        assert bom.claims[0].verdict == "contradicted"

    def test_evidence_id_fallback_out_of_range(self):
        score = _score([_attr("c", divergence=0.1, supported=True, source_index=9)])
        bom = build_answer_bom(score, model="m", scorer="s", threshold=0.6)
        assert bom.claims[0].evidence_ids == ("source:9",)

    def test_evidence_id_fallback_blank_source(self):
        chunks = [EvidenceChunk(text="src", distance=0.1, source="")]
        score = _score([_attr("c", divergence=0.1, supported=True)], chunks)
        bom = build_answer_bom(score, model="m", scorer="s", threshold=0.6)
        assert bom.claims[0].evidence_ids == ("source:0",)

    def test_no_attributions_yields_empty_claims(self):
        bom = build_answer_bom(_score([]), model="m", scorer="s", threshold=0.6)
        assert bom.claims == ()
        assert bom.support_coverage == 1.0

    def test_tenant_and_policy_propagate(self):
        score = _score([_attr("c", divergence=0.1, supported=True)])
        bom = build_answer_bom(
            score,
            model="m",
            scorer="s",
            threshold=0.6,
            tenant="acme",
            freshness="2026-01-01",
            policy_refs=["kyc-1"],
        )
        record = bom.claims[0]
        assert record.tenant == "acme"
        assert record.freshness == "2026-01-01"
        assert record.policy_refs == ("kyc-1",)

    def test_freshness_only_on_supported(self):
        score = _score([_attr("c", divergence=0.9, supported=False)])
        bom = build_answer_bom(
            score, model="m", scorer="s", threshold=0.6, freshness="2026-01-01"
        )
        assert bom.claims[0].freshness == ""

    def test_support_clamped_high_divergence(self):
        score = _score([_attr("c", divergence=1.5, supported=False)])
        bom = build_answer_bom(score, model="m", scorer="s", threshold=0.6)
        assert bom.claims[0].support == 0.0

    def test_support_clamped_negative_divergence(self):
        score = _score([_attr("c", divergence=-0.2, supported=True)])
        bom = build_answer_bom(score, model="m", scorer="s", threshold=0.6)
        assert bom.claims[0].support == 1.0

    def test_answer_id_and_timestamp_overrides(self):
        score = _score([])
        bom = build_answer_bom(
            score,
            model="m",
            scorer="s",
            threshold=0.6,
            answer_id="fixed-id",
            timestamp="2026-06-06T00:00:00Z",
        )
        assert bom.answer_id == "fixed-id"
        assert bom.timestamp == "2026-06-06T00:00:00Z"

    def test_builder_uses_generated_id_by_default(self):
        bom = build_answer_bom(_score([]), model="m", scorer="s", threshold=0.6)
        assert bom.answer_id.startswith("abom_")


# ── guard integration ───────────────────────────────────────────────────


class TestGuardIntegration:
    def test_answer_bom_from_guard_result(self):
        from director_ai.guard import ProductionGuard

        guard = ProductionGuard()
        guard.load_facts({"refund": "Refunds are available within 30 days."})
        result = guard.check(
            "What is the refund window?",
            "Refunds are available within 30 days.",
        )
        bom = guard.answer_bom(result, model="mock-llm", tenant="acme")
        assert bom.model == "mock-llm"
        assert bom.tenant == "acme"
        assert bom.scorer == guard.config.scorer_backend
        assert bom.threshold == guard.config.coherence_threshold

    def test_answer_bom_roundtrips_from_guard(self):
        from director_ai.guard import ProductionGuard

        guard = ProductionGuard()
        result = guard.check("q", "a")
        bom = guard.answer_bom(result, model="m")
        assert AnswerBOM.from_dict(bom.to_dict()).to_dict() == bom.to_dict()
