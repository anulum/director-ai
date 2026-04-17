# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — tests for the generated proto layer

"""Multi-angle coverage for schemas/proto/director/v1/director.proto
and director_ai.proto.converters: construction, serialisation,
round-trip, deterministic bytes, streaming messages, tenant/audit
records, and forward-compatibility (unknown halt-reason tolerated)."""

from __future__ import annotations

import pytest

from director_ai.proto.converters import (
    halt_reason_from_string,
    halt_reason_to_string,
    verdict_from_proto,
    verdict_to_proto,
)
from director_ai.proto.director.v1 import director_pb2 as pb


class TestCoherenceVerdictRoundTrip:
    def test_minimal_construct_and_serialise(self):
        v = verdict_to_proto(score=0.82, halted=False)
        buf = v.SerializeToString()
        restored = pb.CoherenceVerdict.FromString(buf)
        assert restored.score == pytest.approx(0.82)
        assert restored.halted is False
        assert restored.halt_reason == pb.HALT_REASON_NONE
        assert list(restored.sources) == []

    def test_halt_branch_with_sources(self):
        v = verdict_to_proto(
            score=0.31,
            halted=True,
            halt_reason="coherence",
            hard_limit=0.5,
            score_lower=0.28,
            score_upper=0.34,
            sources=[
                {
                    "source_id": "kb:fact-42",
                    "similarity": 0.88,
                    "nli_support": 0.46,
                }
            ],
            message="Coherence below threshold",
        )
        restored = pb.CoherenceVerdict.FromString(v.SerializeToString())
        assert restored.halted is True
        assert restored.halt_reason == pb.HALT_REASON_COHERENCE_BELOW_THRESHOLD
        assert restored.hard_limit == pytest.approx(0.5)
        assert restored.score_lower == pytest.approx(0.28)
        assert restored.score_upper == pytest.approx(0.34)
        assert len(restored.sources) == 1
        assert restored.sources[0].source_id == "kb:fact-42"
        assert restored.sources[0].similarity == pytest.approx(0.88)

    def test_dict_round_trip(self):
        v = verdict_to_proto(
            score=0.6,
            halted=False,
            hard_limit=0.5,
            sources=[{"source_id": "s1", "similarity": 0.7, "nli_support": 0.9}],
        )
        d = verdict_from_proto(v)
        assert d["score"] == pytest.approx(0.6)
        assert d["halted"] is False
        assert d["halt_reason"] == "none"
        assert d["sources"][0]["source_id"] == "s1"

    def test_deterministic_bytes(self):
        """Same inputs → identical serialised bytes (stable wire)."""
        v1 = verdict_to_proto(score=0.5, halted=False, hard_limit=0.5)
        v2 = verdict_to_proto(score=0.5, halted=False, hard_limit=0.5)
        assert v1.SerializeToString(deterministic=True) == v2.SerializeToString(
            deterministic=True
        )


class TestHaltReasonMapping:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("", pb.HALT_REASON_NONE),
            ("none", pb.HALT_REASON_NONE),
            ("coherence", pb.HALT_REASON_COHERENCE_BELOW_THRESHOLD),
            ("COHERENCE", pb.HALT_REASON_COHERENCE_BELOW_THRESHOLD),
            ("coherence_below_threshold", pb.HALT_REASON_COHERENCE_BELOW_THRESHOLD),
            ("injection", pb.HALT_REASON_INJECTION_DETECTED),
            ("policy", pb.HALT_REASON_POLICY_VIOLATION),
            ("token_timeout", pb.HALT_REASON_TOKEN_TIMEOUT),
            ("total_timeout", pb.HALT_REASON_TOTAL_TIMEOUT),
            ("callback_timeout", pb.HALT_REASON_CALLBACK_TIMEOUT),
        ],
    )
    def test_known_strings(self, text, expected):
        assert halt_reason_from_string(text) == expected

    def test_none_string_maps_to_none(self):
        assert halt_reason_from_string(None) == pb.HALT_REASON_NONE

    def test_unknown_string_is_unspecified(self):
        assert halt_reason_from_string("nuclear_meltdown") == pb.HALT_REASON_UNSPECIFIED

    def test_inverse_symmetry(self):
        for code in [
            pb.HALT_REASON_NONE,
            pb.HALT_REASON_COHERENCE_BELOW_THRESHOLD,
            pb.HALT_REASON_INJECTION_DETECTED,
            pb.HALT_REASON_POLICY_VIOLATION,
            pb.HALT_REASON_TOKEN_TIMEOUT,
            pb.HALT_REASON_TOTAL_TIMEOUT,
            pb.HALT_REASON_CALLBACK_TIMEOUT,
        ]:
            s = halt_reason_to_string(code)
            assert halt_reason_from_string(s) == code

    def test_unspecified_inverse(self):
        assert halt_reason_to_string(pb.HALT_REASON_UNSPECIFIED) == "unspecified"

    def test_unknown_enum_code_is_unspecified(self):
        assert halt_reason_to_string(9999) == "unspecified"


class TestChatCompletionShape:
    def test_request_serialise_round_trip(self):
        req = pb.ChatCompletionRequest(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=256,
            stream=True,
            tenant_id="tenant-123",
            request_id="req-abc",
        )
        req.messages.append(pb.ChatMessage(role=pb.ROLE_SYSTEM, content="You are an expert."))
        req.messages.append(pb.ChatMessage(role=pb.ROLE_USER, content="What is 2+2?"))
        buf = req.SerializeToString()
        restored = pb.ChatCompletionRequest.FromString(buf)
        assert restored.model == "gpt-4o-mini"
        assert restored.stream is True
        assert len(restored.messages) == 2
        assert restored.messages[0].role == pb.ROLE_SYSTEM
        assert restored.messages[1].content == "What is 2+2?"

    def test_response_with_coherence_and_usage(self):
        resp = pb.ChatCompletionResponse(
            id="resp-1",
            model="gpt-4o-mini",
            created_unix=1_700_000_000,
            usage=pb.TokenUsage(
                prompt_tokens=12, completion_tokens=4, total_tokens=16
            ),
            coherence=verdict_to_proto(score=0.91, halted=False, hard_limit=0.5),
        )
        resp.choices.append(
            pb.ChatChoice(
                index=0,
                message=pb.ChatMessage(role=pb.ROLE_ASSISTANT, content="4"),
                finish_reason="stop",
            )
        )
        restored = pb.ChatCompletionResponse.FromString(resp.SerializeToString())
        assert restored.usage.total_tokens == 16
        assert restored.coherence.score == pytest.approx(0.91)
        assert restored.choices[0].message.content == "4"


class TestStreamingScoringShape:
    def test_score_token_request_accumulates(self):
        req = pb.ScoreTokenRequest(
            tenant_id="t",
            request_id="r",
            accumulated_text="Paris is ",
            next_token="the capital",
        )
        req.documents.append("Paris is the capital of France.")
        buf = req.SerializeToString()
        restored = pb.ScoreTokenRequest.FromString(buf)
        assert restored.accumulated_text == "Paris is "
        assert restored.next_token == "the capital"
        assert list(restored.documents) == ["Paris is the capital of France."]

    def test_score_token_response_carries_verdict(self):
        resp = pb.ScoreTokenResponse(
            verdict=verdict_to_proto(score=0.72, halted=False, hard_limit=0.5)
        )
        restored = pb.ScoreTokenResponse.FromString(resp.SerializeToString())
        assert restored.verdict.score == pytest.approx(0.72)


class TestTenantAndAuditShape:
    def test_tenant_record_round_trip(self):
        t = pb.Tenant(
            tenant_id="acme-corp",
            display_name="ACME Corp",
            tier=pb.TENANT_TIER_PRO,
            created_unix=1_700_000_000,
            rpm_limit=600,
            rpd_limit=100_000,
        )
        t.api_key_fingerprints.append("abcd1234ef567890")
        restored = pb.Tenant.FromString(t.SerializeToString())
        assert restored.tier == pb.TENANT_TIER_PRO
        assert restored.rpm_limit == 600
        assert list(restored.api_key_fingerprints) == ["abcd1234ef567890"]

    def test_api_key_metadata_round_trip(self):
        k = pb.APIKeyMetadata(
            fingerprint="fingerprint1",
            tenant_id="t1",
            issued_unix=1_700_000_000,
            expires_unix=1_800_000_000,
            revoked=False,
        )
        restored = pb.APIKeyMetadata.FromString(k.SerializeToString())
        assert restored.fingerprint == "fingerprint1"
        assert restored.expires_unix == 1_800_000_000

    def test_audit_record_round_trip(self):
        rec = pb.AuditRecord(
            timestamp="2026-04-17T04:00:00.000Z",
            request_id="req-1",
            tenant_id="t-1",
            api_key_fingerprint="ab12cd34",
            query_hash="deadbeef",
            response_length=120,
            latency_ms=42,
            model="gpt-4o-mini",
            verdict=verdict_to_proto(score=0.88, halted=False, hard_limit=0.5),
        )
        rec.policy_violations.append("pii:email")
        restored = pb.AuditRecord.FromString(rec.SerializeToString())
        assert restored.latency_ms == 42
        assert restored.verdict.score == pytest.approx(0.88)
        assert list(restored.policy_violations) == ["pii:email"]


class TestServiceDescriptors:
    def test_services_are_defined(self):
        """The .proto service entries should appear in the file
        descriptor — regression guard against an incomplete
        regeneration."""
        from director_ai.proto.director.v1 import director_pb2_grpc

        assert hasattr(director_pb2_grpc, "CoherenceScoringServicer")
        assert hasattr(director_pb2_grpc, "ChatGatewayServicer")
        assert hasattr(director_pb2_grpc, "add_CoherenceScoringServicer_to_server")
        assert hasattr(director_pb2_grpc, "add_ChatGatewayServicer_to_server")
