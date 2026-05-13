# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — SafetyEvent tests

from __future__ import annotations

import json
from pathlib import Path

import pytest

from director_ai.core.safety_event import (
    SAFETY_EVENT_JSON_SCHEMA,
    SAFETY_EVENT_SCHEMA_VERSION,
    SafetyEvent,
    new_safety_event_id,
    utc_timestamp,
    validate_safety_event_payload,
)
from director_ai.core.types import EvidenceChunk, HaltEvidence, HaltTraceAttribution

ROOT = Path(__file__).resolve().parents[1]


def _event(**overrides):
    fields = {
        "event_id": "sevt_test",
        "timestamp": "2026-04-29T12:00:00Z",
        "hook_id": "streaming.kernel",
        "hook_scope": "streaming",
        "policy_decision": "halt",
        "halt_reason": "coherence_below_threshold",
        "tenant_safe_explanation": "Review grounding evidence.",
    }
    fields.update(overrides)
    return SafetyEvent(**fields)


class TestSafetyEventSchema:
    def test_generated_ids_and_timestamps_are_wire_safe(self):
        event_id = new_safety_event_id()
        timestamp = utc_timestamp()

        assert event_id.startswith("sevt_")
        assert len(event_id) == len("sevt_") + 32
        assert timestamp.endswith("Z")
        assert "+00:00" not in timestamp

    def test_minimal_event_to_dict(self):
        event = _event(threshold=0.5, observed_score=0.31)

        payload = event.to_dict()

        assert payload["schema_version"] == SAFETY_EVENT_SCHEMA_VERSION
        assert payload["hook_scope"] == "streaming"
        assert payload["policy_decision"] == "halt"
        assert payload["threshold"] == 0.5
        assert payload["observed_score"] == 0.31
        assert payload["trace_attribution"] is None

    def test_from_halt_evidence_keeps_refs_not_text(self):
        trace = HaltTraceAttribution(
            fact_source="kb://physics",
            retrieval_path="hybrid",
            scorer_path="factcg",
            token_offset=3,
            threshold=0.5,
            causal_contribution=0.19,
        )
        evidence = HaltEvidence(
            reason="coherence_below_threshold",
            last_score=0.31,
            evidence_chunks=[
                EvidenceChunk(
                    text="raw evidence text must stay out",
                    distance=0.2,
                    source="kb://physics#1",
                ),
            ],
            suggested_action="Review grounding evidence.",
            trace_attribution=trace,
        )

        event = SafetyEvent.from_halt_evidence(
            evidence,
            hook_id="streaming.kernel",
            event_id="sevt_halt",
            timestamp="2026-04-29T12:00:00Z",
            request_id="req-1",
            tenant_id="tenant-1",
            latency_ms=12.4,
        )
        payload = event.to_dict()

        assert payload["event_id"] == "sevt_halt"
        assert payload["evidence_refs"] == ["kb://physics#1"]
        assert "raw evidence text" not in str(payload)
        assert payload["trace_attribution"]["token_offset"] == 3
        assert payload["latency_ms"] == 12.4

    def test_from_halt_evidence_uses_chunk_index_fallback_ref_and_default_action(self):
        evidence = HaltEvidence(
            reason="coherence_below_threshold",
            last_score=0.31,
            evidence_chunks=[
                EvidenceChunk(text="raw text", distance=0.2, source=""),
            ],
            suggested_action="",
            trace_attribution=None,
        )

        event = SafetyEvent.from_halt_evidence(
            evidence,
            hook_id="streaming.kernel",
            event_id="sevt_halt",
            timestamp="2026-04-29T12:00:00Z",
        )

        assert event.evidence_refs == ("chunk:0",)
        assert event.tenant_safe_explanation == "Review the safety decision."
        assert event.threshold is None

    def test_from_policy_decision(self):
        event = SafetyEvent.from_policy_decision(
            hook_id="containment.guard",
            hook_scope="containment",
            policy_decision="block",
            halt_reason="containment_block",
            tenant_safe_explanation="Containment guard blocked the action.",
            event_id="sevt_policy",
            timestamp="2026-04-29T12:00:00Z",
            threshold=0.7,
            observed_score=0.2,
            evidence_refs=("containment:policy:high",),
            attributes={"finding_count": "1"},
        )

        assert event.event_id == "sevt_policy"
        assert event.evidence_refs == ("containment:policy:high",)
        assert event.attributes["finding_count"] == "1"

    @pytest.mark.parametrize(
        "field,value",
        [
            ("hook_scope", "unknown"),
            ("policy_decision", "pause"),
            ("threshold", 1.2),
            ("observed_score", -0.1),
            ("latency_ms", -1.0),
        ],
    )
    def test_invalid_values_raise(self, field, value):
        with pytest.raises(ValueError):
            _event(**{field: value})

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("schema_version", "v0", "schema_version"),
            ("event_id", " ", "event_id"),
            ("timestamp", "", "timestamp"),
            ("hook_id", "", "hook_id"),
            ("halt_reason", "", "halt_reason"),
            ("tenant_safe_explanation", "", "tenant_safe_explanation"),
        ],
    )
    def test_required_identity_and_explanation_fields_raise(
        self, field, value, message
    ):
        with pytest.raises(ValueError, match=message):
            _event(**{field: value})

    def test_evidence_refs_and_attributes_are_normalised_to_immutable_wire_values(self):
        event = _event(
            evidence_refs=["ref-a", "ref-b"],
            attributes={"count": 2, "approved": True},
        )

        assert event.evidence_refs == ("ref-a", "ref-b")
        assert event.attributes == {"count": "2", "approved": "True"}

    def test_json_schema_declares_uniform_telemetry_contract(self):
        schema = SAFETY_EVENT_JSON_SCHEMA

        assert schema["$id"].endswith("/safety-event.schema.json")
        assert schema["properties"]["schema_version"]["const"] == (
            SAFETY_EVENT_SCHEMA_VERSION
        )
        assert set(schema["required"]) >= {
            "schema_version",
            "event_id",
            "timestamp",
            "hook_id",
            "hook_scope",
            "policy_decision",
            "halt_reason",
            "tenant_safe_explanation",
            "evidence_refs",
            "attributes",
        }
        assert "inference_server" in schema["properties"]["hook_scope"]["enum"]
        assert schema["properties"]["threshold"]["minimum"] == 0.0
        assert schema["properties"]["threshold"]["maximum"] == 1.0
        assert schema["additionalProperties"] is False

    def test_published_interoperability_schema_matches_runtime_schema(self):
        published = json.loads(
            (ROOT / "schemas" / "safety-event.schema.json").read_text(encoding="utf-8")
        )

        assert published == SAFETY_EVENT_JSON_SCHEMA

    def test_validate_payload_round_trips_schema_checked_event(self):
        trace = HaltTraceAttribution(
            fact_source="kb://physics",
            retrieval_path="hybrid",
            scorer_path="factcg",
            token_offset=7,
            threshold=0.5,
            causal_contribution=0.19,
        )
        event = _event(
            hook_scope="inference_server",
            threshold=0.5,
            observed_score=0.31,
            evidence_refs=("trajectory:7",),
            attributes={"server": "vllm", "token_id": "2"},
            trace_attribution=trace,
        )

        restored = validate_safety_event_payload(event.to_dict())

        assert restored == event
        assert restored.trace_attribution == trace

    def test_validate_payload_rejects_unknown_fields_and_bad_bounds(self):
        payload = _event(threshold=0.5).to_dict()
        payload["raw_prompt"] = "do not allow raw prompt text"
        with pytest.raises(ValueError, match="unknown field"):
            validate_safety_event_payload(payload)

        payload = _event(threshold=0.5).to_dict()
        payload["threshold"] = 1.5
        with pytest.raises(ValueError, match="threshold"):
            validate_safety_event_payload(payload)

    def test_validate_payload_rejects_raw_or_secret_telemetry_refs(self):
        payload = _event(
            evidence_refs=("raw_prompt:abc",),
            attributes={"policy_id": "safe"},
        ).to_dict()
        with pytest.raises(ValueError, match="tenant-safe"):
            validate_safety_event_payload(payload)

        payload = _event(
            evidence_refs=("chunk:1",),
            attributes={"api_token": "should-not-ship"},
        ).to_dict()
        with pytest.raises(ValueError, match="tenant-safe"):
            validate_safety_event_payload(payload)

    def test_lazy_import_export(self):
        import director_ai

        assert director_ai.SafetyEvent is SafetyEvent
        assert director_ai.SAFETY_EVENT_JSON_SCHEMA is SAFETY_EVENT_JSON_SCHEMA
        assert director_ai.validate_safety_event_payload(_event().to_dict()) == _event()
