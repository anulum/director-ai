# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — SafetyEvent tests

from __future__ import annotations

import pytest

from director_ai.core.safety_event import SAFETY_EVENT_SCHEMA_VERSION, SafetyEvent
from director_ai.core.types import EvidenceChunk, HaltEvidence, HaltTraceAttribution


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

    def test_lazy_import_export(self):
        from director_ai import SafetyEvent as RootSafetyEvent

        assert RootSafetyEvent is SafetyEvent
