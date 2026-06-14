# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — safety protocol tests

from __future__ import annotations

import json

import pytest

from director_ai.core.safety_event import SafetyEvent
from director_ai.core.safety_protocol import (
    DIRECTOR_SAFETY_PROTOCOL_VERSION,
    DirectorSafetySignal,
    director_safety_signal_from_event,
    validate_director_safety_signal,
)


def _event(**overrides: object) -> SafetyEvent:
    fields = {
        "event_id": "sevt_protocol",
        "timestamp": "2026-05-13T06:00:00Z",
        "hook_id": "streaming.kernel",
        "hook_scope": "streaming",
        "policy_decision": "halt",
        "halt_reason": "coherence_below_threshold",
        "tenant_safe_explanation": "Review grounding evidence.",
        "threshold": 0.5,
        "observed_score": 0.31,
        "evidence_refs": ("kb://physics#1",),
        "attributes": {"policy_id": "policy.streaming.regulated"},
    }
    fields.update(overrides)
    return SafetyEvent(**fields)


def test_protocol_signal_wraps_safety_event_without_raw_payloads() -> None:
    signal = director_safety_signal_from_event(
        _event(),
        producer_id="runtime-alpha",
        framework="generic-agent",
    )

    payload = signal.to_transport_dict()

    assert payload["protocol_version"] == DIRECTOR_SAFETY_PROTOCOL_VERSION
    assert payload["signal_id"].startswith("dsp_")
    assert payload["producer_id"] == "runtime-alpha"
    assert payload["framework"] == "generic-agent"
    assert payload["privacy"]["payload_classification"] == "tenant_safe"
    assert payload["privacy"]["raw_payload_included"] is False
    assert payload["interoperability"]["decision"] == "halt"
    assert payload["interoperability"]["severity"] == "terminal"
    assert payload["event"]["event_id"] == "sevt_protocol"
    assert payload["event"]["evidence_refs"] == ["kb://physics#1"]
    assert "do not serialize" not in json.dumps(payload).lower()
    assert validate_director_safety_signal(payload) == signal


def test_protocol_signal_rejects_sensitive_attribute_keys() -> None:
    event = _event(attributes={"raw_prompt": "do not serialize"})

    with pytest.raises(ValueError, match="tenant-safe"):
        director_safety_signal_from_event(event, producer_id="runtime-alpha")


def test_protocol_signal_requires_stable_transport_identity() -> None:
    with pytest.raises(ValueError, match="producer_id"):
        DirectorSafetySignal(
            signal_id="dsp_test",
            emitted_at="2026-05-13T06:00:00Z",
            producer_id="",
            framework="generic-agent",
            event=_event(),
        )


def test_protocol_signal_requires_protocol_identity_fields() -> None:
    for field, message in [
        ("protocol_version", "protocol_version"),
        ("signal_id", "signal_id"),
        ("emitted_at", "emitted_at"),
        ("framework", "framework"),
    ]:
        kwargs = {
            "signal_id": "dsp_test",
            "emitted_at": "2026-05-13T06:00:00Z",
            "producer_id": "runtime-alpha",
            "framework": "generic-agent",
            "event": _event(),
        }
        if field == "protocol_version":
            kwargs[field] = "v0"
        else:
            kwargs[field] = ""
        with pytest.raises(ValueError, match=message):
            DirectorSafetySignal(**kwargs)


def test_protocol_signal_rejects_sensitive_refs_and_extensions() -> None:
    with pytest.raises(ValueError, match="tenant-safe"):
        director_safety_signal_from_event(
            _event(evidence_refs=("raw_prompt:abc",)),
            producer_id="runtime-alpha",
        )
    with pytest.raises(ValueError, match="tenant-safe"):
        director_safety_signal_from_event(
            _event(),
            producer_id="runtime-alpha",
            extensions={"api_token": "abc"},
        )


def test_transport_payload_is_canonical_json_safe() -> None:
    signal = director_safety_signal_from_event(
        _event(policy_decision="warn", halt_reason="numeric_uncertain"),
        producer_id="runtime-alpha",
        framework="batch-review",
        signal_id="dsp_fixed",
        emitted_at="2026-05-13T06:01:00Z",
    )

    encoded = signal.to_json()
    decoded = json.loads(encoded)

    assert encoded == json.dumps(decoded, sort_keys=True, separators=(",", ":"))
    assert decoded["signal_id"] == "dsp_fixed"
    assert decoded["interoperability"]["severity"] == "advisory"


def test_validate_protocol_payload_rejects_transport_contract_mismatches() -> None:
    payload = director_safety_signal_from_event(
        _event(policy_decision="allow"),
        producer_id="runtime-alpha",
    ).to_transport_dict()

    bad = dict(payload, protocol_version="v0")
    with pytest.raises(ValueError, match="protocol_version"):
        validate_director_safety_signal(bad)

    bad = dict(payload, event="not-an-object")
    with pytest.raises(ValueError, match="event payload"):
        validate_director_safety_signal(bad)

    bad = dict(payload, privacy={})
    with pytest.raises(ValueError, match="tenant-safe"):
        validate_director_safety_signal(bad)

    bad = dict(payload, privacy={**payload["privacy"], "raw_payload_included": True})
    with pytest.raises(ValueError, match="raw payloads"):
        validate_director_safety_signal(bad)

    bad = dict(payload, interoperability={})
    with pytest.raises(ValueError, match="unsupported severity"):
        validate_director_safety_signal(bad)

    bad = dict(
        payload,
        interoperability={**payload["interoperability"], "severity": "terminal"},
    )
    with pytest.raises(ValueError, match="severity does not match"):
        validate_director_safety_signal(bad)


def test_lazy_import_export() -> None:
    from director_ai import DirectorSafetySignal as RootDirectorSafetySignal

    assert RootDirectorSafetySignal is DirectorSafetySignal


def test_validate_rejects_bad_protocol_version_and_raw_payload():
    signal = director_safety_signal_from_event(
        _event(), producer_id="runtime-alpha", framework="generic-agent"
    )
    payload = signal.to_transport_dict()

    bad_version = json.loads(json.dumps(payload))
    bad_version["protocol_version"] = "0.0.0"
    with pytest.raises(ValueError, match="protocol_version"):
        validate_director_safety_signal(bad_version)

    bad_privacy = json.loads(json.dumps(payload))
    bad_privacy["privacy"]["raw_payload_included"] = True
    with pytest.raises(ValueError, match="raw payloads must not be included"):
        validate_director_safety_signal(bad_privacy)
