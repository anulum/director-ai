# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - cross-language contract tests

"""Property checks for Python, Rust, Go, and proto boundary contracts."""

from __future__ import annotations

import math
import re
import tomllib
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("google.protobuf")

from director_ai.core.safety_event import SafetyEvent
from director_ai.core.types import _clamp
from director_ai.proto.converters import (
    safety_event_from_proto,
    safety_event_to_proto,
    verdict_from_proto,
    verdict_to_proto,
)
from director_ai.proto.director.v1 import director_pb2 as pb

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "requirements" / "cross_language_contracts.toml"
PROTO = ROOT / "schemas" / "proto" / "director" / "v1" / "director.proto"
GO_PROTO = ROOT / "gateway" / "go" / "proto" / "director" / "v1" / "director.pb.go"

FINITE = st.floats(
    min_value=-1_000.0,
    max_value=1_000.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)
UNIT = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)
TEXT = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
    min_size=0,
    max_size=48,
)
NON_EMPTY_TEXT = TEXT.filter(lambda value: bool(value.strip()))
HALT_REASON = st.sampled_from(
    [
        None,
        "",
        "none",
        "coherence",
        "coherence_below_threshold",
        "injection",
        "policy",
        "token_timeout",
        "total_timeout",
        "callback_timeout",
        "future_reason",
    ]
)
POLICY_DECISION = st.sampled_from(["allow", "warn", "halt", "block"])
HOOK_SCOPE = st.sampled_from(
    [
        "streaming",
        "containment",
        "attestation",
        "ontology",
        "trajectory",
        "cyber_physical",
        "swarm",
        "agent",
    ]
)


def _expected_clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if math.isnan(value):
        return lo
    if math.isinf(value):
        return hi if value > 0.0 else lo
    return max(lo, min(hi, value))


@settings(max_examples=160, deadline=None)
@given(
    value=st.one_of(
        FINITE,
        st.sampled_from([math.nan, math.inf, -math.inf]),
    ),
    lo=st.floats(
        min_value=-50.0,
        max_value=0.0,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    ),
    hi=st.floats(
        min_value=0.0,
        max_value=50.0,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    ),
)
def test_python_clamp_matches_rust_score_contract(
    value: float, lo: float, hi: float
) -> None:
    if lo > hi:
        lo, hi = hi, lo

    got = _clamp(value, lo, hi)

    assert got == pytest.approx(_expected_clamp(value, lo, hi))
    assert lo <= got <= hi


@settings(max_examples=120, deadline=None)
@given(
    score=UNIT,
    halted=st.booleans(),
    halt_reason=HALT_REASON,
    hard_limit=UNIT,
    score_lower=UNIT,
    score_upper=UNIT,
    message=TEXT,
    sources=st.lists(
        st.fixed_dictionaries(
            {
                "source_id": TEXT,
                "similarity": UNIT,
                "nli_support": UNIT,
            }
        ),
        max_size=4,
    ),
)
def test_python_verdict_proto_round_trip_properties(
    score: float,
    halted: bool,
    halt_reason: str | None,
    hard_limit: float,
    score_lower: float,
    score_upper: float,
    message: str,
    sources: list[dict[str, object]],
) -> None:
    verdict = verdict_to_proto(
        score=score,
        halted=halted,
        halt_reason=halt_reason,
        hard_limit=hard_limit,
        score_lower=score_lower,
        score_upper=score_upper,
        sources=sources,
        message=message,
    )

    restored = pb.CoherenceVerdict.FromString(
        verdict.SerializeToString(deterministic=True)
    )
    plain = verdict_from_proto(restored)

    assert plain["score"] == pytest.approx(score)
    assert plain["halted"] is halted
    assert plain["hard_limit"] == pytest.approx(hard_limit)
    assert plain["score_lower"] == pytest.approx(score_lower)
    assert plain["score_upper"] == pytest.approx(score_upper)
    assert plain["message"] == message
    assert len(plain["sources"]) == len(sources)
    assert all(0.0 <= source["similarity"] <= 1.0 for source in plain["sources"])
    assert all(0.0 <= source["nli_support"] <= 1.0 for source in plain["sources"])


@settings(max_examples=80, deadline=None)
@given(
    event_id=NON_EMPTY_TEXT,
    timestamp=NON_EMPTY_TEXT,
    request_id=TEXT,
    tenant_id=TEXT,
    hook_id=NON_EMPTY_TEXT,
    hook_scope=HOOK_SCOPE,
    policy_decision=POLICY_DECISION,
    halt_reason=NON_EMPTY_TEXT,
    threshold=UNIT,
    observed_score=UNIT,
    latency_ms=st.integers(min_value=0, max_value=1_000_000),
    evidence_refs=st.lists(TEXT, max_size=4),
    tenant_safe_explanation=NON_EMPTY_TEXT,
    attributes=st.dictionaries(TEXT, TEXT, max_size=4),
)
def test_safety_event_proto_round_trip_properties(
    event_id: str,
    timestamp: str,
    request_id: str,
    tenant_id: str,
    hook_id: str,
    hook_scope: str,
    policy_decision: str,
    halt_reason: str,
    threshold: float,
    observed_score: float,
    latency_ms: int,
    evidence_refs: list[str],
    tenant_safe_explanation: str,
    attributes: dict[str, str],
) -> None:
    event = SafetyEvent.from_policy_decision(
        event_id=event_id,
        timestamp=timestamp,
        request_id=request_id,
        tenant_id=tenant_id,
        hook_id=hook_id,
        hook_scope=hook_scope,
        policy_decision=policy_decision,
        halt_reason=halt_reason,
        threshold=threshold,
        observed_score=observed_score,
        latency_ms=latency_ms,
        evidence_refs=evidence_refs,
        tenant_safe_explanation=tenant_safe_explanation,
        attributes=attributes,
    )

    restored = safety_event_from_proto(
        pb.SafetyEvent.FromString(
            safety_event_to_proto(event).SerializeToString(deterministic=True)
        )
    )

    assert restored["event_id"] == event_id
    assert restored["timestamp"] == timestamp
    assert restored["request_id"] == request_id
    assert restored["tenant_id"] == tenant_id
    assert restored["hook_id"] == hook_id
    assert restored["hook_scope"] == hook_scope
    assert restored["policy_decision"] == policy_decision
    assert restored["threshold"] == pytest.approx(threshold)
    assert restored["observed_score"] == pytest.approx(observed_score)
    assert restored["latency_ms"] == latency_ms
    assert restored["evidence_refs"] == evidence_refs
    assert restored["tenant_safe_explanation"] == tenant_safe_explanation
    assert restored["attributes"] == attributes


def test_cross_language_manifest_references_existing_gates() -> None:
    manifest = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["status"] == "active"
    boundary_ids = {boundary["id"] for boundary in manifest["boundaries"]}
    assert boundary_ids == {"python-proto-v1", "go-proto-v1", "rust-python-score"}
    for boundary in manifest["boundaries"]:
        for key in ("schema", "implementation"):
            assert (ROOT / boundary[key]).exists(), boundary[key]
        generated = boundary.get("generated", "")
        if generated:
            assert (ROOT / generated).exists(), generated
        for test_path in boundary["tests"]:
            assert (ROOT / test_path).exists(), test_path

    gate_ids = {gate["id"] for gate in manifest["gates"]}
    assert gate_ids == {"python-contracts", "go-contracts", "rust-contracts"}


def test_proto_schema_and_go_generated_fields_stay_in_sync() -> None:
    proto_text = PROTO.read_text(encoding="utf-8")
    go_text = GO_PROTO.read_text(encoding="utf-8")

    expected_fields = {
        "CoherenceVerdict": {
            "score": 1,
            "halt_reason": 3,
            "hard_limit": 4,
            "score_lower": 5,
            "score_upper": 6,
            "sources": 7,
            "message": 8,
        },
        "SafetyEvent": {
            "schema_version": 1,
            "event_id": 2,
            "hook_id": 6,
            "policy_decision": 8,
            "halt_reason": 9,
            "evidence_refs": 13,
            "tenant_safe_explanation": 14,
            "attributes": 15,
        },
    }
    for message, fields in expected_fields.items():
        assert re.search(rf"message {message} \{{", proto_text), message
        assert f"type {message} struct" in go_text
        for field, number in fields.items():
            assert re.search(rf"\b{re.escape(field)} = {number};", proto_text), field
            assert f",{number},opt,name={field}" in go_text or (
                f",{number},rep,name={field}" in go_text
            )

    enum_names = [
        "HALT_REASON_COHERENCE_BELOW_THRESHOLD",
        "HALT_REASON_INJECTION_DETECTED",
        "POLICY_DECISION_HALT",
        "POLICY_DECISION_BLOCK",
    ]
    for name in enum_names:
        assert name in proto_text
        assert name in go_text
