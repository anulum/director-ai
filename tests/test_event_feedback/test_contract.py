# SPDX-License-Identifier: LicenseRef-BUSL-1.1-ANULUM-TDM
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Event feedback protocol tests
"""Verify complete public request, ACK, correction and refusal bindings."""

from __future__ import annotations

import pytest

from director_ai.agentic.event_feedback.codec import (
    FeedbackInvalidError,
    Json,
    Object,
    object_value,
)
from director_ai.agentic.event_feedback.contract import (
    payload_digest,
    refusal,
    validate_ack,
    validate_request,
    validate_timestamp,
)

from .conftest import acknowledged


def test_unknown_target_with_coherent_evidence_still_refuses(
    request_body: Object,
) -> None:
    """Reject an undeclared label even when its remaining structure is consistent."""
    request_body["label_target"] = "undeclared_target"
    ref = object_value(request_body["evidence_ref"])
    ref["kind"] = "undeclared_target"
    del ref["supplied_snippet_sha256"]
    ref["source_snapshots"] = [
        {"source_id": "source", "source_version": "version", "content_sha256": "b" * 64}
    ]
    with pytest.raises(FeedbackInvalidError):
        validate_request(request_body)


def test_ack_and_correction_bind_entire_request(request_body: Object) -> None:
    """Include the exact optional predecessor in both digest and receipt."""
    validate_request(request_body)
    validate_ack(acknowledged(request_body), "event", request_body)
    original = payload_digest("event", request_body)
    request_body["supersedes_submission_id"] = "prior"
    assert payload_digest("event", request_body) != original
    validate_ack(acknowledged(request_body), "event", request_body)


@pytest.mark.parametrize(
    "member",
    [
        "contract_version",
        "recall_event_id",
        "original_query",
        "label_target",
        "was_correct",
        "verifier_by",
        "feedback_submission_id",
        "payload_sha256",
        "evidence_sha256",
        "outcome_id",
        "recording",
        "calibration_eligible",
        "recorded_at",
    ],
)
@pytest.mark.parametrize("mutation", ["missing", "changed", "type"])
def test_each_ack_field_is_bound(
    request_body: Object,
    member: str,
    mutation: str,
) -> None:
    """Never accept missing, drifted or coerced ACK members."""
    ack = acknowledged(request_body)
    if mutation == "missing":
        del ack[member]
    else:
        ack[member] = None if mutation == "type" else "changed"
        if member == "outcome_id" and mutation == "changed":
            ack[member] = ""
    with pytest.raises(FeedbackInvalidError):
        validate_ack(ack, "event", request_body)


@pytest.mark.parametrize("value", ["", "a" * 63, "A" * 64, "g" * 64, 1])
def test_request_digest_width_and_case(request_body: Object, value: Json) -> None:
    """Reject malformed complete original evidence digests."""
    object_value(request_body["evidence_ref"])["evidence_sha256"] = value
    with pytest.raises(FeedbackInvalidError):
        validate_request(request_body)


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_correctness_is_an_actual_boolean(request_body: Object, value: Json) -> None:
    """Refuse truthy values rather than converting them into labels."""
    request_body["was_correct"] = value
    with pytest.raises(FeedbackInvalidError):
        validate_request(request_body)


def test_request_extensions_and_duplicate_snippets_refuse(request_body: Object) -> None:
    """Reject undeclared fields and duplicate evidence selectors."""
    request_body["extension"] = "forged"
    with pytest.raises(FeedbackInvalidError):
        validate_request(request_body)
    del request_body["extension"]
    ref = object_value(request_body["evidence_ref"])
    ref["supplied_snippet_sha256"] = ["b" * 64, "b" * 64]
    with pytest.raises(FeedbackInvalidError):
        validate_request(request_body)


def test_retrieval_request_binds_snapshot(request_body: Object) -> None:
    """Validate the declared retrieval target without inventing policy evidence."""
    request_body["label_target"] = "retrieval_correctness"
    ref = object_value(request_body["evidence_ref"])
    ref["kind"] = "retrieval_correctness"
    del ref["supplied_snippet_sha256"]
    ref["source_snapshots"] = [
        {
            "source_id": "source",
            "source_version": "version",
            "content_sha256": "c" * 64,
        }
    ]
    validate_request(request_body)
    validate_ack(acknowledged(request_body), "event", request_body)


@pytest.mark.parametrize(
    "stamp",
    [
        "2026-10-04",
        "2026-10-04T19:00:00",
        "2026-02-31T19:00:00Z",
        "2026-10-04T25:00:00Z",
        "2026-10-04T19:00:00+99:00",
        "2026-10-04T19:00:00+00:60",
        "2026-10-04T19:00:00-12:99",
        "2026-10-04T19:00:00+23:60",
        "2026-10-04T19:00:00-23:60",
        "now",
    ],
)
def test_timestamp_requires_real_calendar_and_timezone(stamp: str) -> None:
    """Reject timezone absence, invalid dates and invalid offsets."""
    with pytest.raises(FeedbackInvalidError):
        validate_timestamp(stamp)


@pytest.mark.parametrize(
    "stamp",
    [
        "2026-10-04T19:00:00Z",
        "2026-10-04t19:00:00.1z",
        "2026-10-04T19:00:00+23:59",
        "2026-10-04T19:00:00-23:59",
    ],
)
def test_timestamp_accepts_declared_timezone_encodings(stamp: str) -> None:
    """Admit the actual RFC3339 case variants without changing original values."""
    validate_timestamp(stamp)


@pytest.mark.parametrize(
    "error,status",
    [
        ("malformed", 400),
        ("unknown_event", 404),
        ("binding_conflict", 409),
        ("submission_conflict", 409),
        ("correction_conflict", 409),
        ("evidence_unavailable", 503),
        ("recording_disabled", 503),
        ("ledger_unavailable", 503),
    ],
)
def test_authoritative_refusal_preserves_uncertainty(error: str, status: int) -> None:
    """Distinguish explicit no-write from server-declared uncertain writes."""
    assert (
        refusal(
            status,
            {
                "contract_version": "source-bound-recall-v3",
                "recording": "not_recorded",
                "error": error,
            },
        )
        == "REFUSED"
    )
    assert (
        refusal(
            status,
            {
                "contract_version": "source-bound-recall-v3",
                "recording": "write_unknown",
                "error": error,
            },
        )
        == "WRITE_UNKNOWN"
    )


def test_authentication_and_unknown_errors_are_not_success() -> None:
    """Bind error semantics to the actual status and declared schema."""
    assert refusal(401, {"detail": "authentication required"}) == "REFUSED"
    bodies: tuple[Object, ...] = (
        {
            "contract_version": "other",
            "recording": "not_recorded",
            "error": "malformed",
        },
        {
            "contract_version": "source-bound-recall-v3",
            "recording": "recorded",
            "error": "malformed",
        },
        {
            "contract_version": "source-bound-recall-v3",
            "recording": "not_recorded",
            "error": "unknown",
        },
    )
    for body in bodies:
        with pytest.raises(FeedbackInvalidError):
            refusal(400, body)


def test_wrong_label_target_and_empty_evidence_refuse(request_body: Object) -> None:
    """Reject unknown label semantics and empty evidence selections."""
    request_body["label_target"] = "other"
    with pytest.raises(FeedbackInvalidError):
        validate_request(request_body)
    request_body["label_target"] = "policy_answer_support"
    object_value(request_body["evidence_ref"])["kind"] = "retrieval_correctness"
    with pytest.raises(FeedbackInvalidError):
        validate_request(request_body)
    object_value(request_body["evidence_ref"])["kind"] = "policy_answer_support"
    object_value(request_body["evidence_ref"])["supplied_snippet_sha256"] = []
    with pytest.raises(FeedbackInvalidError):
        validate_request(request_body)
