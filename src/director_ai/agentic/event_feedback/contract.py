# SPDX-License-Identifier: LicenseRef-BUSL-1.1-ANULUM-TDM
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Event feedback request and acknowledgement bindings

"""Validate independently authored consumers of source-bound-recall-v3."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime

from .codec import (
    FeedbackInvalidError,
    Json,
    Object,
    array,
    canonical,
    object_value,
    text,
)

VERSION = "source-bound-recall-v3"
_REQUEST = {
    "original_query",
    "label_target",
    "was_correct",
    "verifier_by",
    "feedback_submission_id",
    "evidence_ref",
}
_OPTIONAL = {"supersedes_submission_id"}
_REFERENCE = {
    "kind",
    "evidence_id",
    "evidence_sha256",
    "adjudicator_id",
    "procedure_id",
    "procedure_version",
}
_ACK = {
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
}
_ERROR_STATUS = {
    "malformed": 400,
    "unknown_event": 404,
    "binding_conflict": 409,
    "submission_conflict": 409,
    "correction_conflict": 409,
    "evidence_unavailable": 503,
    "recording_disabled": 503,
    "ledger_unavailable": 503,
}


def digest(value: Json) -> str:
    """Require a complete lowercase SHA-256 hexadecimal digest.

    Parameters
    ----------
    value : Json
        Original digest member.

    Returns
    -------
    str
        The exact 64-character digest.
    """
    result = text(value)
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise FeedbackInvalidError()
    return result


def members(
    value: Object, required: set[str], optional: set[str] | None = None
) -> None:
    """Refuse missing and undeclared protocol members.

    Parameters
    ----------
    value : Object
        Original protocol object.
    required : set[str]
        Complete mandatory member set.
    optional : set[str] or None
        Explicitly supported extension members.
    """
    if not required <= value.keys() <= required | (optional or set()):
        raise FeedbackInvalidError()


def validate_request(request: Object) -> None:
    """Validate event feedback shape without granting evidence or source rights.

    Parameters
    ----------
    request : Object
        The complete immutable submission, including any correction predecessor.
    """
    members(request, _REQUEST, _OPTIONAL)
    for name in _REQUEST - {"was_correct", "evidence_ref"}:
        text(request[name])
    if type(request["was_correct"]) is not bool:
        raise FeedbackInvalidError()
    target = request["label_target"]
    if target not in ("policy_answer_support", "retrieval_correctness"):
        raise FeedbackInvalidError()
    ref = object_value(request["evidence_ref"])
    extra = (
        "supplied_snippet_sha256"
        if target == "policy_answer_support"
        else "source_snapshots"
    )
    members(ref, _REFERENCE | {extra})
    for name in _REFERENCE:
        text(ref[name])
    digest(ref["evidence_sha256"])
    if ref["kind"] != target:
        raise FeedbackInvalidError()
    values = array(ref[extra])
    if not values or len({canonical(v) for v in values}) != len(values):
        raise FeedbackInvalidError()
    for value in values:
        if target == "policy_answer_support":
            digest(value)
        else:
            snapshot = object_value(value)
            members(snapshot, {"source_id", "source_version", "content_sha256"})
            text(snapshot["source_id"])
            text(snapshot["source_version"])
            digest(snapshot["content_sha256"])
    if "supersedes_submission_id" in request:
        text(request["supersedes_submission_id"])
    canonical(request)


def payload_digest(event_id: str, request: Object) -> str:
    """Hash the complete event-plus-request envelope, retaining all extensions.

    Parameters
    ----------
    event_id : str
        Server-minted event identifier.
    request : Object
        Entire strict feedback request.

    Returns
    -------
    str
        Full lowercase SHA-256 of the canonical envelope.
    """
    text(event_id)
    validate_request(request)
    return hashlib.sha256(
        canonical({"recall_event_id": event_id, "request": request})
    ).hexdigest()


def validate_ack(ack: Object, event_id: str, request: Object) -> None:
    """Bind every acknowledgement field to the original retained submission.

    Parameters
    ----------
    ack : Object
        Strictly decoded acknowledgement; transport success is insufficient.
    event_id : str
        Original server-minted event identifier.
    request : Object
        Entire original feedback request.
    """
    expected: Object = {
        "contract_version": VERSION,
        "recall_event_id": event_id,
        "payload_sha256": payload_digest(event_id, request),
        "evidence_sha256": object_value(request["evidence_ref"])["evidence_sha256"],
        "recording": "recorded",
        "calibration_eligible": False,
    }
    for name in _REQUEST - {"evidence_ref"} | _OPTIONAL:
        if name in request:
            expected[name] = request[name]
    optional = _OPTIONAL if "supersedes_submission_id" in request else set()
    members(ack, _ACK | optional)
    for name, value in expected.items():
        if type(ack[name]) is not type(value) or ack[name] != value:
            raise FeedbackInvalidError()
    text(ack["outcome_id"])
    validate_timestamp(ack["recorded_at"])
    canonical(ack)


def validate_timestamp(value: Json) -> None:
    """Require an actual calendar timestamp with an explicit timezone.

    Parameters
    ----------
    value : Json
        Original recorded_at member without conversion or normalisation.
    """
    stamp = text(value)
    if (
        re.fullmatch(
            r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-](?:[01]\d|2[0-3]):[0-5]\d)",
            stamp,
        )
        is None
    ):
        raise FeedbackInvalidError()
    try:
        when = datetime.fromisoformat(stamp.replace("z", "Z"))
        when.utcoffset()
    except ValueError as exc:
        raise FeedbackInvalidError() from exc


def refusal(status: int, body: Object) -> str:
    """Classify an authoritative HTTP refusal without fabricating success.

    Parameters
    ----------
    status : int
        Actual HTTP response status.
    body : Object
        Complete strictly decoded response body.

    Returns
    -------
    str
        REFUSED or WRITE_UNKNOWN; malformed protocol responses are rejected.
    """
    if status == 401 and body == {"detail": "authentication required"}:
        return "REFUSED"
    members(body, {"contract_version", "recording", "error"})
    if body["contract_version"] != VERSION:
        raise FeedbackInvalidError()
    error = text(body["error"])
    recording = body["recording"]
    if _ERROR_STATUS.get(error) != status or recording not in (
        "not_recorded",
        "write_unknown",
    ):
        raise FeedbackInvalidError()
    return "REFUSED" if recording == "not_recorded" else "WRITE_UNKNOWN"
