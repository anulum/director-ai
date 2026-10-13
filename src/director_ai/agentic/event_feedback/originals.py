# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Current original source and served-event resolution

"""Resolve one original authorised source and its actual served byte selections."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from urllib.parse import quote, urlencode

from .codec import (
    FeedbackInvalidError,
    Json,
    Object,
    array,
    canonical,
    decode,
    object_value,
    text,
)
from .contract import VERSION, digest, members, validate_timestamp
from .transport import HttpEndpoint

_INPUT = {
    "event_encoding",
    "response_encoding",
    "recall_event_id",
    "original_query",
    "served_event_sha256",
    "served_response_sha256",
    "candidate_id",
    "candidate_text",
    "source_snapshots",
    "supplied_snippets",
}
_PROVENANCE = {
    "producer_id",
    "actor_type",
    "principal",
    "adjudicator_id",
    "procedure_id",
    "procedure_version",
    "provider",
    "model",
}
_EVENT = {
    "contract_version",
    "kind",
    "found",
    "original_query",
    "principal",
    "querying_by",
    "recall_event_id",
    "recorded_at",
    "served_response",
    "source_snapshot_status",
}


@dataclass(frozen=True)
class Originals:
    """Carry authenticated originals with separately supplied candidate/provenance.

    Attributes
    ----------
    inputs : Object
        Complete event-bound review inputs with verified original byte digests.
    provenance : Object
        Operator-selected producer identity; service admission still checks rights.
    event : Object
        Actual authenticated served event returned by the memory service.
    source : bytes
        Complete original source bytes, independent of the candidate answer.
    """

    inputs: Object
    provenance: Object
    event: Object
    source: bytes


def _selected(response: Object, selector: Object) -> tuple[bytes, tuple[int, int, str]]:
    kind = text(selector.get("kind"))
    if kind in {"context", "summary"}:
        members(selector, {"kind", "start", "end"})
        value = text(response.get("context" if kind == "context" else "snippet"))
        position = (0, 0, kind)
    else:
        if kind not in {"result", "semantic"}:
            raise FeedbackInvalidError()
        required = {"kind", "item", "start", "end"}
        members(selector, required | ({"field"} if kind == "result" else set()))
        field = text(selector["field"]) if kind == "result" else "key_point"
        if kind == "result" and field not in {
            "content",
            "text",
            "preview",
            "snippet",
            "summary",
        }:
            raise FeedbackInvalidError()
        rows = array(
            response.get("results" if kind == "result" else "semantic_memories")
        )
        item = selector["item"]
        if type(item) is not int or not 0 <= item < len(rows):
            raise FeedbackInvalidError()
        value = text(object_value(rows[item]).get(field))
        position = (1 if kind == "result" else 2, item, field)
    return _slice(
        value.encode("utf-8"), _integer(selector["start"]), _integer(selector["end"])
    ), position


def _slice(raw: bytes, start: int, end: int) -> bytes:
    if not 0 <= start < end <= len(raw):
        raise FeedbackInvalidError()
    return raw[start:end]


def _integer(value: Json) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise FeedbackInvalidError()
    return value


def _candidate_present(response: Object, candidate: str) -> bool:
    values: list[Json] = [response.get("context"), response.get("snippet")]
    for group in ("results", "semantic_memories"):
        for row in array(response.get(group, [])):
            values.extend(
                object_value(row).get(field)
                for field in (
                    "content",
                    "text",
                    "preview",
                    "snippet",
                    "summary",
                    "key_point",
                )
            )
    return candidate in values


def resolve(endpoint: HttpEndpoint, inputs: Object, provenance: Object) -> Originals:
    """Resolve actual current source rights and validate original snippet bindings.

    Parameters
    ----------
    endpoint : HttpEndpoint
        Credential-bearing memory endpoint with no redirect or fallback.
    inputs : Object
        Exact input shape for the registered one-source review procedure.
    provenance : Object
        Exact registered producer/principal/procedure/model metadata.

    Returns
    -------
    Originals
        Original event and source after every supplied digest/slice is verified.

    Raises
    ------
    FeedbackInvalidError
        If rights refuse, originals drift or any binding is unqualified.
    """
    members(inputs, _INPUT)
    members(provenance, _PROVENANCE)
    for name in _INPUT - {"source_snapshots", "supplied_snippets"}:
        text(inputs[name])
    for name in _PROVENANCE - {"principal"}:
        text(provenance[name])
    principal = object_value(provenance["principal"])
    members(principal, {"principal_id", "tenant_id", "project_id"})
    for value in principal.values():
        text(value)
    if (
        provenance["actor_type"] != "ai_project"
        or inputs["event_encoding"] != "sorted-compact-utf8-json-v1"
        or inputs["response_encoding"] != inputs["event_encoding"]
    ):
        raise FeedbackInvalidError()
    event_id = text(inputs["recall_event_id"])
    actual = endpoint.exchange("GET", "/recall/events/" + quote(event_id, safe=""))
    if actual.status != 200:
        raise FeedbackInvalidError()
    event = decode(actual.raw)
    members(event, _EVENT)
    validate_timestamp(event["recorded_at"])
    text(event["source_snapshot_status"])
    actor = object_value(event["principal"])
    members(
        actor,
        {
            "principal_id",
            "tenant_id",
            "project_id",
            "querying_by",
            "verifier_by",
            "source_ids",
        },
    )
    for name in {
        "principal_id",
        "tenant_id",
        "project_id",
        "querying_by",
        "verifier_by",
    }:
        text(actor[name])
    source_ids = array(actor["source_ids"])
    if not source_ids or len({canonical(v) for v in source_ids}) != len(source_ids):
        raise FeedbackInvalidError()
    for value in source_ids:
        text(value)
    if (
        event["kind"] != "served_recall_v3"
        or event["contract_version"] != VERSION
        or event["recall_event_id"] != event_id
        or event["original_query"] != inputs["original_query"]
        or event["querying_by"] != actor["querying_by"]
        or event["found"] is not True
        or any(type(actor[k]) is not str or actor[k] != v for k, v in principal.items())
        or actor["verifier_by"] != provenance["adjudicator_id"]
        or hashlib.sha256(canonical(event)).hexdigest()
        != digest(inputs["served_event_sha256"])
    ):
        raise FeedbackInvalidError()
    response = object_value(event["served_response"])
    if not _candidate_present(response, text(inputs["candidate_text"])):
        raise FeedbackInvalidError()
    if hashlib.sha256(canonical(response)).hexdigest() != digest(
        inputs["served_response_sha256"]
    ):
        raise FeedbackInvalidError()
    snapshots = array(inputs["source_snapshots"])
    if len(snapshots) != 1:
        raise FeedbackInvalidError()
    snapshot = object_value(snapshots[0])
    members(snapshot, {"source_id", "source_version", "content_sha256"})
    source_id, version = text(snapshot["source_id"]), text(snapshot["source_version"])
    if source_id not in array(actor["source_ids"]):
        raise FeedbackInvalidError()
    expected_digest = digest(snapshot["content_sha256"])
    fetched = endpoint.exchange(
        "GET",
        "/recall/sources?"
        + urlencode({"source_id": source_id, "source_version": version}),
    )
    if (
        fetched.status != 200
        or hashlib.sha256(fetched.raw).hexdigest() != expected_digest
        or fetched.headers.get("content-length") != str(len(fetched.raw))
        or fetched.headers.get("x-content-sha256") != expected_digest
        or fetched.headers.get("etag") != '"' + expected_digest + '"'
    ):
        raise FeedbackInvalidError()
    snippets = array(inputs["supplied_snippets"])
    if not snippets:
        raise FeedbackInvalidError()
    previous_end = -1
    served_ends: dict[tuple[int, int, str], int] = {}
    previous_position = (-1, -1, "")
    for value in snippets:
        snippet = object_value(value)
        members(
            snippet,
            {
                "source_snapshot",
                "source_start",
                "source_end",
                "text",
                "sha256",
                "served",
            },
        )
        if canonical(snippet["source_snapshot"]) != canonical(snapshot):
            raise FeedbackInvalidError()
        raw = text(snippet["text"]).encode("utf-8")
        start, end = _integer(snippet["source_start"]), _integer(snippet["source_end"])
        original_slice = _slice(fetched.raw, start, end)
        selector = object_value(snippet["served"])
        served_slice, position = _selected(response, selector)
        selected_start, selected_end = (
            _integer(selector["start"]),
            _integer(selector["end"]),
        )
        if (
            raw != original_slice
            or raw != served_slice
            or hashlib.sha256(raw).hexdigest() != digest(snippet["sha256"])
            or start < previous_end
            or selected_start < served_ends.get(position, -1)
            or position < previous_position
        ):
            raise FeedbackInvalidError()
        previous_end = end
        served_ends[position] = selected_end
        previous_position = position
    return Originals(inputs, provenance, event, fetched.raw)
