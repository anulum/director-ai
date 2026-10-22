# SPDX-License-Identifier: LicenseRef-BUSL-1.1-ANULUM-TDM
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Event feedback protocol tests
"""Validate source rights and byte selectors on the public resolver surface."""

from __future__ import annotations

import hashlib
from urllib.parse import urlencode

import pytest

from director_ai.agentic.event_feedback.codec import (
    FeedbackInvalidError,
    Json,
    Object,
    array,
    canonical,
    object_value,
)
from director_ai.agentic.event_feedback.originals import Originals, resolve

from .conftest import ProtocolControl, protocol_control


def source_control(originals: Originals) -> ProtocolControl:
    """Serve declared source/event wire controls without external admission."""
    control = ProtocolControl()
    sha = hashlib.sha256(originals.source).hexdigest()
    control.routes["/recall/events/event"] = (200, canonical(originals.event), [])
    control.routes[
        "/recall/sources?"
        + urlencode(
            {
                "source_id": "source",
                "source_version": "version",
            }
        )
    ] = (
        200,
        originals.source,
        [("X-Content-SHA256", sha), ("ETag", '"' + sha + '"')],
    )
    return control


def repin_event(originals: Originals) -> None:
    """Rebind mutated protocol controls to test semantic validation rather than hashing."""
    originals.inputs["served_event_sha256"] = hashlib.sha256(
        canonical(originals.event)
    ).hexdigest()
    originals.inputs["served_response_sha256"] = hashlib.sha256(
        canonical(originals.event["served_response"])
    ).hexdigest()


def test_actual_socket_resolution_binds_complete_originals(
    originals: Originals,
) -> None:
    """Validate each fetched original before the model or any memory write."""
    control = source_control(originals)
    with protocol_control(control) as endpoint:
        result = resolve(endpoint, originals.inputs, originals.provenance)
    assert result == originals
    assert [r[0] for r in control.requests] == ["GET", "GET"]


@pytest.mark.parametrize(
    "member,value",
    [
        ("actor_type", "human"),
        ("provider", ""),
        (
            "principal",
            {"principal_id": "other", "tenant_id": "tenant", "project_id": "project"},
        ),
    ],
)
def test_provenance_identity_drift_refuses(
    originals: Originals, member: str, value: Json
) -> None:
    """Reject a differently attributed producer or principal."""
    originals.provenance[member] = value
    with (
        protocol_control(source_control(originals)) as endpoint,
        pytest.raises(FeedbackInvalidError),
    ):
        resolve(endpoint, originals.inputs, originals.provenance)


@pytest.mark.parametrize(
    "member,value",
    [
        ("candidate_text", "invented candidate"),
        ("event_encoding", "different"),
        ("served_event_sha256", "a" * 64),
        ("served_response_sha256", "b" * 64),
        ("source_snapshots", []),
        ("supplied_snippets", []),
    ],
)
def test_input_drift_refuses(originals: Originals, member: str, value: Json) -> None:
    """Reject a candidate absent from the actual response and drifted originals."""
    originals.inputs[member] = value
    with (
        protocol_control(source_control(originals)) as endpoint,
        pytest.raises(FeedbackInvalidError),
    ):
        resolve(endpoint, originals.inputs, originals.provenance)


@pytest.mark.parametrize(
    "member,value",
    [
        ("kind", "other"),
        ("contract_version", "other"),
        ("recall_event_id", "other"),
        ("original_query", "other"),
        ("found", 1),
        ("querying_by", "other"),
        ("recorded_at", "2026-02-31T19:00:00Z"),
        ("recorded_at", "2026-10-04T19:00:00+00:60"),
        ("recorded_at", "2026-10-04T19:00:00-12:99"),
    ],
)
def test_event_semantics_checked_even_with_matching_digest(
    originals: Originals, member: str, value: Json
) -> None:
    """A caller-supplied digest does not authorise malformed event semantics."""
    originals.event[member] = value
    repin_event(originals)
    control = source_control(originals)
    with (
        protocol_control(control) as endpoint,
        pytest.raises(FeedbackInvalidError),
    ):
        resolve(endpoint, originals.inputs, originals.provenance)
    assert [request[0:2] for request in control.requests] == [
        ("GET", "/recall/events/event")
    ]


@pytest.mark.parametrize("source_ids", [[], ["other"], ["source", "source"], [None]])
def test_current_source_scope_is_required(
    originals: Originals, source_ids: list[Json]
) -> None:
    """No source/version fallback or cached historical source rights."""
    object_value(originals.event["principal"])["source_ids"] = source_ids
    repin_event(originals)
    with (
        protocol_control(source_control(originals)) as endpoint,
        pytest.raises(FeedbackInvalidError),
    ):
        resolve(endpoint, originals.inputs, originals.provenance)


@pytest.mark.parametrize(
    "member,value",
    [
        ("source_start", True),
        ("source_start", -1),
        ("source_end", 1000),
        ("text", "changed"),
        ("sha256", "a" * 64),
        ("source_snapshot", {"source_id": "other"}),
        (
            "served",
            {"kind": "result", "item": True, "field": "snippet", "start": 0, "end": 29},
        ),
        (
            "served",
            {
                "kind": "result",
                "item": 0,
                "field": "unsupported",
                "start": 0,
                "end": 29,
            },
        ),
        ("served", {"kind": "unknown", "start": 0, "end": 29}),
    ],
)
def test_original_snippet_tamper_refuses(
    originals: Originals, member: str, value: Json
) -> None:
    """Check source and served UTF-8 byte selections against actual originals."""
    snippet = object_value(array(originals.inputs["supplied_snippets"])[0])
    snippet[member] = value
    with (
        protocol_control(source_control(originals)) as endpoint,
        pytest.raises(FeedbackInvalidError),
    ):
        resolve(endpoint, originals.inputs, originals.provenance)


@pytest.mark.parametrize(
    "selector,field",
    [
        ({"kind": "context", "start": 0, "end": 29}, "context"),
        ({"kind": "summary", "start": 0, "end": 29}, "snippet"),
        ({"kind": "semantic", "item": 0, "start": 0, "end": 29}, "semantic_memories"),
    ],
)
def test_declared_selector_surfaces(
    originals: Originals, selector: Object, field: str
) -> None:
    """Resolve context, summary and semantic source-bound selectors on actual transport."""
    response = object_value(originals.event["served_response"])
    selector["end"] = len(originals.source)
    if field == "semantic_memories":
        response[field] = [{"key_point": originals.source.decode()}]
    else:
        response[field] = originals.source.decode()
    object_value(array(originals.inputs["supplied_snippets"])[0])["served"] = selector
    repin_event(originals)
    with protocol_control(source_control(originals)) as endpoint:
        assert (
            resolve(endpoint, originals.inputs, originals.provenance).source
            == originals.source
        )


def test_duplicate_or_overlapping_snippets_refuse(originals: Originals) -> None:
    """Reject replayed overlapping snippets before any model or feedback operation."""
    snippets = array(originals.inputs["supplied_snippets"])
    snippets.append(snippets[0])
    with (
        protocol_control(source_control(originals)) as endpoint,
        pytest.raises(FeedbackInvalidError),
    ):
        resolve(endpoint, originals.inputs, originals.provenance)


def test_source_headers_and_current_refusal_are_required(originals: Originals) -> None:
    """No successful status or prior event digest replaces source-byte identity."""
    control = source_control(originals)
    source_route = next(
        path for path in control.routes if path.startswith("/recall/sources")
    )
    control.routes[source_route] = (200, originals.source, [])
    with protocol_control(control) as endpoint, pytest.raises(FeedbackInvalidError):
        resolve(endpoint, originals.inputs, originals.provenance)
    control.routes["/recall/events/event"] = (
        401,
        b'{"detail":"authentication required"}',
        [],
    )
    with protocol_control(control) as endpoint, pytest.raises(FeedbackInvalidError):
        resolve(endpoint, originals.inputs, originals.provenance)


def test_wrong_source_bytes_refuse_with_matching_headers(originals: Originals) -> None:
    """Headers and a success status cannot qualify changed original bytes."""
    control = source_control(originals)
    route = next(path for path in control.routes if path.startswith("/recall/sources"))
    status, _, headers = control.routes[route]
    control.routes[route] = (status, b"modified", headers)
    with protocol_control(control) as endpoint, pytest.raises(FeedbackInvalidError):
        resolve(endpoint, originals.inputs, originals.provenance)
