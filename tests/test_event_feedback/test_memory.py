# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Event feedback protocol tests
"""Verify public write/recovery semantics over actual controlled HTTP transport."""

from __future__ import annotations

from pathlib import Path

import pytest

from director_ai.agentic.event_feedback.codec import (
    FeedbackWriteUnknownError,
    Object,
    array,
    canonical,
    decode,
)
from director_ai.agentic.event_feedback.memory import submit

from .conftest import ProtocolControl, acknowledged, protocol_control


def test_truncated_chunk_preserves_received_prefix_and_original_headers(
    request_body: Object, tmp_path: Path
) -> None:
    """Keep actual received bytes when a real chunked response loses its terminator."""
    control = ProtocolControl(
        body=b"5\r\nhello\r\n",
        headers=[("Transfer-Encoding", "chunked")],
        omit_content_length=True,
    )
    with protocol_control(control) as endpoint:
        result = submit(endpoint, "event", request_body, tmp_path / "custody")
    assert (
        result.state == "WRITE_UNKNOWN" and result.status == 200 and result.ack is None
    )
    assert (result.attempt_directory / "response.prefix").read_bytes() == b"hello"
    assert (
        decode((result.attempt_directory / "transport.json").read_bytes())["complete"]
        is False
    )


def test_duplicate_response_headers_retain_complete_original_evidence(
    request_body: Object, tmp_path: Path
) -> None:
    """Keep the received ACK and ambiguous headers without accepting the write."""
    body = canonical(acknowledged(request_body))
    control = ProtocolControl(
        body=body, headers=[("X-Dup", "first"), ("X-Dup", "second")]
    )
    with protocol_control(control) as endpoint:
        result = submit(endpoint, "event", request_body, tmp_path / "custody")
    assert (
        result.state == "WRITE_UNKNOWN" and result.status == 200 and result.ack is None
    )
    assert (result.attempt_directory / "response.raw").read_bytes() == body
    transport = decode((result.attempt_directory / "transport.json").read_bytes())
    assert transport["complete"] is True
    assert ["X-Dup", "first"] in array(transport["header_pairs"])
    assert ["X-Dup", "second"] in array(transport["header_pairs"])


def test_duplicate_response_custody_failure_remains_write_unknown(
    request_body: Object, tmp_path: Path
) -> None:
    """A real file collision cannot discard a possibly recorded response quietly."""
    custody = tmp_path / "custody"
    control = ProtocolControl(
        body=canonical(acknowledged(request_body)),
        headers=[("X-Dup", "a"), ("X-Dup", "b")],
        after_write_directory=custody,
    )
    with (
        protocol_control(control) as endpoint,
        pytest.raises(FeedbackWriteUnknownError),
    ):
        submit(endpoint, "event", request_body, custody)
    assert len(control.requests) == 1


def test_oversized_response_retains_only_declared_incomplete_prefix(
    request_body: Object, tmp_path: Path
) -> None:
    """Bound a real large response and never label the prefix a complete receipt."""
    control = ProtocolControl(body=b"x" * (8 * 1024 * 1024))
    with protocol_control(control) as endpoint:
        result = submit(endpoint, "event", request_body, tmp_path / "custody")
    assert (
        result.state == "WRITE_UNKNOWN" and result.status == 200 and result.ack is None
    )
    assert (result.attempt_directory / "response.prefix").read_bytes() == b"x" * 1048576
    assert not (result.attempt_directory / "response.raw").exists()
    assert (
        decode((result.attempt_directory / "transport.json").read_bytes())["complete"]
        is False
    )


def test_same_request_replay_qualifies_whole_ack(
    request_body: Object, tmp_path: Path
) -> None:
    """Send unchanged original bytes and reconcile all original receipt fields."""
    ack = acknowledged(request_body)
    control = ProtocolControl(body=canonical(ack))
    with protocol_control(control) as endpoint:
        first = submit(endpoint, "event", request_body, tmp_path / "custody")
        again = submit(endpoint, "event", request_body, tmp_path / "custody", ack)
    assert first.state == again.state == "FEEDBACK_RECORDED"
    assert first.ack == again.ack == ack
    assert control.requests[0][2] == control.requests[1][2] == canonical(request_body)
    assert first.attempt_directory != again.attempt_directory
    assert (
        decode((first.attempt_directory / "request.json").read_bytes()) == request_body
    )


@pytest.mark.parametrize(
    "body,status,state",
    [
        (b'{"detail":"authentication required"}', 401, "REFUSED"),
        (
            b'{"contract_version":"source-bound-recall-v3","recording":"not_recorded","error":"evidence_unavailable"}',
            503,
            "REFUSED",
        ),
        (
            b'{"contract_version":"source-bound-recall-v3","recording":"write_unknown","error":"ledger_unavailable"}',
            503,
            "WRITE_UNKNOWN",
        ),
        (b"malformed", 200, "WRITE_UNKNOWN"),
        (b'{"recording":"recorded"}', 200, "WRITE_UNKNOWN"),
        (b'{"recording":"recorded"}', 201, "WRITE_UNKNOWN"),
    ],
)
def test_transport_status_alone_is_not_an_ack(
    request_body: Object,
    tmp_path: Path,
    body: bytes,
    status: int,
    state: str,
) -> None:
    """Preserve explicit refusal or uncertainty for every nonqualified response."""
    with protocol_control(ProtocolControl(body=body, status=status)) as endpoint:
        result = submit(endpoint, "event", request_body, tmp_path / "custody")
    assert result.state == state and result.ack is None
    assert (result.attempt_directory / "response.raw").read_bytes() == body


def test_disconnect_after_request_recovers_same_submission(
    request_body: Object, tmp_path: Path
) -> None:
    """A lost response keeps possible-write uncertainty and the same request identity."""
    control = ProtocolControl(disconnect=True)
    with protocol_control(control) as endpoint:
        unknown = submit(endpoint, "event", request_body, tmp_path / "custody")
        control.disconnect = False
        control.body = canonical(acknowledged(request_body))
        recovered = submit(endpoint, "event", request_body, tmp_path / "custody")
    assert unknown.state == "WRITE_UNKNOWN" and unknown.status is None
    assert recovered.state == "FEEDBACK_RECORDED"
    assert control.requests[0][2] == control.requests[1][2]


def test_changed_complete_ack_remains_unknown(
    request_body: Object, tmp_path: Path
) -> None:
    """A differently minted outcome is not the original durable replay receipt."""
    prior = acknowledged(request_body)
    drifted = dict(prior)
    drifted["outcome_id"] = "different"
    with protocol_control(ProtocolControl(body=canonical(drifted))) as endpoint:
        result = submit(endpoint, "event", request_body, tmp_path / "custody", prior)
    assert result.state == "WRITE_UNKNOWN" and result.ack is None


@pytest.mark.parametrize("offset", ["+00:60", "-12:99", "+23:60"])
def test_ack_with_normalized_invalid_offset_remains_unknown(
    request_body: Object, tmp_path: Path, offset: str
) -> None:
    """Refuse malformed receipt timestamps without inferring that no write occurred."""
    ack = acknowledged(request_body)
    ack["recorded_at"] = "2026-10-04T19:00:00" + offset
    raw = canonical(ack)
    control = ProtocolControl(body=raw)
    with protocol_control(control) as endpoint:
        result = submit(endpoint, "event", request_body, tmp_path / "custody")
    assert result.state == "WRITE_UNKNOWN" and result.ack is None
    assert result.status == 200
    assert (result.attempt_directory / "response.raw").read_bytes() == raw
    assert len(control.requests) == 1


def test_actual_postwrite_custody_conflict_is_unknown(
    request_body: Object, tmp_path: Path
) -> None:
    """Never translate failed local receipt retention into authoritative refusal."""
    custody = tmp_path / "custody"
    control = ProtocolControl(
        body=canonical(acknowledged(request_body)), after_write_directory=custody
    )
    with (
        protocol_control(control) as endpoint,
        pytest.raises(FeedbackWriteUnknownError),
    ):
        submit(endpoint, "event", request_body, custody)
    assert len(control.requests) == 1
    requests = list(custody.glob("attempts/*/request.json"))
    assert len(requests) == 1 and decode(requests[0].read_bytes()) == request_body


@pytest.mark.parametrize("disconnect", [True, False])
def test_actual_disposition_publication_failure_stays_unknown(
    request_body: Object,
    tmp_path: Path,
    disconnect: bool,
) -> None:
    """An IO failure retaining an uncertain or successful disposition stays uncertain."""
    custody = tmp_path / "custody"
    control = ProtocolControl(
        body=canonical(acknowledged(request_body)),
        disconnect=disconnect,
        after_write_directory=custody,
        collision_name="disposition.json",
    )
    with (
        protocol_control(control) as endpoint,
        pytest.raises(FeedbackWriteUnknownError),
    ):
        submit(endpoint, "event", request_body, custody)
    assert len(control.requests) == 1
