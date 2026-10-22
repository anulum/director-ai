# SPDX-License-Identifier: LicenseRef-BUSL-1.1-ANULUM-TDM
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Strict event memory writeback and current-rights replay

"""Accept only complete bound ACKs and retain uncertain submissions unchanged."""

from __future__ import annotations

import http.client
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

from .codec import (
    FeedbackInvalidError,
    FeedbackWriteUnknownError,
    Object,
    canonical,
    decode,
    text,
)
from .contract import refusal, validate_ack, validate_request
from .custody import PinnedDirectory, durable_directory
from .transport import HttpEndpoint, HttpResponseInvalidError


@dataclass(frozen=True)
class FeedbackOutcome:
    """Keep event-scoped durability separate from accepted workflow-result memory.

    Attributes
    ----------
    state : str
        FEEDBACK_RECORDED, REFUSED or WRITE_UNKNOWN.
    status : int or None
        Actual HTTP status, unavailable on transport failure.
    ack : Object or None
        Entire bound receipt only when qualified; never a legacy event-ID string.
    attempt_directory : Path
        Private original request/response custody for this transport attempt.
    """

    state: str
    status: int | None
    ack: Object | None
    attempt_directory: Path


def submit(
    endpoint: HttpEndpoint,
    event_id: str,
    request: Object,
    custody: Path | PinnedDirectory,
    prior_ack: Object | None = None,
) -> FeedbackOutcome:
    """Submit or replay exactly one retained event request under current rights.

    Parameters
    ----------
    endpoint : HttpEndpoint
        Actual authenticated memory transport.
    event_id : str
        Original server-minted served-event identity.
    request : Object
        Complete immutable request, with the same submission ID on recovery.
    custody : Path or PinnedDirectory
        Operator-selected private custody; SDK supplies its registered held directory.
    prior_ack : Object or None
        Original complete ACK; replay must equal its canonical bytes.

    Returns
    -------
    FeedbackOutcome
        Exact event-feedback state; never full-result admission or retention.
    """
    validate_request(request)
    text(event_id)
    if prior_ack is not None:
        validate_ack(prior_ack, event_id, request)
    if isinstance(custody, Path):
        with durable_directory(custody) as directory:
            return submit(endpoint, event_id, request, directory, prior_ack)
    with custody.child("attempts") as attempts, attempts.child(uuid4().hex) as attempt:
        raw_request = canonical(request)
        attempt.retain("request.json", raw_request)
        attempt.synchronise()
        try:
            response = endpoint.exchange(
                "POST",
                "/recall/events/" + quote(event_id, safe="") + "/correctness",
                raw_request,
            )
        except HttpResponseInvalidError as exc:
            response = exc.response
            try:
                attempt.retain(
                    "response.raw" if response.complete else "response.prefix",
                    response.raw,
                )
                attempt.retain(
                    "transport.json",
                    canonical(
                        {
                            "status": response.status,
                            "header_pairs": [
                                list(pair) for pair in response.header_pairs
                            ],
                            "complete": response.complete,
                        }
                    ),
                )
                attempt.retain(
                    "disposition.json",
                    canonical({"state": "WRITE_UNKNOWN", "status": response.status}),
                )
            except (FeedbackInvalidError, OSError) as retention:
                raise FeedbackWriteUnknownError() from retention
            return FeedbackOutcome("WRITE_UNKNOWN", response.status, None, attempt.path)
        except (OSError, http.client.HTTPException, FeedbackInvalidError):
            try:
                attempt.retain(
                    "disposition.json",
                    canonical({"state": "WRITE_UNKNOWN", "status": None}),
                )
            except (FeedbackInvalidError, OSError) as exc:
                raise FeedbackWriteUnknownError() from exc
            return FeedbackOutcome("WRITE_UNKNOWN", None, None, attempt.path)
        try:
            attempt.retain("response.raw", response.raw)
            attempt.retain(
                "transport.json",
                canonical(
                    {"status": response.status, "headers": dict(response.headers)}
                ),
            )
        except (FeedbackInvalidError, OSError) as exc:
            raise FeedbackWriteUnknownError() from exc
        ack: Object | None = None
        state = "WRITE_UNKNOWN"
        try:
            body = decode(response.raw)
            if response.status == 200:
                validate_ack(body, event_id, request)
                if prior_ack is not None and canonical(body) != canonical(prior_ack):
                    raise FeedbackInvalidError()
                state, ack = "FEEDBACK_RECORDED", body
            else:
                state = refusal(response.status, body)
        except FeedbackInvalidError:
            pass
        try:
            attempt.retain(
                "disposition.json",
                canonical({"state": state, "status": response.status}),
            )
        except (FeedbackInvalidError, OSError) as exc:
            raise FeedbackWriteUnknownError() from exc
        return FeedbackOutcome(state, response.status, ack, attempt.path)
