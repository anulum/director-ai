# SPDX-License-Identifier: LicenseRef-BUSL-1.1-ANULUM-TDM
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Source-grounded event feedback CLI

"""Expose original evaluation, retained-response import and current-rights replay."""

from __future__ import annotations

import argparse
import http.client
from pathlib import Path

from .codec import FeedbackInvalidError, FeedbackWriteUnknownError, Object, canonical
from .session import EventFeedbackSession


def main(argv: list[str]) -> None:
    """Run one explicit event-feedback action without exposing private evidence.

    Parameters
    ----------
    argv : list[str]
        Action and operator-profile arguments supplied by the registered CLI.
    """
    parser = argparse.ArgumentParser(prog="director-ai event-feedback")
    parser.add_argument("action", choices=("evaluate", "import-response", "replay"))
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--original-response", type=Path)
    args = parser.parse_args(argv)
    if (args.action == "import-response") != (args.original_response is not None):
        parser.error("--original-response is required only for import-response")
    try:
        session = EventFeedbackSession(args.profile)
        if args.action == "evaluate":
            result = session.evaluate()
        elif args.action == "replay":
            result = session.replay()
        else:
            result = session.import_response(args.original_response)
        output: Object = {"state": "ABSTAINED", "feedback_ack_qualified": False}
        if result is not None:
            output = {
                "state": result.state,
                "http_status": result.status,
                "feedback_ack_qualified": result.ack is not None,
            }
            if result.ack is not None:
                output["outcome_id"] = result.ack["outcome_id"]
                output["feedback_submission_id"] = result.ack["feedback_submission_id"]
        print(canonical(output).decode("utf-8"))
        if result is None:
            raise SystemExit(3)
        if result.state == "WRITE_UNKNOWN":
            raise SystemExit(5)
        if result.ack is None:
            raise SystemExit(4)
    except FeedbackWriteUnknownError:
        print('{"state":"WRITE_UNKNOWN","feedback_ack_qualified":false}')
        raise SystemExit(5) from None
    except (FeedbackInvalidError, OSError, http.client.HTTPException):
        print(
            '{"state":"REFUSED","feedback_ack_qualified":false,"error":"Feedback evidence could not be validated."}'
        )
        raise SystemExit(4) from None
