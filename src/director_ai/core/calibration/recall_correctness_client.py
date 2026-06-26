# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — record recall-correctness verdicts into REMANENTIA

"""Post a derived recall-correctness label to REMANENTIA's HTTP seam.

REMANENTIA exposes ``POST /recall/correctness`` on the same FastAPI app that
serves vector search. It resolves the supplied ``query`` to the latest matching
recall on its side and attaches a ``was_correct`` outcome to that ledger record,
closing the correctness feedback loop with Director-AI's verification verdict.

The endpoint is bearer-authenticated (it is not in REMANENTIA's public-exempt
set, unlike ``/vector/search/public``), so the client carries a token. It is
deliberately a thin, dependency-free transport over :mod:`http.client` — the
correctness label itself is derived upstream by
:mod:`director_ai.core.calibration.recall_correctness`; this module only ships
it. A *404* is a normal, non-error outcome: it means the answered query was not
served from REMANENTIA recall (e.g. a different vector backend), so there is no
record to label and :meth:`record` returns ``None`` rather than raising. Hard
transport, auth, and protocol failures raise :class:`RemanentiaCorrectnessError`
so a misconfigured loop is loud, while :meth:`try_record` is the hot-path-safe
variant that swallows every failure to a ``None`` — recording memory feedback
must never break answering.
"""

from __future__ import annotations

import http.client
import json
import logging
from urllib.parse import urlparse

from .recall_correctness import RecallOutcome

__all__ = [
    "RemanentiaCorrectnessClient",
    "RemanentiaCorrectnessError",
]

_logger = logging.getLogger(__name__)


class RemanentiaCorrectnessError(RuntimeError):
    """Raised when REMANENTIA cannot accept a recall-correctness outcome."""


class RemanentiaCorrectnessClient:
    """Record recall-correctness outcomes into REMANENTIA over HTTP.

    Parameters
    ----------
    base_url:
        REMANENTIA API root, e.g. ``http://127.0.0.1:8001``. Must be an
        ``http``/``https`` URL with a host and no query/params/fragment — the
        same shape the read-only vector backend accepts. Embedded credentials
        are rejected; pass bearer material through ``token``.
    token:
        Bearer token for the authenticated ``/recall`` family. When empty no
        ``Authorization`` header is sent, which the server will reject with
        *401*; supply the same token used for other authenticated calls.
    timeout_s:
        Per-request timeout in seconds; must be positive.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8001",
        token: str | None = None,
        timeout_s: float = 5.0,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("base_url scheme must be http or https")
        if not parsed.netloc:
            raise ValueError("base_url must include a host")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("base_url must not include credentials")
        if parsed.params or parsed.query or parsed.fragment:
            raise ValueError("base_url must not include params, query, or fragment")

        self._scheme = parsed.scheme
        self._host = parsed.hostname or ""
        self._port = parsed.port
        self._base_path = parsed.path.rstrip("/")
        self._token = token or ""
        self._timeout_s = timeout_s

    def record(self, outcome: RecallOutcome) -> str | None:
        """Attach ``outcome`` to its REMANENTIA recall; return the event id.

        Returns the matched ``event_id`` on success, or ``None`` when REMANENTIA
        has no prior recall for the query (*404*) — the answer did not come from
        REMANENTIA recall, so there is nothing to label. Raises
        :class:`RemanentiaCorrectnessError` on auth, protocol, or transport
        failure.
        """
        payload = {
            "query": outcome.query,
            "was_correct": outcome.was_correct,
            "by": outcome.by,
        }
        status, decoded = self._post("/recall/correctness", payload)
        if status == 404:
            _logger.debug("no REMANENTIA recall to label for query")
            return None
        if status < 200 or status >= 300:
            detail = decoded.get("detail") if isinstance(decoded, dict) else None
            raise RemanentiaCorrectnessError(
                f"REMANENTIA /recall/correctness returned HTTP {status}"
                + (f": {detail}" if detail else "")
            )
        if not isinstance(decoded, dict) or "event_id" not in decoded:
            raise RemanentiaCorrectnessError(
                "REMANENTIA /recall/correctness response missing event_id"
            )
        return str(decoded["event_id"])

    def try_record(self, outcome: RecallOutcome) -> str | None:
        """Hot-path-safe :meth:`record`: never raises, logs and returns ``None``.

        Use on the answering path where a memory-feedback failure must not break
        the response. Returns the event id on success, ``None`` on any failure or
        a missing recall.
        """
        try:
            return self.record(outcome)
        except (RemanentiaCorrectnessError, OSError) as exc:
            _logger.warning("recall-correctness record failed: %s", exc)
            return None

    def _post(
        self, path: str, payload: dict[str, object]
    ) -> tuple[int, dict[str, object] | None]:
        """POST JSON and return ``(status, decoded_body_or_None)``."""
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        connection_cls = (
            http.client.HTTPSConnection
            if self._scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_cls(
            self._host, port=self._port, timeout=self._timeout_s
        )
        try:
            connection.request(
                "POST", f"{self._base_path}{path}", body=body, headers=headers
            )
            response = connection.getresponse()
            raw = response.read()
            status = response.status
        except (OSError, TimeoutError, http.client.HTTPException) as exc:
            raise RemanentiaCorrectnessError(
                "REMANENTIA correctness request failed"
            ) from exc
        finally:
            connection.close()

        if not raw:
            return status, None
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RemanentiaCorrectnessError(
                "REMANENTIA correctness response was not valid JSON"
            ) from exc
        return status, decoded if isinstance(decoded, dict) else None
