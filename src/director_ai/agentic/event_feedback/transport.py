# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Credential-scoped HTTP feedback transport

"""Use verified TLS or explicit loopback without forwarding credentials on redirects."""

from __future__ import annotations

import http.client
import math
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .codec import FeedbackInvalidError
from .custody import private_bytes


@dataclass(frozen=True)
class HttpResult:
    """Retain an actual complete HTTP response before protocol validation.

    Attributes
    ----------
    status : int
        Original HTTP status.
    headers : dict[str, str]
        Header names lowercased, with ambiguous duplicates refused.
    raw : bytes
        Original body bytes, not a decoded or reconstructed substitute.
    header_pairs : tuple
        Original parsed header pairs, including any refused duplicate names.
    complete : bool
        False when the body exceeds the 1 MiB custody boundary. Raw then contains
        only the bounded prefix and cannot be admitted as a complete response.
    """

    status: int
    headers: dict[str, str]
    raw: bytes
    header_pairs: tuple[tuple[str, str], ...] = ()
    complete: bool = True


class HttpResponseInvalidError(FeedbackInvalidError):
    """Carry received evidence while refusing ambiguous or oversized responses.

    Parameters
    ----------
    response : HttpResult
        Received original status, parsed header pairs and complete body or prefix.
    """

    def __init__(self, response: HttpResult) -> None:
        """Keep transport evidence outside the fixed public error message."""
        super().__init__()
        self.response = response


class HttpEndpoint:
    """Connect to an operator-selected memory or local model endpoint.

    Parameters
    ----------
    base_url : str
        HTTPS endpoint or explicit local loopback HTTP endpoint.
    token_file : Path or None
        Owner-only bearer credential; never a routing label or CLI token value.
    timeout : float or None
        Transport wait, with None preserving complete-on-response model policy.
    """

    def __init__(
        self,
        base_url: str,
        token_file: Path | None = None,
        timeout: float | None = 30,
    ) -> None:
        """Validate the transport destination without performing a request."""
        try:
            parsed = urlsplit(base_url)
            port = parsed.port
        except ValueError as exc:
            raise FeedbackInvalidError() from exc
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or (
                parsed.scheme == "http"
                and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            )
            or any(ord(char) <= 32 or ord(char) >= 127 for char in base_url)
            or (timeout is not None and (not math.isfinite(timeout) or timeout <= 0))
        ):
            raise FeedbackInvalidError()
        self.host = parsed.hostname
        self.port = port
        self.secure = parsed.scheme == "https"
        self.prefix = parsed.path.rstrip("/")
        self.token_file = token_file
        self.timeout = timeout

    def exchange(
        self, method: str, route: str, body: bytes | None = None
    ) -> HttpResult:
        """Perform one request with no redirect, fallback or automatic retry.

        Parameters
        ----------
        method : str
            HTTP method used by the declared endpoint.
        route : str
            Encoded absolute route under the selected endpoint prefix.
        body : bytes or None
            Exact original JSON bytes for a write, absent for a read.

        Returns
        -------
        HttpResult
            Original status, headers and complete body.

        Raises
        ------
        FeedbackInvalidError
            If route, credentials or HTTP headers are ambiguous.
        OSError
            If transport fails; a possibly submitted write remains uncertain.
        """
        if (
            not route.startswith("/")
            or route.startswith("//")
            or any(ord(char) <= 32 or ord(char) >= 127 for char in route)
        ):
            raise FeedbackInvalidError()
        headers = {"Accept": "application/json"}
        if self.token_file is not None:
            try:
                token = private_bytes(self.token_file).decode("utf-8")
            except UnicodeError as exc:
                raise FeedbackInvalidError() from exc
            if not token or any(ord(char) <= 32 or ord(char) >= 127 for char in token):
                raise FeedbackInvalidError()
            headers["Authorization"] = "Bearer " + token
        if body is not None:
            headers["Content-Type"] = "application/json"
        cls = http.client.HTTPSConnection if self.secure else http.client.HTTPConnection
        connection = cls(self.host, port=self.port, timeout=self.timeout)
        try:
            connection.request(method, self.prefix + route, body=body, headers=headers)
            response = connection.getresponse()
            incomplete = False
            try:
                raw = response.read(1048577)
            except http.client.IncompleteRead as exc:
                raw = exc.partial
                incomplete = True
            except (OSError, http.client.HTTPException):
                raw = b""
                incomplete = True
            result_headers: dict[str, str] = {}
            pairs = tuple(response.getheaders())
            duplicate = False
            for key, value in pairs:
                name = key.lower()
                if name in result_headers:
                    duplicate = True
                else:
                    result_headers[name] = value
            complete = (
                not incomplete and len(raw) <= 1048576 and response.length in (None, 0)
            )
            result = HttpResult(
                response.status, result_headers, raw[:1048576], pairs, complete
            )
            if (
                duplicate
                or not complete
                or (
                    "content-length" in result_headers
                    and "transfer-encoding" in result_headers
                )
            ):
                raise HttpResponseInvalidError(result)
            return result
        finally:
            connection.close()
