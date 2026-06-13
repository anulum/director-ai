# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — SIEM audit-log sinks (Splunk / Datadog / Elasticsearch)

"""Ship audit records to a SIEM as they are written.

Each sink implements the :class:`~director_ai.core.safety.audit.AuditLogger`
sink contract — ``write(entry)`` and ``write_batch(entries)`` — so an operator
adds one with ``audit_logger.add_sink(SplunkHECSink(...))`` and every decision
is forwarded to Splunk, Datadog, or Elasticsearch alongside the local
hash-chained JSONL trail.

Only the :class:`AuditEntry` is forwarded, and it is already tenant-safe — the
query is HMAC-hashed and the response is recorded as a length, never raw text —
so nothing sensitive leaves the process. ``AuditLogger`` invokes sinks inside a
try/except, so a SIEM outage degrades to local-only logging rather than
breaking the request path; sinks raise on transport errors so the failure is
recorded.

Uses :mod:`requests` (a core dependency); no SIEM SDKs are required.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import requests

if TYPE_CHECKING:
    from .audit import AuditEntry

__all__ = [
    "SplunkHECSink",
    "DatadogLogsSink",
    "ElasticsearchSink",
]

_DEFAULT_TIMEOUT = 5.0


def _entry_dict(entry: AuditEntry) -> dict[str, Any]:
    return asdict(entry)


class SplunkHECSink:
    """Forward audit entries to a Splunk HTTP Event Collector.

    Parameters
    ----------
    url:
        HEC base URL, e.g. ``https://splunk.example.com:8088``.
    token:
        HEC token.
    sourcetype:
        Splunk sourcetype for the events.
    index:
        Optional target index.
    verify_tls:
        Verify the server certificate (default True).
    timeout:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        url: str,
        token: str,
        *,
        sourcetype: str = "director-ai:audit",
        index: str | None = None,
        verify_tls: bool = True,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        if not url or not token:
            raise ValueError("Splunk HEC url and token are required")
        self._endpoint = url.rstrip("/") + "/services/collector/event"
        self._headers = {"Authorization": f"Splunk {token}"}
        self._sourcetype = sourcetype
        self._index = index
        self._verify_tls = verify_tls
        self._timeout = timeout

    def _envelope(self, entry: AuditEntry) -> dict[str, Any]:
        env: dict[str, Any] = {
            "sourcetype": self._sourcetype,
            "event": _entry_dict(entry),
        }
        if self._index:
            env["index"] = self._index
        return env

    def write(self, entry: AuditEntry) -> None:
        resp = requests.post(
            self._endpoint,
            headers=self._headers,
            json=self._envelope(entry),
            verify=self._verify_tls,
            timeout=self._timeout,
        )
        resp.raise_for_status()

    def write_batch(self, entries: list[AuditEntry]) -> None:
        if not entries:
            return
        # Splunk HEC accepts concatenated event objects in one request body.
        import json as _json

        body = "".join(_json.dumps(self._envelope(e)) for e in entries)
        resp = requests.post(
            self._endpoint,
            headers=self._headers,
            data=body,
            verify=self._verify_tls,
            timeout=self._timeout,
        )
        resp.raise_for_status()


class DatadogLogsSink:
    """Forward audit entries to the Datadog logs intake."""

    def __init__(
        self,
        api_key: str,
        *,
        site: str = "datadoghq.com",
        service: str = "director-ai",
        source: str = "director-ai",
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        if not api_key:
            raise ValueError("Datadog api_key is required")
        self._endpoint = f"https://http-intake.logs.{site}/api/v2/logs"
        self._headers = {"DD-API-KEY": api_key, "Content-Type": "application/json"}
        self._service = service
        self._source = source
        self._timeout = timeout

    def _record(self, entry: AuditEntry) -> dict[str, Any]:
        payload = _entry_dict(entry)
        return {
            "ddsource": self._source,
            "service": self._service,
            "ddtags": f"tenant:{payload.get('tenant_id', '')}",
            "message": payload,
        }

    def write(self, entry: AuditEntry) -> None:
        self.write_batch([entry])

    def write_batch(self, entries: list[AuditEntry]) -> None:
        if not entries:
            return
        resp = requests.post(
            self._endpoint,
            headers=self._headers,
            json=[self._record(e) for e in entries],
            timeout=self._timeout,
        )
        resp.raise_for_status()


class ElasticsearchSink:
    """Forward audit entries to an Elasticsearch index.

    Single writes use ``POST {url}/{index}/_doc``; batches use the ``_bulk`` API.
    Authenticate with an API key (``Authorization: ApiKey ...``) or HTTP basic
    auth.
    """

    def __init__(
        self,
        url: str,
        index: str = "director-ai-audit",
        *,
        api_key: str | None = None,
        basic_auth: tuple[str, str] | None = None,
        verify_tls: bool = True,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        if not url:
            raise ValueError("Elasticsearch url is required")
        self._base = url.rstrip("/")
        self._index = index
        self._verify_tls = verify_tls
        self._timeout = timeout
        self._auth = basic_auth
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"ApiKey {api_key}"

    def write(self, entry: AuditEntry) -> None:
        resp = requests.post(
            f"{self._base}/{self._index}/_doc",
            headers=self._headers,
            json=_entry_dict(entry),
            auth=self._auth,
            verify=self._verify_tls,
            timeout=self._timeout,
        )
        resp.raise_for_status()

    def write_batch(self, entries: list[AuditEntry]) -> None:
        if not entries:
            return
        import json as _json

        action = _json.dumps({"index": {"_index": self._index}})
        lines = []
        for e in entries:
            lines.append(action)
            lines.append(_json.dumps(_entry_dict(e)))
        body = "\n".join(lines) + "\n"
        headers = {**self._headers, "Content-Type": "application/x-ndjson"}
        resp = requests.post(
            f"{self._base}/_bulk",
            headers=headers,
            data=body,
            auth=self._auth,
            verify=self._verify_tls,
            timeout=self._timeout,
        )
        resp.raise_for_status()
