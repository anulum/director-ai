# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Google Drive ingestion plugin

"""Ingest Google Drive files into the knowledge base.

Wraps a pre-authenticated ``googleapiclient.discovery`` resource.
The adapter lists files under an optional folder filter, fetches
text-shaped MIME types via ``files.get_media`` and exports Google
Docs / Sheets / Slides via ``files.export_media``. Binary formats
not in the allowlist are skipped.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterable, Iterator
from typing import Any

from .base import IngestedDocument, IngestionPlugin

logger = logging.getLogger("DirectorAI.Ingestion.GDrive")


# MIME types that ``files.get_media`` can return as text verbatim.
_DIRECT_MIME_TYPES: frozenset[str] = frozenset(
    {
        "text/plain",
        "text/markdown",
        "text/html",
        "text/csv",
        "application/json",
        "application/xml",
    }
)


# Google-native MIME types plus the export target. ``files.export_media``
# converts them on the fly.
_NATIVE_EXPORTS: dict[str, str] = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}


class GoogleDrivePlugin(IngestionPlugin):
    """Google Drive ingestion adapter.

    Parameters
    ----------
    service : object
        A pre-authenticated Drive v3 resource — for production
        ``googleapiclient.discovery.build("drive", "v3", credentials=...)``.
    folder_id : str | None
        If set, only files whose parents list contains ``folder_id``
        are ingested. ``None`` means "every file the service account /
        OAuth user can see".
    query : str | None
        Free-form Drive query string appended to the listing filter.
        Example: ``"modifiedTime > '2025-01-01T00:00:00'"``.
    page_size : int
        ``pageSize`` for ``files.list``. Drive caps at 1000.
    direct_mime_types : Iterable[str] | None
        Allowlist for :meth:`_fetch_text`. Defaults to the shipped
        text-friendly set.
    """

    name = "gdrive"

    def __init__(
        self,
        service: Any,
        *,
        folder_id: str | None = None,
        query: str | None = None,
        page_size: int = 100,
        direct_mime_types: Iterable[str] | None = None,
    ) -> None:
        if service is None:
            raise ValueError("service is required")
        if page_size <= 0:
            raise ValueError(f"page_size must be positive; got {page_size}")
        self._service = service
        self._folder_id = folder_id
        self._query = query
        self._page_size = min(page_size, 1000)
        self._direct = frozenset(
            direct_mime_types
            if direct_mime_types is not None
            else _DIRECT_MIME_TYPES
        )

    @classmethod
    def from_service_account(
        cls,
        credentials: Any,
        **kwargs: Any,
    ) -> GoogleDrivePlugin:
        """Build a plugin from
        ``google.oauth2.service_account.Credentials``. Raises
        :class:`ImportError` with install instructions if
        ``google-api-python-client`` is missing."""
        try:
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise ImportError(
                "GoogleDrivePlugin.from_service_account requires "
                "google-api-python-client. Install with: "
                "pip install director-ai[ingestion-gdrive]",
            ) from exc
        service = build("drive", "v3", credentials=credentials, cache_discovery=False)
        return cls(service=service, **kwargs)

    def iter_documents(self) -> Iterator[IngestedDocument]:
        for file_meta in self._iter_files():
            file_id = file_meta.get("id")
            mime = str(file_meta.get("mimeType", ""))
            if not file_id:
                continue
            text = self._fetch_text(file_id, mime)
            if not text:
                continue
            metadata: dict[str, Any] = {
                "file_id": file_id,
                "mime_type": mime,
                "name": file_meta.get("name", ""),
            }
            modified = file_meta.get("modifiedTime")
            if modified:
                metadata["modified_time"] = modified
            size = file_meta.get("size")
            if size is not None:
                with contextlib.suppress(TypeError, ValueError):
                    metadata["size_bytes"] = int(size)
            yield IngestedDocument(
                key=f"gdrive:{file_id}",
                text=text,
                source=self.name,
                source_id=file_id,
                metadata=metadata,
            )

    def _iter_files(self) -> Iterator[dict[str, Any]]:
        clauses: list[str] = ["trashed = false"]
        if self._folder_id:
            clauses.append(f"'{self._folder_id}' in parents")
        if self._query:
            clauses.append(f"({self._query})")
        q = " and ".join(clauses)
        page_token: str | None = None
        while True:
            try:
                request = self._service.files().list(
                    q=q,
                    pageSize=self._page_size,
                    fields=(
                        "nextPageToken,"
                        "files(id,name,mimeType,modifiedTime,size)"
                    ),
                    pageToken=page_token,
                )
                page = request.execute()
            except Exception as exc:  # pragma: no cover — defensive
                raise RuntimeError(f"gdrive files.list failed: {exc}") from exc
            yield from (page.get("files") or [])
            page_token = page.get("nextPageToken")
            if not page_token:
                return

    def _fetch_text(self, file_id: str, mime: str) -> str:
        if mime in _NATIVE_EXPORTS:
            export_mime = _NATIVE_EXPORTS[mime]
            try:
                request = self._service.files().export_media(
                    fileId=file_id,
                    mimeType=export_mime,
                )
                raw = request.execute()
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning("gdrive export %s failed: %s", file_id, exc)
                return ""
            return _decode(raw)
        if mime in self._direct:
            try:
                request = self._service.files().get_media(fileId=file_id)
                raw = request.execute()
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning("gdrive get_media %s failed: %s", file_id, exc)
                return ""
            return _decode(raw)
        logger.debug("skip %s — mime %s not supported", file_id, mime)
        return ""


def _decode(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, bytes):
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError:
            return payload.decode("utf-8", errors="replace")
    return str(payload)
