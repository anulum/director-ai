# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — S3 ingestion plugin

"""Ingest text-shaped S3 objects into the knowledge base.

Wraps a ``boto3.client('s3')``-compatible object. The adapter
streams object summaries via ``list_objects_v2`` and fetches each
body with ``get_object``. Binary objects (content type outside the
allowlist) are skipped — this plugin is for plain-text,
markdown, JSON, HTML, and similar; PDF / DOCX / image ingestion
lives in :mod:`director_ai.core.retrieval.doc_parser`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any

from .base import IngestedDocument, IngestionPlugin

if TYPE_CHECKING:  # pragma: no cover
    pass

logger = logging.getLogger("DirectorAI.Ingestion.S3")

_DEFAULT_ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset(
    {
        "text/plain",
        "text/markdown",
        "text/html",
        "text/csv",
        "application/json",
        "application/xml",
    }
)


class S3Plugin(IngestionPlugin):
    """S3 ingestion adapter.

    Parameters
    ----------
    client : object
        A pre-authenticated S3 client exposing ``list_objects_v2``
        and ``get_object``. Typically ``boto3.client('s3', ...)``.
    bucket : str
        Bucket name. The plugin never mutates the bucket.
    prefix : str
        Object-key prefix to filter on. Empty = whole bucket.
    allowed_content_types : Iterable[str] | None
        Content-Type allowlist. Objects with a content type outside
        the list are skipped. ``None`` uses the shipped default.
    text_encoding : str
        Decoder for object bodies. UTF-8 is the production default;
        operators with legacy data can override.
    page_size : int
        ``MaxKeys`` forwarded to ``list_objects_v2``. Zero or
        negative raises :class:`ValueError`.
    """

    name = "s3"

    def __init__(
        self,
        client: Any,
        bucket: str,
        *,
        prefix: str = "",
        allowed_content_types: Iterable[str] | None = None,
        text_encoding: str = "utf-8",
        page_size: int = 1000,
    ) -> None:
        if client is None:
            raise ValueError("client is required")
        if not bucket:
            raise ValueError("bucket is required")
        if page_size <= 0:
            raise ValueError(f"page_size must be positive; got {page_size}")
        self._client = client
        self._bucket = bucket
        self._prefix = prefix
        self._allowed = frozenset(
            allowed_content_types
            if allowed_content_types is not None
            else _DEFAULT_ALLOWED_CONTENT_TYPES
        )
        self._encoding = text_encoding
        self._page_size = page_size

    @classmethod
    def from_default_client(cls, bucket: str, **kwargs: Any) -> S3Plugin:
        """Build a plugin backed by a stock ``boto3`` S3 client. Raises
        :class:`ImportError` with install instructions if ``boto3`` is
        missing."""
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "S3Plugin.from_default_client requires boto3. "
                "Install with: pip install director-ai[ingestion-s3]",
            ) from exc
        return cls(client=boto3.client("s3"), bucket=bucket, **kwargs)

    def iter_documents(self) -> Iterator[IngestedDocument]:
        for summary in self._iter_summaries():
            key = summary.get("Key", "")
            if not key:
                continue
            try:
                obj = self._client.get_object(Bucket=self._bucket, Key=key)
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning("s3 get_object(%s) failed: %s", key, exc)
                continue
            content_type = str(obj.get("ContentType", "")).split(";", 1)[0].strip()
            if self._allowed and content_type and content_type not in self._allowed:
                logger.debug("skip %s — content-type %s not allowed", key, content_type)
                continue
            body = obj.get("Body")
            raw = body.read() if body is not None else b""
            try:
                text = raw.decode(self._encoding)
            except UnicodeDecodeError:
                logger.debug("skip %s — not %s text", key, self._encoding)
                continue
            metadata: dict[str, Any] = {
                "bucket": self._bucket,
                "content_type": content_type or "unknown",
            }
            last_modified = obj.get("LastModified") or summary.get("LastModified")
            if last_modified is not None:
                metadata["last_modified"] = str(last_modified)
            size = obj.get("ContentLength") or summary.get("Size")
            if size is not None:
                metadata["size_bytes"] = int(size)
            yield IngestedDocument(
                key=f"s3:{self._bucket}/{key}",
                text=text,
                source=self.name,
                source_id=key,
                metadata=metadata,
            )

    def _iter_summaries(self) -> Iterator[dict[str, Any]]:
        continuation: str | None = None
        while True:
            params: dict[str, Any] = {
                "Bucket": self._bucket,
                "MaxKeys": self._page_size,
            }
            if self._prefix:
                params["Prefix"] = self._prefix
            if continuation:
                params["ContinuationToken"] = continuation
            try:
                page = self._client.list_objects_v2(**params)
            except Exception as exc:  # pragma: no cover — defensive
                raise RuntimeError(f"s3 list_objects_v2 failed: {exc}") from exc
            contents = page.get("Contents", []) or []
            yield from contents
            if not page.get("IsTruncated"):
                break
            continuation = page.get("NextContinuationToken")
            if not continuation:
                break
