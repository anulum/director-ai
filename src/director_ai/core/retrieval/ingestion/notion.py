# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Notion ingestion plugin

"""Ingest Notion pages from a workspace.

The plugin wraps a ``notion-client`` ``Client`` instance. It uses
``databases.query`` or ``search`` to enumerate candidate pages and
``blocks.children.list`` to recursively pull the text of each page.
Tests substitute a stub client with the same
``databases.query``/``search``/``blocks.children.list`` surface so
the adapter is exercisable without the real SDK.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from typing import Any

from .base import IngestedDocument, IngestionPlugin

logger = logging.getLogger("DirectorAI.Ingestion.Notion")


# Block types whose ``rich_text`` is worth pulling. Blocks outside the
# list (databases, divider, embed, …) are skipped — the plugin is
# text-first.
_TEXT_BLOCK_TYPES: frozenset[str] = frozenset(
    {
        "paragraph",
        "heading_1",
        "heading_2",
        "heading_3",
        "bulleted_list_item",
        "numbered_list_item",
        "to_do",
        "toggle",
        "quote",
        "callout",
        "code",
    }
)


class NotionPlugin(IngestionPlugin):
    """Notion workspace adapter.

    Parameters
    ----------
    client : object
        A pre-authenticated Notion client with ``databases``,
        ``search``, and ``blocks`` sub-namespaces. Typically
        ``notion_client.Client(auth=...)``.
    database_ids : Iterable[str] | None
        Database IDs to query. When ``None``, the plugin uses
        ``search`` to enumerate all pages the token has access to.
    include_block_types : Iterable[str] | None
        Block-type allowlist. Falls back to the shipped default.
    max_pages : int | None
        Hard ceiling on pages pulled. Helpful when piloting.
    """

    name = "notion"

    def __init__(
        self,
        client: Any,
        *,
        database_ids: Iterable[str] | None = None,
        include_block_types: Iterable[str] | None = None,
        max_pages: int | None = None,
    ) -> None:
        if client is None:
            raise ValueError("client is required")
        self._client = client
        self._database_ids = list(database_ids) if database_ids else []
        self._block_types = frozenset(
            include_block_types
            if include_block_types is not None
            else _TEXT_BLOCK_TYPES
        )
        self._max_pages = max_pages

    @classmethod
    def from_token(cls, token: str, **kwargs: Any) -> NotionPlugin:
        """Build a plugin backed by a stock ``notion-client.Client``.
        Raises :class:`ImportError` with install instructions if
        ``notion-client`` is missing."""
        try:
            from notion_client import Client
        except ImportError as exc:
            raise ImportError(
                "NotionPlugin.from_token requires notion-client. "
                "Install with: pip install director-ai[ingestion-notion]",
            ) from exc
        return cls(client=Client(auth=token), **kwargs)

    def iter_documents(self) -> Iterator[IngestedDocument]:
        seen = 0
        for page in self._iter_pages():
            if self._max_pages is not None and seen >= self._max_pages:
                return
            page_id = page.get("id")
            if not page_id:
                continue
            text = self._page_text(page_id)
            if not text:
                continue
            title = _extract_title(page)
            metadata: dict[str, Any] = {"page_id": page_id}
            last_edited = page.get("last_edited_time")
            if last_edited:
                metadata["last_edited_time"] = last_edited
            created = page.get("created_time")
            if created:
                metadata["created_time"] = created
            if title:
                metadata["title"] = title
            yield IngestedDocument(
                key=f"notion:{page_id}",
                text=text,
                source=self.name,
                source_id=page_id,
                metadata=metadata,
            )
            seen += 1

    def _iter_pages(self) -> Iterator[dict[str, Any]]:
        if self._database_ids:
            for db_id in self._database_ids:
                yield from _paginate(
                    lambda cursor, db_id=db_id: self._client.databases.query(
                        database_id=db_id,
                        start_cursor=cursor,
                    )
                )
            return
        yield from _paginate(
            lambda cursor: self._client.search(
                filter={"property": "object", "value": "page"},
                start_cursor=cursor,
            )
        )

    def _page_text(self, page_id: str) -> str:
        parts: list[str] = []
        for block in self._iter_blocks(page_id):
            parts.extend(self._block_text(block))
        return "\n".join(p for p in parts if p)

    def _iter_blocks(self, block_id: str) -> Iterator[dict[str, Any]]:
        for block in _paginate(
            lambda cursor, block_id=block_id: self._client.blocks.children.list(
                block_id=block_id,
                start_cursor=cursor,
            )
        ):
            yield block
            if block.get("has_children"):
                child_id = block.get("id")
                if child_id:
                    yield from self._iter_blocks(child_id)

    def _block_text(self, block: dict[str, Any]) -> list[str]:
        block_type = str(block.get("type", ""))
        if block_type not in self._block_types:
            return []
        payload = block.get(block_type, {}) or {}
        rich_text = payload.get("rich_text", []) or []
        return [segment.get("plain_text", "") for segment in rich_text]


def _paginate(
    page_fn,
) -> Iterator[dict[str, Any]]:
    cursor: str | None = None
    while True:
        page = page_fn(cursor)
        results = page.get("results", []) or []
        yield from results
        if not page.get("has_more"):
            return
        next_cursor = page.get("next_cursor")
        if not next_cursor:
            return
        cursor = next_cursor


def _extract_title(page: dict[str, Any]) -> str:
    props = page.get("properties", {}) or {}
    for value in props.values():
        if not isinstance(value, dict):
            continue
        if value.get("type") != "title":
            continue
        title = value.get("title", []) or []
        return "".join(segment.get("plain_text", "") for segment in title)
    return ""
