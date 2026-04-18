# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Auto-KB ingestion plugin tests

"""Multi-angle coverage for the ingestion package: shared ``ingest``
loop, S3 paginator with content-type and encoding filters, Notion
block walker with rich-text extraction, and Google Drive with
native-export handling. Every test runs against an in-memory stub
client — no network, no cloud creds, no boto3/notion-client/
googleapiclient imports required."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from director_ai.core.retrieval.ingestion import (
    GoogleDrivePlugin,
    IngestedDocument,
    IngestionPlugin,
    NotionPlugin,
    S3Plugin,
)
from director_ai.core.retrieval.ingestion.base import chunks

# --- Shared in-memory store ------------------------------------------


class _RecordingStore:
    """Minimal GroundTruthStore double — only the methods the plugin
    base class exercises."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def add(self, key: str, value: str, **kwargs: Any) -> None:
        self.records.append({"key": key, "value": value, **kwargs})


# --- Base class + chunks ---------------------------------------------


class _FixedPlugin(IngestionPlugin):
    name = "fixed"

    def __init__(self, docs: list[IngestedDocument]) -> None:
        self._docs = docs

    def iter_documents(self):
        yield from self._docs


class TestBase:
    def test_ingest_skips_empty_text(self):
        docs = [
            IngestedDocument(key="a", text="hello", source="x", source_id="1"),
            IngestedDocument(key="b", text="   ", source="x", source_id="2"),
            IngestedDocument(key="c", text="", source="x", source_id="3"),
        ]
        store = _RecordingStore()
        plugin = _FixedPlugin(docs)
        written = plugin.ingest(store, tenant_id="t1")
        assert written == 1
        assert store.records[0]["key"] == "a"
        assert store.records[0]["tenant_id"] == "t1"
        assert store.records[0]["metadata"]["source"] == "x"

    def test_ingest_honours_max_documents(self):
        docs = [
            IngestedDocument(key=f"k{i}", text="t", source="x", source_id=str(i))
            for i in range(5)
        ]
        store = _RecordingStore()
        _FixedPlugin(docs).ingest(store, max_documents=2)
        assert len(store.records) == 2

    def test_chunks_batches_correctly(self):
        assert list(chunks(range(7), 3)) == [[0, 1, 2], [3, 4, 5], [6]]

    def test_chunks_rejects_non_positive_size(self):
        with pytest.raises(ValueError):
            list(chunks([1, 2], 0))


# --- S3 ---------------------------------------------------------------


@dataclass
class _FakeBody:
    payload: bytes

    def read(self) -> bytes:
        return self.payload


class _FakeS3:
    def __init__(self, pages: list[dict[str, Any]], objects: dict[str, dict[str, Any]]):
        self._pages = pages
        self._objects = objects
        self.list_calls: list[dict[str, Any]] = []
        self.get_calls: list[str] = []

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        self.list_calls.append(kwargs)
        token = kwargs.get("ContinuationToken")
        idx = 0
        if token is not None:
            idx = int(token)
        if idx >= len(self._pages):
            return {"Contents": [], "IsTruncated": False}
        return self._pages[idx]

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        self.get_calls.append(Key)
        meta = dict(self._objects[Key])
        meta["Body"] = _FakeBody(meta.pop("_body"))
        return meta


class TestS3:
    def test_single_page_ingestion(self):
        pages = [
            {
                "Contents": [
                    {"Key": "a.md"},
                    {"Key": "b.txt"},
                ],
                "IsTruncated": False,
            }
        ]
        objs = {
            "a.md": {
                "ContentType": "text/markdown",
                "ContentLength": 5,
                "_body": b"hello",
                "LastModified": "2026-04-17T00:00:00Z",
            },
            "b.txt": {
                "ContentType": "text/plain",
                "_body": b"world",
            },
        }
        plugin = S3Plugin(_FakeS3(pages, objs), bucket="kb")
        docs = list(plugin.iter_documents())
        assert {d.source_id for d in docs} == {"a.md", "b.txt"}
        assert docs[0].metadata["bucket"] == "kb"
        assert docs[0].metadata["content_type"] == "text/markdown"
        assert docs[0].metadata["size_bytes"] == 5

    def test_pagination(self):
        pages = [
            {
                "Contents": [{"Key": "a"}],
                "IsTruncated": True,
                "NextContinuationToken": "1",
            },
            {
                "Contents": [{"Key": "b"}],
                "IsTruncated": True,
                "NextContinuationToken": "2",
            },
            {
                "Contents": [{"Key": "c"}],
                "IsTruncated": False,
            },
        ]
        objs = {k: {"ContentType": "text/plain", "_body": k.encode()} for k in "abc"}
        fake = _FakeS3(pages, objs)
        docs = list(S3Plugin(fake, bucket="k").iter_documents())
        assert [d.source_id for d in docs] == ["a", "b", "c"]
        assert len(fake.list_calls) == 3

    def test_content_type_filter(self):
        pages = [{"Contents": [{"Key": "a.bin"}], "IsTruncated": False}]
        objs = {
            "a.bin": {
                "ContentType": "application/octet-stream",
                "_body": b"\x00\x01\x02",
            }
        }
        docs = list(S3Plugin(_FakeS3(pages, objs), bucket="k").iter_documents())
        assert docs == []

    def test_non_utf8_skipped(self):
        pages = [{"Contents": [{"Key": "a.txt"}], "IsTruncated": False}]
        objs = {"a.txt": {"ContentType": "text/plain", "_body": b"\xff\xfe\xfd"}}
        docs = list(S3Plugin(_FakeS3(pages, objs), bucket="k").iter_documents())
        assert docs == []

    def test_rejects_empty_bucket(self):
        with pytest.raises(ValueError, match="bucket is required"):
            S3Plugin(_FakeS3([], {}), bucket="")

    def test_rejects_none_client(self):
        with pytest.raises(ValueError, match="client is required"):
            S3Plugin(None, bucket="k")  # type: ignore[arg-type]

    def test_rejects_non_positive_page_size(self):
        with pytest.raises(ValueError, match="page_size"):
            S3Plugin(_FakeS3([], {}), bucket="k", page_size=0)

    def test_prefix_forwarded(self):
        pages = [{"Contents": [], "IsTruncated": False}]
        fake = _FakeS3(pages, {})
        list(S3Plugin(fake, bucket="k", prefix="docs/").iter_documents())
        assert fake.list_calls[0]["Prefix"] == "docs/"

    def test_end_to_end_ingest(self):
        pages = [{"Contents": [{"Key": "x"}], "IsTruncated": False}]
        objs = {"x": {"ContentType": "text/plain", "_body": b"hello"}}
        store = _RecordingStore()
        written = S3Plugin(_FakeS3(pages, objs), bucket="kb").ingest(
            store, tenant_id="tenant-a"
        )
        assert written == 1
        assert store.records[0]["metadata"]["bucket"] == "kb"
        assert store.records[0]["tenant_id"] == "tenant-a"


# --- Notion -----------------------------------------------------------


class _FakeNotionDatabases:
    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self._pages = pages

    def query(
        self, *, database_id: str, start_cursor: str | None = None
    ) -> dict[str, Any]:
        return {"results": self._pages, "has_more": False, "next_cursor": None}


class _FakeNotionSearch:
    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self._pages = pages
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, *, filter: dict[str, str], start_cursor: str | None = None
    ) -> dict[str, Any]:
        self.calls.append({"filter": filter, "start_cursor": start_cursor})
        return {"results": self._pages, "has_more": False, "next_cursor": None}


class _FakeNotionBlocksChildren:
    def __init__(self, blocks_by_page: dict[str, list[dict[str, Any]]]) -> None:
        self._blocks = blocks_by_page

    def list(self, *, block_id: str, start_cursor: str | None = None) -> dict[str, Any]:
        return {
            "results": list(self._blocks.get(block_id, [])),
            "has_more": False,
            "next_cursor": None,
        }


@dataclass
class _FakeNotionBlocks:
    children: _FakeNotionBlocksChildren


@dataclass
class _FakeNotion:
    databases: _FakeNotionDatabases = field(
        default_factory=lambda: _FakeNotionDatabases([])
    )
    blocks: _FakeNotionBlocks = field(
        default_factory=lambda: _FakeNotionBlocks(_FakeNotionBlocksChildren({}))
    )
    search: _FakeNotionSearch = field(default_factory=lambda: _FakeNotionSearch([]))


def _text_block(
    block_id: str, block_type: str, text: str, children: bool = False
) -> dict[str, Any]:
    return {
        "id": block_id,
        "type": block_type,
        "has_children": children,
        block_type: {"rich_text": [{"plain_text": text}]},
    }


class TestNotion:
    def test_database_query_walks_pages_and_blocks(self):
        page_id = "page-1"
        pages = [
            {
                "id": page_id,
                "properties": {
                    "Name": {"type": "title", "title": [{"plain_text": "Hello"}]}
                },
                "last_edited_time": "2026-04-01T00:00:00Z",
            }
        ]
        blocks = {
            page_id: [
                _text_block("b1", "paragraph", "Paragraph one"),
                _text_block("b2", "heading_1", "Header two"),
                {"id": "b3", "type": "divider", "has_children": False, "divider": {}},
            ]
        }
        client = _FakeNotion(
            databases=_FakeNotionDatabases(pages),
            blocks=_FakeNotionBlocks(_FakeNotionBlocksChildren(blocks)),
        )
        plugin = NotionPlugin(client, database_ids=["db-1"])
        docs = list(plugin.iter_documents())
        assert len(docs) == 1
        assert "Paragraph one" in docs[0].text
        assert "Header two" in docs[0].text
        assert docs[0].metadata["title"] == "Hello"

    def test_nested_blocks_are_followed(self):
        page_id = "page-1"
        pages = [{"id": page_id, "properties": {}, "last_edited_time": ""}]
        blocks = {
            page_id: [
                _text_block("parent", "toggle", "parent text", children=True),
            ],
            "parent": [
                _text_block("child", "paragraph", "child text"),
            ],
        }
        client = _FakeNotion(
            databases=_FakeNotionDatabases(pages),
            blocks=_FakeNotionBlocks(_FakeNotionBlocksChildren(blocks)),
        )
        plugin = NotionPlugin(client, database_ids=["db-1"])
        docs = list(plugin.iter_documents())
        assert "parent text" in docs[0].text
        assert "child text" in docs[0].text

    def test_search_used_when_no_database_ids(self):
        pages = [{"id": "p", "properties": {}, "last_edited_time": ""}]
        client = _FakeNotion(
            search=_FakeNotionSearch(pages),
            blocks=_FakeNotionBlocks(
                _FakeNotionBlocksChildren(
                    {"p": [_text_block("b", "paragraph", "content")]}
                )
            ),
        )
        plugin = NotionPlugin(client)
        docs = list(plugin.iter_documents())
        assert len(docs) == 1
        assert client.search.calls, "search should be invoked"

    def test_max_pages_cap(self):
        pages = [
            {"id": f"p{i}", "properties": {}, "last_edited_time": ""} for i in range(5)
        ]
        blocks = {
            f"p{i}": [_text_block(f"b{i}", "paragraph", f"body {i}")] for i in range(5)
        }
        client = _FakeNotion(
            databases=_FakeNotionDatabases(pages),
            blocks=_FakeNotionBlocks(_FakeNotionBlocksChildren(blocks)),
        )
        plugin = NotionPlugin(client, database_ids=["db"], max_pages=2)
        docs = list(plugin.iter_documents())
        assert len(docs) == 2

    def test_rejects_none_client(self):
        with pytest.raises(ValueError, match="client is required"):
            NotionPlugin(None)  # type: ignore[arg-type]


# --- Google Drive -----------------------------------------------------


class _FakeDriveRequest:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def execute(self) -> Any:
        return self._payload


class _FakeFilesEndpoint:
    def __init__(
        self,
        pages: list[dict[str, Any]],
        texts: dict[str, Any],
        native_exports: dict[str, Any] | None = None,
    ) -> None:
        self._pages = pages
        self._texts = texts
        self._exports = native_exports or {}
        self.list_calls: list[dict[str, Any]] = []
        self.get_media_calls: list[str] = []
        self.export_calls: list[tuple[str, str]] = []

    def list(self, **kwargs: Any) -> _FakeDriveRequest:
        self.list_calls.append(kwargs)
        token = kwargs.get("pageToken")
        idx = int(token) if token else 0
        if idx >= len(self._pages):
            return _FakeDriveRequest({"files": [], "nextPageToken": None})
        return _FakeDriveRequest(self._pages[idx])

    def get_media(self, *, fileId: str) -> _FakeDriveRequest:
        self.get_media_calls.append(fileId)
        return _FakeDriveRequest(self._texts.get(fileId, b""))

    def export_media(self, *, fileId: str, mimeType: str) -> _FakeDriveRequest:
        self.export_calls.append((fileId, mimeType))
        return _FakeDriveRequest(self._exports.get(fileId, ""))


class _FakeDriveService:
    def __init__(self, endpoint: _FakeFilesEndpoint) -> None:
        self._endpoint = endpoint

    def files(self) -> _FakeFilesEndpoint:
        return self._endpoint


class TestGoogleDrive:
    def test_ingests_text_files(self):
        pages = [
            {
                "files": [
                    {
                        "id": "f1",
                        "name": "notes.md",
                        "mimeType": "text/markdown",
                        "modifiedTime": "2026-04-17T00:00:00Z",
                        "size": "42",
                    },
                    {
                        "id": "f2",
                        "name": "binary.png",
                        "mimeType": "image/png",
                    },
                ],
                "nextPageToken": None,
            }
        ]
        texts = {"f1": b"markdown body"}
        endpoint = _FakeFilesEndpoint(pages, texts)
        plugin = GoogleDrivePlugin(_FakeDriveService(endpoint))
        docs = list(plugin.iter_documents())
        assert len(docs) == 1
        assert docs[0].source_id == "f1"
        assert docs[0].metadata["size_bytes"] == 42
        assert endpoint.get_media_calls == ["f1"]

    def test_google_docs_are_exported(self):
        pages = [
            {
                "files": [
                    {
                        "id": "doc",
                        "name": "Doc",
                        "mimeType": "application/vnd.google-apps.document",
                    }
                ],
                "nextPageToken": None,
            }
        ]
        endpoint = _FakeFilesEndpoint(pages, {}, native_exports={"doc": "doc text"})
        docs = list(GoogleDrivePlugin(_FakeDriveService(endpoint)).iter_documents())
        assert docs[0].text == "doc text"
        assert endpoint.export_calls == [("doc", "text/plain")]

    def test_folder_and_query_clauses(self):
        pages = [{"files": [], "nextPageToken": None}]
        endpoint = _FakeFilesEndpoint(pages, {})
        plugin = GoogleDrivePlugin(
            _FakeDriveService(endpoint),
            folder_id="folder-1",
            query="modifiedTime > '2026-01-01'",
        )
        list(plugin.iter_documents())
        q = endpoint.list_calls[0]["q"]
        assert "'folder-1' in parents" in q
        assert "modifiedTime > '2026-01-01'" in q
        assert "trashed = false" in q

    def test_rejects_none_service(self):
        with pytest.raises(ValueError, match="service is required"):
            GoogleDrivePlugin(None)  # type: ignore[arg-type]

    def test_rejects_non_positive_page_size(self):
        with pytest.raises(ValueError, match="page_size"):
            GoogleDrivePlugin(
                _FakeDriveService(_FakeFilesEndpoint([], {})), page_size=0
            )

    def test_end_to_end_ingest(self):
        pages = [
            {
                "files": [
                    {
                        "id": "f1",
                        "name": "doc.txt",
                        "mimeType": "text/plain",
                    }
                ],
                "nextPageToken": None,
            }
        ]
        endpoint = _FakeFilesEndpoint(pages, {"f1": b"body"})
        store = _RecordingStore()
        written = GoogleDrivePlugin(_FakeDriveService(endpoint)).ingest(
            store, tenant_id="tenant-b"
        )
        assert written == 1
        assert store.records[0]["metadata"]["mime_type"] == "text/plain"
