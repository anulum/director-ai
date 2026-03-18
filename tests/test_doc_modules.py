# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Â© Concepts 1996â€“2026 Miroslav Ĺ otek. All rights reserved.
# Â© Code 2020â€“2026 Miroslav Ĺ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI â€” Document Module Tests (chunker, parser, registry)

from __future__ import annotations

import pytest

from director_ai.core.doc_chunker import ChunkConfig, split
from director_ai.core.doc_parser import parse
from director_ai.core.doc_registry import DocRegistry


class TestChunker:
    def test_empty(self):
        assert split("") == []

    def test_short_text(self):
        assert split("Hello world.", ChunkConfig(chunk_size=100)) == ["Hello world."]

    def test_splits_on_sentence(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = split(text, ChunkConfig(chunk_size=30, overlap=0))
        assert len(chunks) >= 2

    def test_overlap(self):
        text = "A" * 100 + " " + "B" * 100
        chunks = split(text, ChunkConfig(chunk_size=110, overlap=10))
        assert len(chunks) >= 2

    def test_respects_max_size(self):
        text = "word " * 200
        chunks = split(text, ChunkConfig(chunk_size=50, overlap=0))
        for chunk in chunks:
            assert len(chunk) <= 60  # some slack for separator

    def test_unicode(self):
        text = "HĂ©llo wĂ¶rld. ĂśnĂŻcĂ¶dĂ© text here. More sentences follow."
        chunks = split(text, ChunkConfig(chunk_size=30, overlap=0))
        assert len(chunks) >= 1

    def test_single_long_word(self):
        text = "A" * 1000
        chunks = split(text, ChunkConfig(chunk_size=100, overlap=10))
        assert len(chunks) >= 5


class TestParser:
    def test_txt(self):
        assert parse(b"Hello world", "test.txt") == "Hello world"

    def test_md(self):
        assert parse(b"# Heading\nBody", "doc.md") == "# Heading\nBody"

    def test_csv(self):
        result = parse(b"name,age\nAlice,30\nBob,25", "data.csv")
        assert "Alice" in result
        assert "30" in result

    def test_unknown_extension(self):
        result = parse(b"some content", "file.xyz")
        assert result == "some content"

    def test_utf8_decode(self):
        result = parse("HĂ©llo".encode(), "test.txt")
        assert "HĂ©llo" in result

    def test_pdf_missing_dep(self):
        import contextlib

        with contextlib.suppress(ImportError, Exception):
            parse(b"not a pdf", "test.pdf")

    def test_docx_missing_dep(self):
        import contextlib

        with contextlib.suppress(ImportError, Exception):
            parse(b"not a docx", "test.docx")


class TestRegistry:
    def test_register_and_get(self):
        reg = DocRegistry()
        rec = reg.register("d1", "test.txt", "t1", ["d1:chunk:0", "d1:chunk:1"])
        assert rec.doc_id == "d1"
        assert rec.chunk_count == 2
        fetched = reg.get("d1", "t1")
        assert fetched is not None
        assert fetched.source == "test.txt"

    def test_tenant_isolation(self):
        reg = DocRegistry()
        reg.register("d1", "f.txt", "t1", ["c0"])
        assert reg.get("d1", "t1") is not None
        assert reg.get("d1", "t2") is None

    def test_list_for_tenant(self):
        reg = DocRegistry()
        reg.register("d1", "a.txt", "t1", ["c0"])
        reg.register("d2", "b.txt", "t1", ["c1"])
        reg.register("d3", "c.txt", "t2", ["c2"])
        assert len(reg.list_for_tenant("t1")) == 2
        assert len(reg.list_for_tenant("t2")) == 1

    def test_delete(self):
        reg = DocRegistry()
        reg.register("d1", "f.txt", "t1", ["c0"])
        deleted = reg.delete("d1")
        assert deleted is not None
        assert reg.get("d1", "t1") is None

    def test_update(self):
        reg = DocRegistry()
        reg.register("d1", "f.txt", "t1", ["c0"])
        reg.update("d1", ["c0", "c1", "c2"])
        rec = reg.get("d1", "t1")
        assert rec.chunk_count == 3

    def test_duplicate_register_raises(self):
        reg = DocRegistry()
        reg.register("d1", "f.txt", "t1", ["c0"])
        with pytest.raises(ValueError, match="already registered"):
            reg.register("d1", "g.txt", "t1", ["c1"])

    def test_exists(self):
        reg = DocRegistry()
        assert not reg.exists("d1")
        reg.register("d1", "f.txt", "t1", ["c0"])
        assert reg.exists("d1")

    def test_count(self):
        reg = DocRegistry()
        assert reg.count == 0
        reg.register("d1", "f.txt", "t1", ["c0"])
        assert reg.count == 1
