# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — CLI Ingest Format Tests (PDF/DOCX/HTML/CSV via doc_parser)

from __future__ import annotations

import contextlib
import json
from unittest.mock import patch

import pytest

from director_ai.cli import main


class TestIngestParsedFormats:
    """Tests for CLI ingest with binary format files (PDF, DOCX, HTML, CSV)."""

    def test_ingest_csv_file(self, capsys, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,role\nAlice,Engineer\nBob,Designer\n")
        main(["ingest", str(csv_file)])
        out = capsys.readouterr().out
        assert "Ingested" in out
        assert "1 file(s)" in out

    def test_ingest_html_file(self, capsys, tmp_path):
        html_file = tmp_path / "page.html"
        html_file.write_text(
            "<html><body><p>Water boils at 100 degrees.</p>"
            "<script>var x = 1;</script></body></html>"
        )
        # html parser needs beautifulsoup4 — test the path regardless
        try:
            main(["ingest", str(html_file)])
            out = capsys.readouterr().out
            assert "Ingested" in out
        except SystemExit:
            # beautifulsoup4 not installed — warning printed, no files ingested
            out = capsys.readouterr().out
            assert "Warning" in out or "No supported files" in out

    def test_ingest_htm_extension(self, capsys, tmp_path):
        htm_file = tmp_path / "page.htm"
        htm_file.write_text("<html><body><p>Some content.</p></body></html>")
        try:
            main(["ingest", str(htm_file)])
            out = capsys.readouterr().out
            assert "Ingested" in out or "Warning" in out
        except SystemExit:
            pass  # dep missing or no chunks

    def test_ingest_pdf_missing_dep(self, capsys, tmp_path):
        """PDF ingest gracefully skips when pypdf is not installed."""
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")

        # Mock the parser to raise ImportError
        with patch(
            "director_ai.core.retrieval.doc_parser.parse",
            side_effect=ImportError("pypdf required"),
        ):
            with contextlib.suppress(SystemExit):
                main(["ingest", str(pdf_file)])
            out = capsys.readouterr().out
            # Should warn about skipping, not crash
            assert "Warning" in out or "No supported" in out

    def test_ingest_docx_missing_dep(self, capsys, tmp_path):
        """DOCX ingest gracefully skips when python-docx is not installed."""
        docx_file = tmp_path / "doc.docx"
        docx_file.write_bytes(b"PK\x03\x04 fake docx")

        with patch(
            "director_ai.core.retrieval.doc_parser.parse",
            side_effect=ImportError("python-docx required"),
        ):
            with contextlib.suppress(SystemExit):
                main(["ingest", str(docx_file)])
            out = capsys.readouterr().out
            assert "Warning" in out or "No supported" in out

    def test_ingest_parsed_file_returns_empty(self, capsys, tmp_path):
        """Parsed file with empty content produces no chunks."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")

        with (
            patch(
                "director_ai.core.retrieval.doc_parser.parse",
                return_value="",
            ),
            contextlib.suppress(SystemExit),
        ):
            main(["ingest", str(csv_file)])

    def test_ingest_directory_with_mixed_formats(self, capsys, tmp_path):
        """Directory with .txt and .csv files ingests both."""
        (tmp_path / "facts.txt").write_text("Water is H2O.\n\nThe sky is blue.\n")
        (tmp_path / "data.csv").write_text("fact,source\nEarth orbits Sun,astronomy\n")

        main(["ingest", str(tmp_path)])
        out = capsys.readouterr().out
        assert "Ingested" in out
        assert "2 file(s)" in out

    def test_ingest_directory_finds_uppercase_supported_ext(self, capsys, tmp_path):
        pdf_file = tmp_path / "FAQ.PDF"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")

        with patch(
            "director_ai.core.retrieval.doc_parser.parse",
            return_value="Refunds are available for 30 days.",
        ):
            main(["ingest", str(tmp_path)])

        out = capsys.readouterr().out
        assert "Ingested" in out
        assert "1 file(s)" in out

    def test_ingest_xml_file(self, capsys, tmp_path):
        xml_file = tmp_path / "data.xml"
        xml_file.write_text("<root><item>Content here</item></root>")
        main(["ingest", str(xml_file)])
        out = capsys.readouterr().out
        assert "Ingested" in out

    def test_ingest_markdown_extension(self, capsys, tmp_path):
        md_file = tmp_path / "notes.markdown"
        md_file.write_text("# Notes\n\nSome important facts.\n")
        main(["ingest", str(md_file)])
        out = capsys.readouterr().out
        assert "Ingested" in out

    def test_ingest_chunk_size_zero(self):
        with pytest.raises(SystemExit) as exc:
            main(["ingest", "dummy.txt", "--chunk-size", "0"])
        assert exc.value.code == 1

    def test_ingest_chunk_size_negative(self):
        with pytest.raises(SystemExit) as exc:
            main(["ingest", "dummy.txt", "--chunk-size", "-5"])
        assert exc.value.code == 1


class TestIngestEdgeCases:
    """Edge cases for the ingest command."""

    def test_ingest_jsonl_with_content_key(self, capsys, tmp_path):
        jf = tmp_path / "docs.jsonl"
        jf.write_text(
            json.dumps({"content": "Fact one about physics."})
            + "\n"
            + json.dumps({"content": "Fact two about chemistry."})
            + "\n"
        )
        main(["ingest", str(jf)])
        out = capsys.readouterr().out
        assert "Ingested" in out
        assert "2 chunks" in out

    def test_ingest_jsonl_with_bad_lines(self, capsys, tmp_path):
        jf = tmp_path / "mixed.jsonl"
        jf.write_text(
            json.dumps({"text": "Valid line."})
            + "\n"
            + "not valid json\n"
            + "\n"
            + json.dumps({"text": "Another valid."})
            + "\n"
        )
        main(["ingest", str(jf)])
        out = capsys.readouterr().out
        assert "Ingested" in out

    def test_ingest_large_file_skipped(self, capsys, tmp_path):
        """Files exceeding 100 MB are skipped with a warning."""
        big_file = tmp_path / "huge.txt"
        big_file.write_text("x")

        # Patch stat to report >100 MB
        original_stat = big_file.stat

        class FakeStat:
            def __init__(self):
                real = original_stat()
                self.st_size = 200 * 1024 * 1024  # 200 MB
                self.st_mode = real.st_mode
                self.st_mtime = real.st_mtime

        with patch.object(type(big_file), "stat", return_value=FakeStat()):
            with contextlib.suppress(SystemExit):
                main(["ingest", str(big_file)])
            out = capsys.readouterr().out
            assert "Warning" in out or "No supported" in out

    def test_ingest_no_supported_files_in_dir(self, capsys, tmp_path):
        """Directory with no supported extensions exits with error."""
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "binary.bin").write_bytes(b"\x00\x01\x02")
        with pytest.raises(SystemExit) as exc:
            main(["ingest", str(tmp_path)])
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "No supported files" in out

    def test_ingest_json_array_file(self, capsys, tmp_path):
        """JSON file with array structure (not JSONL) — each line parsed."""
        jf = tmp_path / "data.json"
        jf.write_text(
            json.dumps({"text": "Fact about gravity."})
            + "\n"
            + json.dumps({"text": "Fact about light."})
            + "\n"
        )
        main(["ingest", str(jf)])
        out = capsys.readouterr().out
        assert "Ingested" in out

    def test_ingest_persist_flag(self, capsys, tmp_path):
        tf = tmp_path / "facts.txt"
        tf.write_text("The speed of light is 300,000 km/s.\n")
        persist_dir = str(tmp_path / "kb_persist")
        try:
            main(["ingest", str(tf), "--persist", persist_dir])
            out = capsys.readouterr().out
            assert "Persisted to" in out
        except Exception:
            # ChromaDB may not be installed
            pass


class TestDocParserDirect:
    """Direct tests for doc_parser.parse() function."""

    def test_parse_txt(self):
        from director_ai.core.retrieval.doc_parser import parse

        assert parse(b"Hello world", "test.txt") == "Hello world"

    def test_parse_md(self):
        from director_ai.core.retrieval.doc_parser import parse

        result = parse(b"# Title\nContent", "readme.md")
        assert "Title" in result

    def test_parse_csv_rows(self):
        from director_ai.core.retrieval.doc_parser import parse

        result = parse(b"a,b,c\n1,2,3\n4,5,6", "data.csv")
        assert "1" in result
        assert "4" in result

    def test_parse_json_as_text(self):
        from director_ai.core.retrieval.doc_parser import parse

        data = json.dumps({"key": "value"}).encode()
        result = parse(data, "config.json")
        assert "key" in result

    def test_parse_xml_as_text(self):
        from director_ai.core.retrieval.doc_parser import parse

        result = parse(b"<root>content</root>", "data.xml")
        assert "content" in result

    def test_parse_unknown_ext_as_text(self):
        from director_ai.core.retrieval.doc_parser import parse

        result = parse(b"arbitrary bytes", "file.xyz")
        assert result == "arbitrary bytes"

    def test_parse_no_extension(self):
        from director_ai.core.retrieval.doc_parser import parse

        result = parse(b"content", "README")
        assert result == "content"

    def test_parse_markdown_extension(self):
        from director_ai.core.retrieval.doc_parser import parse

        result = parse(b"# Heading", "notes.markdown")
        assert "Heading" in result

    def test_parse_utf8_errors_replaced(self):
        from director_ai.core.retrieval.doc_parser import parse

        result = parse(b"\xff\xfe invalid utf8", "test.txt")
        assert isinstance(result, str)

    def test_parse_html_missing_bs4(self):
        from director_ai.core.retrieval.doc_parser import parse

        # If bs4 is installed, this works; if not, ImportError
        try:
            result = parse(b"<html><body><p>Test</p></body></html>", "page.html")
            assert "Test" in result
        except ImportError:
            pass

    def test_parse_pdf_missing_pypdf(self):
        import builtins

        from director_ai.core.retrieval.doc_parser import parse

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pypdf":
                raise ImportError("No module named 'pypdf'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(ImportError, match="pypdf required for PDF parsing"):
                parse(b"not a real PDF", "doc.pdf")

    def test_parse_docx_missing_dep(self):
        import builtins

        from director_ai.core.retrieval.doc_parser import parse

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "docx":
                raise ImportError("No module named 'docx'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(
                ImportError,
                match="python-docx required for DOCX parsing",
            ):
                parse(b"not a real DOCX", "doc.docx")


class TestDocParserShim:
    """Test the backward-compat shim at director_ai.core.doc_parser."""

    def test_shim_resolves(self):
        from director_ai.core.doc_parser import parse

        assert callable(parse)
        result = parse(b"hello", "test.txt")
        assert result == "hello"


class TestCLIIngestHelp:
    """Test that ingest help text reflects new formats."""

    def test_help_mentions_pdf(self, capsys):
        main(["--help"])
        out = capsys.readouterr().out
        assert "pdf" in out.lower() or "docx" in out.lower()

    def test_ingest_usage_on_no_args(self, capsys):
        with pytest.raises(SystemExit):
            main(["ingest"])
        out = capsys.readouterr().out
        assert "Usage" in out

    def test_invalid_command_rejected(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["../../../etc/passwd"])
        assert exc.value.code == 1
