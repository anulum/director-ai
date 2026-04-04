# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — CLI Document Ingestion Command
"""CLI subcommand for document ingestion.

Extracted from cli.py to reduce module size.
"""

from __future__ import annotations

import json
import sys

_INGEST_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


def _cmd_ingest(args: list[str]) -> None:
    """Ingest files or directories into a VectorGroundTruthStore.

    Supported formats: ``.txt``, ``.md``, ``.json``/``.jsonl``,
    ``.pdf``, ``.docx``, ``.html``, ``.csv``.
    PDF/DOCX/HTML require ``pip install director-ai[ingestion]``.
    Directories are walked recursively for supported file types.
    """
    if not args:
        print(
            "Usage: director-ai ingest <file-or-dir> "
            "[--persist <dir>] [--chunk-size <tokens>]",
        )
        sys.exit(1)

    import os
    from pathlib import Path

    input_path = args[0]
    persist_dir = None
    chunk_size = 500
    if "--persist" in args:
        idx = args.index("--persist")
        if idx + 1 < len(args):  # pragma: no branch
            persist_dir = args[idx + 1]
    if "--chunk-size" in args:
        idx = args.index("--chunk-size")
        if idx + 1 < len(args):  # pragma: no branch
            chunk_size = int(args[idx + 1])
    if chunk_size <= 0:
        print(f"Error: --chunk-size must be > 0, got {chunk_size}")
        sys.exit(1)

    if not os.path.exists(input_path):
        print(f"Error: path not found: {input_path}")
        sys.exit(1)

    from director_ai.core.config import DirectorConfig

    cfg = DirectorConfig.from_env()
    if persist_dir:
        cfg.vector_backend = "chroma"
        cfg.chroma_persist_dir = persist_dir
    store = cfg.build_store()

    text_exts = {".txt", ".md", ".json", ".jsonl", ".xml", ".markdown"}
    parsed_exts = {".pdf", ".docx", ".html", ".htm", ".csv"}
    supported_exts = text_exts | parsed_exts

    def _collect_files(path: str) -> list[Path]:
        p = Path(path)
        if p.is_file():
            return [p]
        return sorted(
            f
            for f in p.rglob("*")
            if f.is_file() and f.suffix.lower() in supported_exts
        )

    def _chunk_paragraphs(text: str, max_tokens: int) -> list[str]:
        paragraphs = text.split("\n\n")
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for para in paragraphs:
            para = para.strip()
            if not para:  # pragma: no cover — empty paragraphs from text.split
                continue
            word_count = len(para.split())
            if current_len + word_count > max_tokens and current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            current.append(para)
            current_len += word_count
        if current:  # pragma: no branch
            chunks.append("\n\n".join(current))
        return chunks

    def _read_file(path: Path) -> list[str]:
        size = path.stat().st_size
        if size > _INGEST_MAX_FILE_SIZE:
            print(f"Warning: skipping {path} ({size / 1024 / 1024:.1f} MB > limit)")
            return []

        ext = path.suffix.lower()

        # PDF, DOCX, HTML, CSV — delegate to doc_parser (binary read)
        if ext in parsed_exts:
            from director_ai.core.retrieval.doc_parser import parse

            try:
                raw = path.read_bytes()
                text = parse(raw, path.name)
            except ImportError as exc:
                print(f"Warning: skipping {path} ({exc})")
                return []
            if not text.strip():
                return []
            return _chunk_paragraphs(text, chunk_size)

        text = path.read_text(encoding="utf-8", errors="replace")

        if ext in (".json", ".jsonl"):
            docs = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    doc = data.get("text", data.get("content", ""))
                    if doc:  # pragma: no branch
                        docs.append(doc)
                except json.JSONDecodeError:
                    pass
            return docs
        return _chunk_paragraphs(text, chunk_size)

    files = _collect_files(input_path)
    if not files:
        print(f"No supported files found in {input_path}")
        sys.exit(1)

    texts: list[str] = []
    for f in files:
        texts.extend(_read_file(f))

    count = store.ingest(texts)
    print(f"Ingested {count} chunks from {len(files)} file(s).")
    if persist_dir:
        print(f"Persisted to: {persist_dir}")
    else:
        print("(in-memory only — use --persist <dir> to save)")
