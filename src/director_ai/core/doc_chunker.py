# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Document Chunker
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""Recursive character splitter with sentence-boundary snapping.

No external dependencies. Splits text into overlapping chunks
suitable for embedding, preferring natural break points.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkConfig:
    chunk_size: int = 512
    overlap: int = 64
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", "")


def split(text: str, config: ChunkConfig | None = None) -> list[str]:
    """Split text into overlapping chunks.

    Tries separators in order, preferring paragraph breaks over
    sentence breaks over word breaks. Carries ``config.overlap``
    characters forward between consecutive chunks.
    """
    if not text:
        return []
    cfg = config or ChunkConfig()
    if len(text) <= cfg.chunk_size:
        return [text]
    return _recursive_split(text, cfg.separators, cfg.chunk_size, cfg.overlap)


def _recursive_split(
    text: str,
    separators: tuple[str, ...],
    chunk_size: int,
    overlap: int,
) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    sep = ""
    for candidate in separators:
        if candidate and candidate in text:
            sep = candidate
            break

    if not sep:
        return _force_split(text, chunk_size, overlap)

    segments = text.split(sep)
    chunks: list[str] = []
    current = ""

    for segment in segments:
        candidate = current + sep + segment if current else segment
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(segment) > chunk_size:
                remaining_seps = tuple(s for s in separators if s != sep)
                if remaining_seps:
                    sub = _recursive_split(segment, remaining_seps, chunk_size, overlap)
                else:
                    sub = _force_split(segment, chunk_size, overlap)
                chunks.extend(sub)
                current = ""
            else:
                current = segment

    if current:
        chunks.append(current)

    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    return _apply_overlap(chunks, overlap)


def _force_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Character-level split when no separator works."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap if overlap > 0 else end
        if start >= end:
            break
    return chunks


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Carry trailing characters from each chunk into the next."""
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        merged = prev_tail + chunks[i]
        result.append(merged)
    return result
