# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Document Chunker

"""Document chunking: recursive character splitter + semantic splitter.

The character splitter uses no external deps. The semantic splitter
detects topic shifts via sentence embedding similarity and splits
between dissimilar consecutive sentences.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkConfig:
    chunk_size: int = 512
    overlap: int = 64
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", "")
    semantic: bool = False
    similarity_threshold: float = 0.3


_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def split(text: str, config: ChunkConfig | None = None) -> list[str]:
    """Split text into overlapping chunks.

    When ``config.semantic=True``, detects topic shifts via sentence
    embedding cosine similarity and splits between dissimilar sentences.
    Falls back to character-level recursive splitting otherwise.
    """
    if not text:
        return []
    cfg = config or ChunkConfig()
    if len(text) <= cfg.chunk_size:
        return [text]
    if cfg.semantic:
        return _semantic_split(text, cfg)
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


def _semantic_split(text: str, cfg: ChunkConfig) -> list[str]:
    """Split on topic shifts detected by sentence embedding similarity.

    Embeds each sentence, computes cosine similarity between consecutive
    pairs, splits where similarity drops below threshold, then merges
    small groups up to chunk_size.
    """
    import numpy as np

    sentences = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    if len(sentences) <= 1:
        return _recursive_split(text, cfg.separators, cfg.chunk_size, cfg.overlap)

    embeddings = _embed_sentences(sentences)
    if embeddings is None:
        return _recursive_split(text, cfg.separators, cfg.chunk_size, cfg.overlap)

    # Cosine similarity between consecutive sentences
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = embeddings / norms

    similarities = np.sum(normed[:-1] * normed[1:], axis=1)

    # Find split points where similarity drops below threshold
    split_indices = [0]
    for i, sim in enumerate(similarities):
        if sim < cfg.similarity_threshold:
            split_indices.append(i + 1)

    # Build groups from split points
    groups: list[list[str]] = []
    for idx in range(len(split_indices)):
        start = split_indices[idx]
        end = split_indices[idx + 1] if idx + 1 < len(split_indices) else len(sentences)
        groups.append(sentences[start:end])

    # Merge small groups up to chunk_size
    chunks: list[str] = []
    current = ""
    for group in groups:
        group_text = " ".join(group)
        candidate = current + " " + group_text if current else group_text
        if len(candidate) <= cfg.chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(group_text) > cfg.chunk_size:
                chunks.extend(
                    _recursive_split(
                        group_text, cfg.separators, cfg.chunk_size, cfg.overlap
                    )
                )
                current = ""
            else:
                current = group_text
    if current:
        chunks.append(current)

    return chunks


def _embed_sentences(sentences: list[str]):
    """Embed sentences using sentence-transformers. Returns None if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(sentences, show_progress_bar=False)
    except ImportError:
        return None
