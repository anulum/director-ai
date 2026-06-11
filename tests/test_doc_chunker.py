# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Document chunking contracts
"""Behavioural tests for document chunk splitting boundaries."""

from director_ai.core.retrieval.doc_chunker import ChunkConfig, split


def test_long_token_is_split_without_exceeding_configured_window() -> None:
    chunks = split("A" * 500, ChunkConfig(chunk_size=100, overlap=10))

    assert len(chunks) >= 5
    assert all(len(chunk) <= 110 for chunk in chunks)


def test_empty_document_produces_no_chunks() -> None:
    assert split("") == []


def test_short_document_is_not_fragmented() -> None:
    text = "Fits in one chunk."

    assert split(text, ChunkConfig(chunk_size=1000)) == [text]
