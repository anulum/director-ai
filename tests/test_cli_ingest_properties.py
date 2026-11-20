# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — CLI ingest properties
"""Bounded file corpus for the public ingest command."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from director_ai.cli import main

_INGEST_LIMIT = 100 * 1024 * 1024
_TEXT = st.text(alphabet="abc Àé中žΩ", min_size=1, max_size=48)


@contextmanager
def _local_ingest_output() -> Iterator[StringIO]:
    """Capture public CLI output with the local vector backend configured."""
    with (
        pytest.MonkeyPatch.context() as environment,
        redirect_stdout(StringIO()) as output,
    ):
        environment.setenv("DIRECTOR_MODE", "grounded")
        environment.setenv("DIRECTOR_VECTOR_BACKEND", "memory")
        environment.setenv("DIRECTOR_RERANKER_ENABLED", "false")
        environment.delenv("DIRECTOR_REDIS_URL", raising=False)
        yield output


@settings(max_examples=8, deadline=2000, database=None)
@given(extra_bytes=st.integers(min_value=1, max_value=4096), text=_TEXT)
def test_ingest_skips_oversized_files_and_keeps_valid_neighbour(
    extra_bytes: int, text: str
) -> None:
    """Exclude a real file beyond the cap while admitting a small neighbour."""
    with TemporaryDirectory() as directory:
        root = Path(directory)
        large = root / "too-large.txt"
        with large.open("wb") as stream:
            stream.write(b"oversized document")
            stream.truncate(_INGEST_LIMIT + extra_bytes)
        assert large.stat().st_size == _INGEST_LIMIT + extra_bytes
        (root / "valid.txt").write_text(f"Valid document {text}", encoding="utf-8")
        with _local_ingest_output() as output:
            main(["ingest", str(root)])

    assert f"Warning: skipping {large}" in output.getvalue()
    assert "Ingested 1 chunks from 2 file(s)." in output.getvalue()


@settings(max_examples=12, deadline=2000, database=None)
@given(
    documents=st.lists(_TEXT, min_size=1, max_size=4),
    malformed=st.sampled_from(("{", "[]", "null")),
)
def test_ingest_refuses_malformed_batch_then_accepts_corrected_batch(
    documents: list[str], malformed: str
) -> None:
    """Refuse a malformed JSONL row and accept the complete corrected input."""
    with TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "notes.jsonl"
        valid = [json.dumps({"text": f"Document {text}"}) for text in documents]
        path.write_text("\n".join([*valid, malformed]), encoding="utf-8")
        with (
            _local_ingest_output() as refused,
            pytest.raises(ValueError, match=f"line {len(valid) + 1}:"),
        ):
            main(["ingest", str(root)])

        path.write_text("\n".join(valid), encoding="utf-8")
        with _local_ingest_output() as retried:
            main(["ingest", str(root)])

    assert "Ingested" not in refused.getvalue()
    assert f"Ingested {len(valid)} chunks from 1 file(s)." in retried.getvalue()
