# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Content-Addressed Dataset Fingerprints

"""Content-addressed dataset fingerprints for training provenance.

``TrainingJobSpec.dataset_hash`` fingerprints the dataset *URI string*, so two
different datasets published under the same URI collide and a re-uploaded
dataset keeps a stale hash. This module fingerprints the dataset *content*
where the bytes are reachable (local files and directories), and degrades
honestly where they are not: remote URIs and missing local paths fall back to
a URI-string digest that is explicitly labelled ``hash_source="uri-only"``
with a machine-readable ``reason``. A URI digest is never presented as a
content digest.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_CHUNK_BYTES = 1 << 20
_CONTENT = "content"
_URI_ONLY = "uri-only"
_REASON_REMOTE = "remote-uri-without-reader"
_REASON_MISSING = "missing-local-path"

RemoteReader = Callable[[str], bytes]
"""Callable returning the raw bytes behind a remote dataset URI."""


@dataclass(frozen=True)
class DatasetFingerprint:
    """Provenance fingerprint for one training dataset.

    Attributes
    ----------
    uri : str
        The dataset URI or local path as supplied by the caller.
    digest : str
        Full SHA-256 hex digest. Of the dataset bytes when
        ``hash_source="content"``; of the URI string when
        ``hash_source="uri-only"``.
    hash_source : str
        ``"content"`` or ``"uri-only"`` — what the digest covers.
    reason : str
        Why a uri-only fallback happened; empty for content digests.
    byte_size : int
        Total content bytes hashed; 0 for uri-only fingerprints.
    file_count : int
        Files covered by a content digest; 0 for uri-only fingerprints.
    algorithm : str
        Digest algorithm name (always ``"sha256"``).
    """

    uri: str
    digest: str
    hash_source: str
    reason: str = ""
    byte_size: int = 0
    file_count: int = 0
    algorithm: str = "sha256"

    @property
    def is_content_addressed(self) -> bool:
        """Return whether the digest covers dataset bytes, not the URI."""
        return self.hash_source == _CONTENT

    @property
    def short_digest(self) -> str:
        """Return the 16-character short form used in job provenance."""
        return self.digest[:16]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of this fingerprint."""
        return {
            "uri": self.uri,
            "digest": self.digest,
            "hash_source": self.hash_source,
            "reason": self.reason,
            "byte_size": self.byte_size,
            "file_count": self.file_count,
            "algorithm": self.algorithm,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DatasetFingerprint:
        """Rebuild a fingerprint from its serialised dictionary shape."""
        return cls(
            uri=str(payload["uri"]),
            digest=str(payload["digest"]),
            hash_source=str(payload["hash_source"]),
            reason=str(payload.get("reason", "")),
            byte_size=int(payload.get("byte_size", 0)),
            file_count=int(payload.get("file_count", 0)),
            algorithm=str(payload.get("algorithm", "sha256")),
        )


def fingerprint_dataset(
    uri: str,
    *,
    remote_reader: RemoteReader | None = None,
    strict: bool = True,
) -> DatasetFingerprint:
    """Fingerprint the dataset behind *uri*.

    Parameters
    ----------
    uri : str
        Local path, ``file://`` URI, or remote URI (``gs://``, ``s3://``, …).
    remote_reader : RemoteReader | None
        Optional callable that fetches the raw bytes behind a remote URI so
        remote datasets can be content-addressed. Without it, remote URIs
        produce a labelled uri-only fingerprint.
    strict : bool
        When true (default), a missing local path raises
        :class:`FileNotFoundError`. When false, it degrades to a labelled
        uri-only fingerprint — the mode job builders use so that dry-run
        requests for not-yet-staged datasets still carry honest provenance.

    Returns
    -------
    DatasetFingerprint
        Content fingerprint when the bytes are reachable, else a uri-only
        fingerprint whose ``reason`` explains the fallback.
    """
    if not uri:
        raise ValueError("dataset uri is required")

    if _is_remote_uri(uri):
        if remote_reader is None:
            return _uri_only(uri, reason=_REASON_REMOTE)
        payload = remote_reader(uri)
        digest = hashlib.sha256(payload).hexdigest()
        return DatasetFingerprint(
            uri=uri,
            digest=digest,
            hash_source=_CONTENT,
            byte_size=len(payload),
            file_count=1,
        )

    path = _local_path(uri)
    if not path.exists():
        if strict:
            raise FileNotFoundError(f"dataset path not found: {path}")
        return _uri_only(uri, reason=_REASON_MISSING)
    if path.is_dir():
        return _fingerprint_directory(uri, path)
    return _fingerprint_file(uri, path)


def _fingerprint_file(uri: str, path: Path) -> DatasetFingerprint:
    digest = hashlib.sha256()
    byte_size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
            byte_size += len(chunk)
    return DatasetFingerprint(
        uri=uri,
        digest=digest.hexdigest(),
        hash_source=_CONTENT,
        byte_size=byte_size,
        file_count=1,
    )


def _fingerprint_directory(uri: str, path: Path) -> DatasetFingerprint:
    files = sorted(
        (entry for entry in path.rglob("*") if entry.is_file()),
        key=lambda entry: entry.relative_to(path).as_posix(),
    )
    if not files:
        raise ValueError(f"dataset directory is empty: {path}")
    aggregate = hashlib.sha256()
    byte_size = 0
    for entry in files:
        record = _fingerprint_file(uri, entry)
        relative = entry.relative_to(path).as_posix()
        aggregate.update(f"{relative}\x00{record.digest}\n".encode())
        byte_size += record.byte_size
    return DatasetFingerprint(
        uri=uri,
        digest=aggregate.hexdigest(),
        hash_source=_CONTENT,
        byte_size=byte_size,
        file_count=len(files),
    )


def _uri_only(uri: str, *, reason: str) -> DatasetFingerprint:
    return DatasetFingerprint(
        uri=uri,
        digest=hashlib.sha256(uri.encode("utf-8")).hexdigest(),
        hash_source=_URI_ONLY,
        reason=reason,
    )


def _is_remote_uri(uri: str) -> bool:
    parsed = urlparse(uri)
    return bool(parsed.scheme and parsed.scheme != "file" and parsed.netloc)


def _local_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(parsed.path)
    return Path(uri)
