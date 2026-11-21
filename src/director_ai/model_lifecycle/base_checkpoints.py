# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Base Checkpoints

"""Upstream provenance records of base checkpoints.

A base checkpoint is re-audited at its first custody ingest. Its record pins
the upstream repository, revision and published digest, the digest of the
file custody ingested, its tensor-set digest (``None`` exactly for weight
format ``none``), its weight licence and whether commercial use and
derivatives were found clear.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ._checks import require_bool, require_choice, require_sha256, require_text
from .errors import CustodyRecordError

_GIT_OID = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


_PUBLISHED_DIGEST = {
    "sha256": re.compile(r"[0-9a-f]{64}"),
    "sha384": re.compile(r"[0-9a-f]{96}"),
    "git_blob_oid": _GIT_OID,
    "none": re.compile(r""),
}


@dataclass(frozen=True, slots=True)
class BaseCheckpointRecord:
    """Upstream provenance of a base checkpoint, audited at first ingest.

    Attributes
    ----------
    schema : Literal["base-checkpoint-record.v1"]
        Record schema.
    repository_url : str
        HTTPS location of the upstream repository.
    upstream_revision_kind : Literal["git_commit_sha", "doi_version"]
        How the upstream revision is named.
    upstream_revision : str
        The pinned upstream revision.
    upstream_published_digest_kind : Literal["sha256", "sha384", "git_blob_oid", "none"]
        Kind of digest the upstream publishes.
    upstream_published_digest : str
        That digest, or ``""`` when the upstream publishes none.
    custody_ingested_file_sha256 : str
        SHA-256 of the file custody ingested.
    tensor_set_digest : str or None
        Tensor-set digest, ``None`` exactly for weight format ``none``.
    weight_spdx_expression : str
        SPDX expression of the weights' licence.
    commercial_and_derivative_clear : bool
        Whether commercial use and derivatives were found clear.
    """

    schema: Literal["base-checkpoint-record.v1"]
    repository_url: str
    upstream_revision_kind: Literal["git_commit_sha", "doi_version"]
    upstream_revision: str
    upstream_published_digest_kind: Literal["sha256", "sha384", "git_blob_oid", "none"]
    upstream_published_digest: str
    custody_ingested_file_sha256: str
    tensor_set_digest: str | None
    weight_spdx_expression: str
    commercial_and_derivative_clear: bool


def validate_base_checkpoint_record(
    record: BaseCheckpointRecord,
) -> BaseCheckpointRecord:
    """Check a base checkpoint record's fields.

    Parameters
    ----------
    record : BaseCheckpointRecord
        The record to check.

    Returns
    -------
    BaseCheckpointRecord
        The same record.

    Raises
    ------
    CustodyRecordError
        If a field is malformed.
    MalformedDigestError
        If a SHA-256 field is malformed.
    """
    if not isinstance(record, BaseCheckpointRecord):
        raise CustodyRecordError("a BaseCheckpointRecord is required")
    require_choice(
        record.schema, "schema", {"base-checkpoint-record.v1"}, CustodyRecordError
    )
    url = require_text(record.repository_url, "repository_url", CustodyRecordError)
    if not url.startswith("https://"):
        raise CustodyRecordError("repository_url must be an https URL")
    kind = require_choice(
        record.upstream_revision_kind,
        "upstream_revision_kind",
        {"git_commit_sha", "doi_version"},
        CustodyRecordError,
    )
    revision = require_text(
        record.upstream_revision, "upstream_revision", CustodyRecordError
    )
    if kind == "git_commit_sha" and _GIT_OID.fullmatch(revision) is None:
        raise CustodyRecordError("upstream_revision must be a full git commit id")
    digest_kind = require_choice(
        record.upstream_published_digest_kind,
        "upstream_published_digest_kind",
        _PUBLISHED_DIGEST,
        CustodyRecordError,
    )
    published = record.upstream_published_digest
    if (
        not isinstance(published, str)
        or _PUBLISHED_DIGEST[digest_kind].fullmatch(published) is None
    ):
        raise CustodyRecordError(
            f"upstream_published_digest does not match kind {digest_kind!r}"
        )
    require_sha256(record.custody_ingested_file_sha256, "custody_ingested_file_sha256")
    if record.tensor_set_digest is not None:
        require_sha256(record.tensor_set_digest, "tensor_set_digest")
    require_text(
        record.weight_spdx_expression, "weight_spdx_expression", CustodyRecordError
    )
    require_bool(
        record.commercial_and_derivative_clear,
        "commercial_and_derivative_clear",
        CustodyRecordError,
    )
    return record
