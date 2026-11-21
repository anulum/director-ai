# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Custody Records

"""Custody ingest records, base checkpoints and preregistration approvals.

Blob custody writes one :class:`CustodyIngestRecord` per digest. Its
``custody_ingest_seq`` is the only clock for preregistration and taint
ordering, and its submitting role is the proposer context: lineage is never
taken from a request.

Every record has exactly the same 17 keys whatever its kind. The kinds
``split``, ``split_pool``, ``corpus`` and ``grader_bundle`` are sealed: they
never leave custody. The sealed set is fixed here and is never configured.

**Disclosure.** On a host where every seat runs as one OS user, the
submitting role is recorded procedurally
(``procedural_shared_os_user_until_d8``): a seat's identity key can be read
by every process of that user, so the lineage rule is exactly as strong as
that procedure. The record keeps the submitting uid and pid beside the
claimed role. Per-role OS accounts, or keys held outside the shared user,
make it technical (``kernel_peercred_isolated_uid``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._checks import (
    require_choice,
    require_int,
    require_sha256,
    require_text,
    require_utc_timestamp,
)
from .base_checkpoints import BaseCheckpointRecord, validate_base_checkpoint_record
from .errors import CustodyRecordError
from .licences import MODEL_DATA_CLASSES, CustodyDataClass

CustodyObjectKind = Literal[
    "artefact",
    "base_checkpoint",
    "preregistration",
    "preregistration_approval",
    "split",
    "split_pool",
    "corpus",
    "grader_bundle",
    "sealed_look_opening",
    "sealed_look_journal_entry",
]
CUSTODY_OBJECT_KINDS: frozenset[str] = frozenset(
    {
        "artefact",
        "base_checkpoint",
        "preregistration",
        "preregistration_approval",
        "split",
        "split_pool",
        "corpus",
        "grader_bundle",
        "sealed_look_opening",
        "sealed_look_journal_entry",
    }
)
SEALED_CUSTODY_KINDS: frozenset[str] = frozenset(
    {"split", "split_pool", "corpus", "grader_bundle"}
)
"""Kinds that never leave custody; fixed in code, never configurable."""

MODEL_CUSTODY_KINDS: frozenset[str] = frozenset({"artefact", "base_checkpoint"})
"""Kinds that hold model weights and take part in lineage."""

WeightFormat = Literal["safetensors", "safetensors_tar", "none"]
WEIGHT_FORMATS: frozenset[str] = frozenset({"safetensors", "safetensors_tar", "none"})
RoleAuthenticationMode = Literal[
    "procedural_shared_os_user_until_d8", "kernel_peercred_isolated_uid"
]
ROLE_AUTHENTICATION_MODES: frozenset[str] = frozenset(
    {"procedural_shared_os_user_until_d8", "kernel_peercred_isolated_uid"}
)

_DATA_CLASSES_BY_KIND: dict[str, frozenset[str]] = {
    "artefact": MODEL_DATA_CLASSES,
    "base_checkpoint": MODEL_DATA_CLASSES,
    "split": MODEL_DATA_CLASSES,
    "split_pool": MODEL_DATA_CLASSES,
    "corpus": MODEL_DATA_CLASSES,
    "preregistration": frozenset({"not_applicable"}),
    "preregistration_approval": frozenset({"not_applicable"}),
    "grader_bundle": frozenset({"not_applicable"}),
    "sealed_look_opening": frozenset({"not_applicable"}),
    "sealed_look_journal_entry": frozenset({"not_applicable"}),
}


@dataclass(frozen=True, slots=True)
class CustodyIngestRecord:
    """The authoritative record custody writes when it ingests a blob.

    The 17 fields are, in order: ``schema``, ``programme``, ``kind``,
    ``content_digest``, ``tensor_set_digest``, ``custody_ingest_seq``,
    ``submitting_role``, ``submitting_holder``, ``submitting_uid``,
    ``submitting_pid``, ``role_authentication_mode``, ``byte_length``,
    ``ingested_at_utc``, ``data_class``, ``weight_format``,
    ``weight_format_reason`` and ``base_checkpoint``.
    """

    schema: Literal["custody-ingest-record.v1"]
    programme: str
    kind: CustodyObjectKind
    content_digest: str
    tensor_set_digest: str | None
    custody_ingest_seq: int
    submitting_role: str
    submitting_holder: str
    submitting_uid: int
    submitting_pid: int
    role_authentication_mode: RoleAuthenticationMode
    byte_length: int
    ingested_at_utc: str
    data_class: CustodyDataClass
    weight_format: WeightFormat
    weight_format_reason: str | None
    base_checkpoint: BaseCheckpointRecord | None


def is_sealed_kind(kind: str) -> bool:
    """Return whether a custody kind is sealed and may never be exported.

    Parameters
    ----------
    kind : str
        A custody object kind.

    Returns
    -------
    bool
        ``True`` for ``split``, ``split_pool``, ``corpus`` and ``grader_bundle``.
    """
    return kind in SEALED_CUSTODY_KINDS


def validate_custody_ingest_record(record: CustodyIngestRecord) -> CustodyIngestRecord:
    """Check a custody ingest record against its schema and kind rules.

    Parameters
    ----------
    record : CustodyIngestRecord
        The record to check.

    Returns
    -------
    CustodyIngestRecord
        The same record.

    Raises
    ------
    CustodyRecordError
        If a field is malformed, the data class or weight format is not
        allowed for the kind, or the base checkpoint part is inconsistent.
    MalformedDigestError
        If a digest field is malformed.
    """
    if not isinstance(record, CustodyIngestRecord):
        raise CustodyRecordError("a CustodyIngestRecord is required")
    require_choice(
        record.schema, "schema", {"custody-ingest-record.v1"}, CustodyRecordError
    )
    require_text(record.programme, "programme", CustodyRecordError)
    kind = require_choice(record.kind, "kind", CUSTODY_OBJECT_KINDS, CustodyRecordError)
    require_sha256(record.content_digest, "content_digest")
    require_int(
        record.custody_ingest_seq, "custody_ingest_seq", CustodyRecordError, minimum=0
    )
    require_text(record.submitting_role, "submitting_role", CustodyRecordError)
    require_text(record.submitting_holder, "submitting_holder", CustodyRecordError)
    require_int(record.submitting_uid, "submitting_uid", CustodyRecordError, minimum=0)
    require_int(record.submitting_pid, "submitting_pid", CustodyRecordError, minimum=1)
    require_choice(
        record.role_authentication_mode,
        "role_authentication_mode",
        ROLE_AUTHENTICATION_MODES,
        CustodyRecordError,
    )
    require_int(record.byte_length, "byte_length", CustodyRecordError, minimum=0)
    require_utc_timestamp(record.ingested_at_utc, "ingested_at_utc", CustodyRecordError)
    require_choice(
        record.data_class,
        f"data_class for kind {kind!r}",
        _DATA_CLASSES_BY_KIND[kind],
        CustodyRecordError,
    )
    weight_format = require_choice(
        record.weight_format, "weight_format", WEIGHT_FORMATS, CustodyRecordError
    )
    if kind not in MODEL_CUSTODY_KINDS and weight_format != "none":
        raise CustodyRecordError(f"kind {kind!r} must have weight_format 'none'")
    if weight_format == "none":
        if record.tensor_set_digest is not None:
            raise CustodyRecordError(
                "weight_format 'none' records no tensor_set_digest"
            )
    else:
        require_sha256(record.tensor_set_digest, "tensor_set_digest")
    if kind in MODEL_CUSTODY_KINDS and weight_format == "none":
        require_text(
            record.weight_format_reason, "weight_format_reason", CustodyRecordError
        )
    elif record.weight_format_reason is not None:
        raise CustodyRecordError(
            "weight_format_reason is only for model kinds of format 'none'"
        )
    if kind != "base_checkpoint":
        if record.base_checkpoint is not None:
            raise CustodyRecordError(
                "only base_checkpoint records carry a base checkpoint"
            )
        return record
    if record.base_checkpoint is None:
        raise CustodyRecordError(
            "a base_checkpoint record needs its base checkpoint part"
        )
    base = validate_base_checkpoint_record(record.base_checkpoint)
    if base.custody_ingested_file_sha256 != record.content_digest:
        raise CustodyRecordError(
            "base checkpoint file digest differs from content_digest"
        )
    if base.tensor_set_digest != record.tensor_set_digest:
        raise CustodyRecordError(
            "base checkpoint tensor_set_digest differs from the record"
        )
    return record
