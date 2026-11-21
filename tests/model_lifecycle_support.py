# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — model lifecycle test builders

"""Builders of valid records, weight files and look scenarios for tests.

Every builder returns a valid object; tests change one field with
:func:`dataclasses.replace` to show one refusal at a time.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field, replace

from director_ai.model_lifecycle import (
    BaseCheckpointRecord,
    ComputeAndTuningBudget,
    CustodyIngestRecord,
    CustodyProgrammeRoles,
    DualAxisLicenceRecord,
    GraderSpec,
    LineageComponent,
    LineageEdge,
    ModelArtefactEnvelope,
    PreregistrationApproval,
    PreregistrationRecord,
    SealedLookJournalEntry,
    SealedLookRequest,
    SealedSplitPool,
    admit_sealed_look,
    record_digest,
)

TensorSpec = tuple[str, str, list[int], bytes]
TENSOR_A: TensorSpec = (
    "embed.weight",
    "BF16",
    [2, 2],
    b"\x80\x43\x00\x3f\x80\x3f\x00\x40",
)
TENSOR_B: TensorSpec = ("norm.weight", "F32", [2], b"\x00\x00\x80\x3f\x00\x00\x00\x40")
FIXTURE_DIGEST = "1fd6bf19860d336d7d6e0917fa65bbcd8a5a7ba3784b821125427f603779b83c"
FLIPPED_DIGEST = "33292c5a646cb45c63dd8701565b8de2f0c7a6249a09896c42a155ffcfb6090d"
UTC = "2026-11-21T00:00:00Z"
PROGRAMME = "pilot-a"
CUSTODIAN = "role/custodian"
APPROVER = "role/threshold-approver"
PROPOSER_A = "role/proposer-a"
PROPOSER_B = "role/proposer-b"


def digest_of(label: str) -> str:
    """Return a stable SHA-256 hex digest standing for ``label``."""
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def safetensors_bytes(
    tensors: list[TensorSpec],
    metadata: dict[str, str] | None = None,
    *,
    entries: dict[str, object] | None = None,
    trailing: bytes = b"",
) -> bytes:
    """Return a safetensors file with contiguous tensors, or raw ``entries``."""
    header: dict[str, object] = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    data = b""
    for name, dtype, shape, payload in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [len(data), len(data) + len(payload)],
        }
        data += payload
    header.update(entries or {})
    raw = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw + data + trailing


def ustar_header(
    name: bytes,
    size: int,
    *,
    typeflag: bytes = b"0",
    magic: bytes = b"ustar\x00",
    version: bytes = b"00",
    prefix: bytes = b"",
    size_field: bytes | None = None,
    checksum_delta: int = 0,
) -> bytes:
    """Return one 512-byte ustar header with a correct checksum."""
    block = bytearray(512)
    block[0 : len(name)] = name
    block[100:108] = b"0000644\x00"
    block[108:116] = b"0001750\x00"
    block[116:124] = b"0001750\x00"
    field = size_field if size_field is not None else b"%011o\x00" % size
    block[124:136] = field.ljust(12, b"\x00")
    block[136:148] = b"00000000000\x00"
    block[156:157] = typeflag
    block[257:263] = magic
    block[263:265] = version
    block[345 : 345 + len(prefix)] = prefix
    block[148:156] = b" " * 8
    checksum = sum(block) + checksum_delta
    block[148:156] = b"%06o\x00 " % checksum
    return bytes(block)


def ustar_archive(
    members: list[tuple[bytes, bytes]], *, end_region: int = 1024
) -> bytes:
    """Return a ustar archive of regular files followed by ``end_region`` zeros."""
    out = b""
    for name, payload in members:
        out += ustar_header(name, len(payload)) + payload
        out += bytes(-len(payload) % 512)
    return out + bytes(end_region)


def licence(**changes: object) -> DualAxisLicenceRecord:
    """Return a valid licence record with ``changes`` applied."""
    base = DualAxisLicenceRecord(
        schema="dual-axis-licence-record.v1",
        spdx_expression="Apache-2.0",
        weight_and_output_terms_digest=digest_of("terms"),
        tdm_opted_out=False,
        data_class="pilot",
        commercial_use_permitted=True,
        derivative_weights_permitted=True,
        output_training_permitted=True,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def budget(**changes: object) -> ComputeAndTuningBudget:
    """Return a valid budget with ``changes`` applied."""
    base = ComputeAndTuningBudget(
        schema="compute-and-tuning-budget.v1",
        max_wall_seconds=3600,
        max_accelerator_hours=1.5,
        max_hyperparameter_trials=8,
        max_sealed_looks=1,
        min_training_seeds=3,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def edge(parent: str, child: str, **changes: object) -> LineageEdge:
    """Return a valid lineage edge from ``parent`` to ``child``."""
    base = LineageEdge(parent, child, "finetuned_from", "search-1", UTC)
    return replace(base, **changes)  # type: ignore[arg-type]


def envelope(**changes: object) -> ModelArtefactEnvelope:
    """Return a valid language-model envelope with ``changes`` applied."""
    content = digest_of("candidate")
    base = ModelArtefactEnvelope(
        schema="model-artefact-envelope.v1",
        family="language",
        artefact_kind="safetensors_adapter",
        content_digest=content,
        lineage_edges=(edge(digest_of("base"), content),),
        recipe_digest=digest_of("recipe"),
        licence_record=licence(),
        budget_spent=budget(),
        spend_reservation_digest=digest_of("reservation"),
        publishable=False,
        excluded_reviewer_vendors=("vendor-a",),
        family_contract_schema="language-family-contract.v1",
        family_contract_digest=digest_of("family"),
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def base_checkpoint(**changes: object) -> BaseCheckpointRecord:
    """Return a valid base checkpoint record with ``changes`` applied."""
    base = BaseCheckpointRecord(
        schema="base-checkpoint-record.v1",
        repository_url="https://example.org/models/base",
        upstream_revision_kind="git_commit_sha",
        upstream_revision="a" * 40,
        upstream_published_digest_kind="sha256",
        upstream_published_digest=digest_of("upstream"),
        custody_ingested_file_sha256=digest_of("base-file"),
        tensor_set_digest=FIXTURE_DIGEST,
        weight_spdx_expression="Apache-2.0",
        commercial_and_derivative_clear=True,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def custody_record(
    digest: str,
    kind: str = "artefact",
    seq: int = 1,
    role: str = PROPOSER_A,
    **changes: object,
) -> CustodyIngestRecord:
    """Return a valid custody record for ``digest`` with ``changes`` applied."""
    model = kind in {"artefact", "base_checkpoint"}
    base = CustodyIngestRecord(
        schema="custody-ingest-record.v1",
        programme=PROGRAMME,
        kind=kind,  # type: ignore[arg-type]
        content_digest=digest,
        tensor_set_digest=FIXTURE_DIGEST if model else None,
        custody_ingest_seq=seq,
        submitting_role=role,
        submitting_holder="seat-1",
        submitting_uid=1000,
        submitting_pid=4242,
        role_authentication_mode="procedural_shared_os_user_until_d8",
        byte_length=1024,
        ingested_at_utc=UTC,
        data_class="pilot"
        if kind not in {"preregistration", "preregistration_approval", "grader_bundle"}
        else "not_applicable",
        weight_format="safetensors" if model else "none",
        weight_format_reason=None,
        base_checkpoint=(
            base_checkpoint(custody_ingested_file_sha256=digest)
            if kind == "base_checkpoint"
            else None
        ),
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def grader(**changes: object) -> GraderSpec:
    """Return a valid sealed, exact grader with ``changes`` applied."""
    base = GraderSpec("grader-sealed", digest_of("grader"), ("sealed",), True, None)
    return replace(base, **changes)  # type: ignore[arg-type]


SPLITS = (digest_of("split-1"), digest_of("split-2"), digest_of("split-3"))


def pool(**changes: object) -> SealedSplitPool:
    """Return a valid three-split pool with ``changes`` applied."""
    base = SealedSplitPool("pool-1", SPLITS, digest_of("dedup"))
    return replace(base, **changes)  # type: ignore[arg-type]


def preregistration(**changes: object) -> PreregistrationRecord:
    """Return a valid preregistration with ``changes`` applied."""
    base = PreregistrationRecord(
        schema="lifecycle-preregistration.v1",
        study_id="study-1",
        declared_arms=("candidate",),
        search_spaces_and_budgets_digest=digest_of("search"),
        split_pool=pool(),
        graders=(grader(),),
        non_inferiority_margins_bps=(("accuracy", 100),),
        planned_sample_size=3925,
        target_power_permille=800,
        ci_method="paired_exact_mcnemar",
        resampling_unit="source_cluster",
        pass_rule_digest=digest_of("pass-rule"),
        frozen_look_candidate_digests=(digest_of("A1"),),
        analysis_code_digest=digest_of("analysis"),
        declared_registered_at_utc=UTC,
        base_checkpoint_digests=(),
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def journal_entry(
    lineage_id: str,
    frozen: tuple[str, ...],
    *,
    split: str = SPLITS[0],
    status: str = "released",
    release_seq: int = 25,
    role: str = PROPOSER_A,
    look_kind: str = "initial_confirmatory",
) -> SealedLookJournalEntry:
    """Return a valid journal entry for an earlier look."""
    fault = (
        digest_of("attestation")
        if status in {"infrastructure_fault", "voided_evaluator_fault"}
        else ""
    )
    return SealedLookJournalEntry(
        schema="sealed-look-journal-entry.v1",
        journal_seq=1,
        custody_seq_at_release=release_seq,
        derived_proposer_lineage_id=lineage_id,
        split_digest=split,
        frozen_artefact_digests=frozen,
        requester_role=role,
        requester_holder="seat-1",
        look_kind=look_kind,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        fault_attestation_digest=fault,
        recorded_at_utc=UTC,
        programme=PROGRAMME,
        evaluator_name="evaluator-x",
        evaluator_digest=digest_of("evaluator"),
    )


@dataclass
class LookScenario:
    """A complete, admissible look: one preregistration, approval and arm.

    Splits are ingested at sequences 1 to 3 by the custodian, the
    preregistration at 10, its approval at 11, and arm ``A1`` at 20.
    """

    prereg: PreregistrationRecord = field(default_factory=preregistration)
    records: dict[str, CustodyIngestRecord] = field(default_factory=dict)
    edges: tuple[LineageEdge, ...] = ()
    journal: list[SealedLookJournalEntry] = field(default_factory=list)
    study_entries: list[SealedLookJournalEntry] = field(default_factory=list)
    look_kind: str = "initial_confirmatory"
    requester: str = PROPOSER_A
    split: str = SPLITS[0]
    watermark: int = 100
    prereg_seq: int = 10
    approval_seq: int = 11
    approval_role: str = APPROVER
    configured_custodian: str = CUSTODIAN
    configured_approver: str = APPROVER
    ingest_approval: bool = True

    def __post_init__(self) -> None:
        """Ingest the splits and the arm unless the test supplied records."""
        if not self.records:
            for index, split in enumerate(SPLITS, start=1):
                self.add(split, "split", index, CUSTODIAN)
            self.add(digest_of("A1"), "artefact", 20, PROPOSER_A)

    def add(
        self, digest: str, kind: str, seq: int, role: str, **changes: object
    ) -> str:
        """Ingest one record and return its digest."""
        self.records[digest] = custody_record(digest, kind, seq, role, **changes)
        return digest

    def approval(self) -> PreregistrationApproval:
        """Return the approval of this scenario's preregistration."""
        return PreregistrationApproval(
            "preregistration-approval.v1",
            record_digest(self.prereg),
            self.prereg.split_pool.pool_id,
            UTC,
        )

    def request(self, **changes: object) -> SealedLookRequest:
        """Return the look request with ``changes`` applied."""
        base = SealedLookRequest(
            schema="sealed-look-request.v1",
            preregistration_digest=record_digest(self.prereg),
            split_digest=self.split,
            frozen_artefact_digests=self.prereg.frozen_look_candidate_digests,
            declared_lineage_edges=self.edges,
            requester_role=self.requester,
            requester_holder="seat-1",
            look_kind=self.look_kind,  # type: ignore[arg-type]
            programme=PROGRAMME,
            evaluator_name="evaluator-x",
            evaluator_digest=digest_of("evaluator"),
        )
        return replace(base, **changes)  # type: ignore[arg-type]

    def admit(self, **request_changes: object) -> LineageComponent:
        """Ingest the preregistration and approval, then admit the look."""
        records = dict(self.records)
        prereg_digest = record_digest(self.prereg)
        records.setdefault(
            prereg_digest,
            custody_record(prereg_digest, "preregistration", self.prereg_seq),
        )
        approval = self.approval()
        approval_digest = record_digest(approval)
        if self.ingest_approval:
            records.setdefault(
                approval_digest,
                custody_record(
                    approval_digest,
                    "preregistration_approval",
                    self.approval_seq,
                    self.approval_role,
                ),
            )
        return admit_sealed_look(
            self.request(**request_changes),
            preregistration=self.prereg,
            approval=approval,
            custody_records=records,
            roles=CustodyProgrammeRoles(
                PROGRAMME,
                self.configured_custodian,
                self.configured_approver,
                "deployment-operator",
            ),
            prior_journal=self.journal,
            prior_study_entries=self.study_entries,
            look_custody_seq=self.watermark,
        )
