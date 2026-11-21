# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Artefact Envelopes

"""Model artefact envelopes and proposer-declared lineage edges.

An envelope describes one model artefact exactly: its family, kind,
content digest, recipe, licence, spent budget, a real spend reservation and
its family contract. No envelope field is optional (``None``) or empty.

Four families are registered: ``language``, ``dynamics``, ``spiking`` and
``quantum_variational``. Each family's rule and companion contract are in
:mod:`.family_registry` and :mod:`.family_companions`.

A :class:`LineageEdge` is advisory. It may merge two lineages, never split
one, and its declared time is informational only: lineage and ordering are
computed from custody ingest records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._checks import (
    require_bool,
    require_choice,
    require_sha256,
    require_text,
    require_tuple,
    require_unique,
    require_utc_timestamp,
)
from .errors import EnvelopeError
from .licences import (
    ComputeAndTuningBudget,
    DualAxisLicenceRecord,
    validate_compute_budget,
    validate_licence_record,
)

ActiveModelFamily = Literal["language", "dynamics", "spiking", "quantum_variational"]
ACTIVE_MODEL_FAMILIES: frozenset[str] = frozenset(
    {"language", "dynamics", "spiking", "quantum_variational"}
)

LineageRelation = Literal[
    "finetuned_from",
    "merged_from",
    "patched_from",
    "exported_from",
    "distilled_from",
    "pretrained_from_scratch",
    "continued_pretrained_from",
]
LINEAGE_RELATIONS: frozenset[str] = frozenset(
    {
        "finetuned_from",
        "merged_from",
        "patched_from",
        "exported_from",
        "distilled_from",
        "pretrained_from_scratch",
        "continued_pretrained_from",
    }
)
ArtefactKind = Literal[
    "safetensors_weights", "safetensors_adapter", "word_patch", "exported_binary"
]
ARTEFACT_KINDS: frozenset[str] = frozenset(
    {"safetensors_weights", "safetensors_adapter", "word_patch", "exported_binary"}
)


@dataclass(frozen=True, slots=True)
class LineageEdge:
    """An advisory edge a proposer declares between two artefacts.

    Attributes
    ----------
    parent_digest : str
        Content digest of the parent artefact.
    child_digest : str
        Content digest of the derived artefact.
    relation : LineageRelation
        How the child was derived.
    search_run_id : str
        The proposer's search run.
    declared_created_at_utc : str
        Informational; never used for lineage or ordering.
    """

    parent_digest: str
    child_digest: str
    relation: LineageRelation
    search_run_id: str
    declared_created_at_utc: str


@dataclass(frozen=True, slots=True)
class ModelArtefactEnvelope:
    """The exact description of one model artefact.

    Attributes
    ----------
    schema : Literal["model-artefact-envelope.v1"]
        Record schema.
    family : ActiveModelFamily
        Model family.
    artefact_kind : ArtefactKind
        Weights, adapter, word patch or exported binary.
    content_digest : str
        SHA-256 of the artefact bytes.
    lineage_edges : tuple[LineageEdge, ...]
        Declared edges whose child is this artefact.
    recipe_digest : str
        SHA-256 of the recipe that produced it.
    licence_record : DualAxisLicenceRecord
        Its licence and output terms.
    budget_spent : ComputeAndTuningBudget
        What producing it spent.
    spend_reservation_digest : str
        SHA-256 of the real spend reservation, also for zero-cost local runs.
    publishable : bool
        Whether it may be published.
    excluded_reviewer_vendors : tuple[str, ...]
        Vendors whose reviewers may not judge it.
    family_contract_schema : str
        Schema name of the family contract.
    family_contract_digest : str
        SHA-256 of the family contract.
    """

    schema: Literal["model-artefact-envelope.v1"]
    family: ActiveModelFamily
    artefact_kind: ArtefactKind
    content_digest: str
    lineage_edges: tuple[LineageEdge, ...]
    recipe_digest: str
    licence_record: DualAxisLicenceRecord
    budget_spent: ComputeAndTuningBudget
    spend_reservation_digest: str
    publishable: bool
    excluded_reviewer_vendors: tuple[str, ...]
    family_contract_schema: str
    family_contract_digest: str


def validate_lineage_edge(edge: LineageEdge) -> LineageEdge:
    """Check one declared lineage edge.

    Parameters
    ----------
    edge : LineageEdge
        The edge to check.

    Returns
    -------
    LineageEdge
        The same edge.

    Raises
    ------
    EnvelopeError
        If a field is malformed or the edge is a self-loop.
    MalformedDigestError
        If an endpoint digest is malformed.
    """
    if not isinstance(edge, LineageEdge):
        raise EnvelopeError("a LineageEdge is required")
    require_sha256(edge.parent_digest, "parent_digest")
    require_sha256(edge.child_digest, "child_digest")
    if edge.parent_digest == edge.child_digest:
        raise EnvelopeError("a lineage edge cannot join an artefact to itself")
    require_choice(edge.relation, "relation", LINEAGE_RELATIONS, EnvelopeError)
    require_text(edge.search_run_id, "search_run_id", EnvelopeError)
    require_utc_timestamp(
        edge.declared_created_at_utc, "declared_created_at_utc", EnvelopeError
    )
    return edge


def require_active_family(family: object) -> str:
    """Return a registered family name, refusing unknown names.

    Parameters
    ----------
    family : object
        A family name.

    Returns
    -------
    str
        One of the four registered families.

    Raises
    ------
    EnvelopeError
        If the name is not registered.
    """
    return require_choice(family, "family", ACTIVE_MODEL_FAMILIES, EnvelopeError)


def validate_envelope(envelope: ModelArtefactEnvelope) -> ModelArtefactEnvelope:
    """Check a model artefact envelope completely.

    Parameters
    ----------
    envelope : ModelArtefactEnvelope
        The envelope to check.

    Returns
    -------
    ModelArtefactEnvelope
        The same envelope.

    Raises
    ------
    EnvelopeError
        If a field is malformed or an edge does not end at this artefact.
    MalformedDigestError
        If a digest field is malformed or empty.
    LicenceRecordError
        If the licence record or budget is malformed.
    """
    if not isinstance(envelope, ModelArtefactEnvelope):
        raise EnvelopeError("a ModelArtefactEnvelope is required")
    require_choice(
        envelope.schema, "schema", {"model-artefact-envelope.v1"}, EnvelopeError
    )
    require_active_family(envelope.family)
    require_choice(
        envelope.artefact_kind, "artefact_kind", ARTEFACT_KINDS, EnvelopeError
    )
    content = require_sha256(envelope.content_digest, "content_digest")
    for edge in require_tuple(envelope.lineage_edges, "lineage_edges", EnvelopeError):
        if not isinstance(edge, LineageEdge):
            raise EnvelopeError("lineage_edges must hold LineageEdge records")
        if validate_lineage_edge(edge).child_digest != content:
            raise EnvelopeError(
                "every envelope lineage edge must end at its content_digest"
            )
    require_sha256(envelope.recipe_digest, "recipe_digest")
    validate_licence_record(envelope.licence_record)
    validate_compute_budget(envelope.budget_spent)
    require_sha256(envelope.spend_reservation_digest, "spend_reservation_digest")
    require_bool(envelope.publishable, "publishable", EnvelopeError)
    vendors = require_tuple(
        envelope.excluded_reviewer_vendors, "excluded_reviewer_vendors", EnvelopeError
    )
    for vendor in vendors:
        require_text(vendor, "excluded_reviewer_vendors", EnvelopeError)
    require_unique(vendors, "excluded_reviewer_vendors", EnvelopeError)
    require_text(
        envelope.family_contract_schema, "family_contract_schema", EnvelopeError
    )
    require_sha256(envelope.family_contract_digest, "family_contract_digest")
    return envelope
