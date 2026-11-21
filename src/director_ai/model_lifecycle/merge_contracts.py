# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Adapter Merge Rules

"""The rules for merging language artefacts, and for the merged result (M-J4-3).

A merge combines at least two distinct language artefacts of one mergeable
kind, adapters or full weights, each bound to its family contract by
digest. The parents must share one tokeniser, and a set and equal
architecture identity, embedding space and tied-embedding flag. Two unset
identities prove nothing, and the embedding space is what keeps two
separately trained bases with one tokeniser and one architecture apart.
Linear weights sum exactly to 1 as the decimals they were declared as;
``ties`` and ``dare_ties`` weights are scalings with no sum rule.

The merged artefact is checked as well: it is a new language artefact whose
contract claims the parents' tokeniser, architecture, embedding space and
tie. Lineage edges are advisory, so they are not a rule here.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal, Inexact, localcontext

from ._checks import require_choice, require_finite, require_tuple, require_unique
from .digests import record_digest
from .envelopes import ModelArtefactEnvelope, validate_envelope
from .errors import AdapterMergeError
from .family_companions import LanguageFamilyContract, validate_language_family_contract
from .spend_contracts import MONEY_CONTEXT

E = AdapterMergeError
MERGEABLE_KINDS: frozenset[str] = frozenset(
    {"safetensors_adapter", "safetensors_weights"}
)
MERGE_METHODS: frozenset[str] = frozenset({"linear", "ties", "dare_ties"})
Identity = tuple[str, str | None, str | None, bool]


def _identity(contract: LanguageFamilyContract) -> Identity:
    """Return a contract's tokeniser, architecture, embedding space and tie."""
    return (
        contract.tokeniser_envelope_digest,
        contract.architecture_spec_digest,
        contract.embedding_space_digest,
        contract.tie_word_embeddings,
    )


def _bound(envelope: ModelArtefactEnvelope, contract: LanguageFamilyContract) -> None:
    """Check a language artefact and the contract it names by digest."""
    validate_envelope(envelope)
    if envelope.family != "language":
        raise E("only language artefacts are merged")
    validate_language_family_contract(contract)
    if record_digest(contract) != envelope.family_contract_digest:
        raise E("an artefact's contract is not the one it names")


def merge_identity(
    parents: Sequence[ModelArtefactEnvelope],
    parent_contracts: Sequence[LanguageFamilyContract],
) -> Identity:
    """Return the identity every parent of a merge shares, or refuse the merge.

    Parameters
    ----------
    parents : Sequence[ModelArtefactEnvelope]
        The artefacts to merge.
    parent_contracts : Sequence[LanguageFamilyContract]
        Their family contracts, in the same order.

    Returns
    -------
    Identity
        The shared tokeniser, architecture, embedding space and tie.

    Raises
    ------
    AdapterMergeError
        If there are fewer than two parents or not one contract each, a
        parent is not a language artefact, names another contract, repeats
        another, or is of another kind, or the parents differ in tokeniser
        (M-J4-3) or in their set identity.
    EnvelopeError
        If an envelope or a contract is malformed.
    MalformedDigestError
        If a digest is malformed.
    """
    envelopes, contracts = tuple(parents), tuple(parent_contracts)
    if len(envelopes) < 2:
        raise E("a merge needs at least two parents")
    if len(contracts) != len(envelopes):
        raise E("a merge needs one family contract per parent")
    for envelope, contract in zip(envelopes, contracts, strict=True):
        _bound(envelope, contract)
    require_unique([p.content_digest for p in envelopes], "parent content digests", E)
    kinds = {p.artefact_kind for p in envelopes}
    if len(kinds) != 1 or not kinds <= MERGEABLE_KINDS:
        raise E("the parents are all adapters or all full weights")
    if len({c.tokeniser_envelope_digest for c in contracts}) != 1:
        raise E("M-J4-3: the parents use different tokenisers")
    for contract in contracts:
        if (
            contract.architecture_spec_digest is None
            or contract.embedding_space_digest is None
        ):
            raise E("a merge needs every parent's architecture and embedding identity")
    identities = {_identity(c) for c in contracts}
    if len(identities) != 1:
        raise E(
            "the parents differ in architecture, embedding space or tied embeddings"
        )
    return identities.pop()


def validate_adapter_merge(
    parents: Sequence[ModelArtefactEnvelope],
    parent_contracts: Sequence[LanguageFamilyContract],
    *,
    method: str,
    weights: Sequence[float],
) -> Identity:
    """Check a merge's parents, method and weights before it runs.

    Parameters
    ----------
    parents : Sequence[ModelArtefactEnvelope]
        The artefacts to merge.
    parent_contracts : Sequence[LanguageFamilyContract]
        Their family contracts, in the same order.
    method : str
        ``linear``, ``ties`` or ``dare_ties``.
    weights : Sequence[float]
        One weight per parent.

    Returns
    -------
    Identity
        The identity the merged artefact must carry.

    Raises
    ------
    AdapterMergeError
        If the parents are refused, the method is unknown, the weights are
        not one finite positive value per parent, or linear weights do not
        sum exactly to 1.
    EnvelopeError
        If an envelope or a contract is malformed.
    MalformedDigestError
        If a digest is malformed.
    """
    identity = merge_identity(parents, parent_contracts)
    chosen = require_choice(method, "method", MERGE_METHODS, E)
    values = require_tuple(tuple(weights), "weights", E)
    if len(values) != len(tuple(parents)):
        raise E("a merge needs one weight per parent")
    for value in values:
        if require_finite(value, "weight", E) <= 0.0:
            raise E("a merge weight is above zero")
    if chosen == "linear":
        try:
            with localcontext(MONEY_CONTEXT):
                total = sum((Decimal(str(value)) for value in values), Decimal(0))
        except Inexact as error:
            raise E(
                "the linear weights need more precision than the exact sum holds"
            ) from error
        if total != 1:
            raise E("linear merge weights sum exactly to 1")
    return identity


def validate_merge_result(
    merged: ModelArtefactEnvelope,
    merged_contract: LanguageFamilyContract,
    *,
    parents: Sequence[ModelArtefactEnvelope],
    parent_contracts: Sequence[LanguageFamilyContract],
) -> ModelArtefactEnvelope:
    """Check the artefact a merge produced against its parents.

    Parameters
    ----------
    merged : ModelArtefactEnvelope
        The merged artefact.
    merged_contract : LanguageFamilyContract
        The family contract it names.
    parents : Sequence[ModelArtefactEnvelope]
        The merged parents.
    parent_contracts : Sequence[LanguageFamilyContract]
        Their family contracts, in the same order.

    Returns
    -------
    ModelArtefactEnvelope
        The same merged artefact.

    Raises
    ------
    AdapterMergeError
        If the parents are refused, the result is not a language artefact,
        names another contract, claims another tokeniser, architecture,
        embedding space or tie, or is one of its parents.
    EnvelopeError
        If an envelope or a contract is malformed.
    MalformedDigestError
        If a digest is malformed.
    """
    identity = merge_identity(parents, parent_contracts)
    _bound(merged, merged_contract)
    if _identity(merged_contract) != identity:
        raise E(
            "the merged artefact claims another tokeniser, architecture or embedding"
        )
    if merged.content_digest in {p.content_digest for p in parents}:
        raise E("the merged artefact must be a new artefact")
    return merged
