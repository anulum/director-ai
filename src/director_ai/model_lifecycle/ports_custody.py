# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Custody and Deployment Ports

"""Ports across the custody boundary and the chained deployment of a model.

The custody port invokes the isolated blob-custody tool under its own user:
it ingests typed objects, requests sealed looks, sets and clears scoped
pins and reads the journal. The deployment port declares a promotion,
publishes or seals generation ``base_generation + 1`` and verifies the swap
afterwards. Adapters outside the kernel implement them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from .cost_and_promotion_contracts import PromotionDecisionRecord
from .custody_records import CustodyIngestRecord, CustodyObjectKind
from .deployment_records import PostSwapVerification
from .evaluator_outputs import SealedLookResult
from .licences import CustodyDataClass
from .promotion_records import PromotionDependencyPins, ServingMemoryBudget
from .sealed_looks import SealedLookRequest


@runtime_checkable
class CustodySubprocessPort(Protocol):
    """Invokes the isolated blob-custody CLI boundary with typed CustodyObjectKind.

    The journal's contract binds custody in two ways. It ingests a look's
    opening, as a ``sealed_look_opening`` record, before it releases any data
    of the look's split; a result whose opening was ingested later is never
    used. It also ingests each journal entry, as a
    ``sealed_look_journal_entry`` record, before the programme's next look
    opens: admission refuses a journal whose entries or openings custody
    does not hold up to the look's watermark, so a lagging ingest stops the
    next look rather than passing it. A look that is aborted after its
    opening still gets its entry, with a fault outcome and its attestation.
    """

    def ingest(
        self,
        *,
        programme: str,
        kind: CustodyObjectKind,
        data_class: CustodyDataClass,
        weight_format: Literal["safetensors", "safetensors_tar", "none"],
        weight_format_reason: str | None,
        role: str,
        holder: str,
        path: Path,
        expected_digest: str | None = None,
        base_checkpoint_meta: Path | None = None,
    ) -> CustodyIngestRecord:
        """Ingest one object into custody and return its record.

        Parameters
        ----------
        programme : str
            The programme.
        kind : CustodyObjectKind
            The object's kind.
        data_class : CustodyDataClass
            The object's data class.
        weight_format : str
            The weight format, or none.
        weight_format_reason : str or None
            Why the object carries no weights, when it carries none.
        role : str
            The submitting role.
        holder : str
            The role's holder.
        path : Path
            The object.
        expected_digest : str or None
            The digest a recall must match.
        base_checkpoint_meta : Path or None
            A base checkpoint's metadata.

        Returns
        -------
        CustodyIngestRecord
            The ingest record.
        """
        ...

    def request_look(
        self,
        *,
        request: SealedLookRequest,
    ) -> SealedLookResult:
        """Request one sealed look and return its result.

        Parameters
        ----------
        request : SealedLookRequest
            The look request.

        Returns
        -------
        SealedLookResult
            The look's result.
        """
        ...

    def set_pin(
        self,
        *,
        digest: str,
        pin_scope: str,
        reason: str,
        role: str,
        holder: str,
        pinned: bool,
    ) -> None:
        """Pin or unpin a digest in one scope.

        Parameters
        ----------
        digest : str
            The digest.
        pin_scope : str
            The slot or service the pin is scoped to.
        reason : str
            Why.
        role : str
            The acting role.
        holder : str
            The role's holder.
        pinned : bool
            Pin when true, unpin when false.
        """
        ...

    def read_journal(
        self,
        *,
        programme: str | None = None,
        since_seq: int = 0,
    ) -> tuple[dict[str, object], ...]:
        """Read custody journal entries.

        Parameters
        ----------
        programme : str or None
            The programme, or every programme.
        since_seq : int
            Read entries after this sequence.

        Returns
        -------
        tuple[dict[str, object], ...]
            The entries, as decoded JSON objects.
        """
        ...


@runtime_checkable
class ChainedDeploymentPort(Protocol):
    """Declares, publishes/seals at base_generation + 1, and verifies chained swaps."""

    def declare_deployment(
        self,
        *,
        decision: PromotionDecisionRecord,
        pins: PromotionDependencyPins,
        serving_budget: ServingMemoryBudget,
        chain_dir: Path,
        start_record_path: Path,
        base_generation: int,
        base_artefact_sha256: str,
        artefact_sha256: str,
        revision: str,
    ) -> str:
        """Declare a deployment of a promotion and return its operation id.

        Parameters
        ----------
        decision : PromotionDecisionRecord
            The admitted promotion.
        pins : PromotionDependencyPins
            Its dependency pins.
        serving_budget : ServingMemoryBudget
            Its serving memory budget.
        chain_dir : Path
            The generation chain.
        start_record_path : Path
            The serving start record.
        base_generation : int
            The generation the swap starts from.
        base_artefact_sha256 : str
            Its artefact, 64 zeros at generation 0.
        artefact_sha256 : str
            The artefact to deploy.
        revision : str
            The serving revision.

        Returns
        -------
        str
            The operation id.
        """
        ...

    def publish_or_seal_generation(
        self,
        *,
        chain_dir: Path,
        base_generation: int,
        operation_id: str,
        attempt_id: str,
        artefact_sha256: str | None,
        seal: bool,
    ) -> Literal["created", "refused"]:
        """Publish or seal generation ``base_generation + 1``.

        Parameters
        ----------
        chain_dir : Path
            The generation chain.
        base_generation : int
            The base generation, at least 0.
        operation_id : str
            The operation.
        attempt_id : str
            The attempt.
        artefact_sha256 : str or None
            The artefact to publish; ``None`` to seal.
        seal : bool
            Whether to seal instead of publishing.

        Returns
        -------
        str
            ``"created"`` or ``"refused"``.
        """
        ...

    def verify_post_swap(
        self,
        *,
        chain_dir: Path,
        start_record_path: Path,
        base_generation: int,
        base_artefact_sha256: str,
        artefact_sha256: str,
        revision: str,
        operation_id: str,
        attempt_id: str,
    ) -> PostSwapVerification:
        """Verify a swap after publication and return the typed verification.

        Parameters
        ----------
        chain_dir : Path
            The generation chain.
        start_record_path : Path
            The serving start record.
        base_generation : int
            The base generation.
        base_artefact_sha256 : str
            Its artefact, 64 zeros at generation 0.
        artefact_sha256 : str
            The deployed artefact.
        revision : str
            The serving revision.
        operation_id : str
            The operation.
        attempt_id : str
            The attempt.

        Returns
        -------
        PostSwapVerification
            The verification.
        """
        ...
