# SPDX-License-Identifier: LicenseRef-BUSL-1.1-ANULUM-TDM
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Registered event feedback SDK session

"""Wire current source resolution, original review custody and strict writeback."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .codec import (
    FeedbackInvalidError,
    FeedbackWriteUnknownError,
    Object,
    canonical,
    decode,
    object_value,
    text,
)
from .contract import (
    members,
    payload_digest,
    validate_ack,
    validate_request,
    validate_timestamp,
)
from .custody import EvidenceRoot, durable_directory, private_bytes
from .memory import FeedbackOutcome, submit
from .originals import Originals, resolve
from .producer import ReviewSubmission, evaluate, retain_review, validate_submission
from .transport import HttpEndpoint


class EventFeedbackSession:
    """Use actual operator registration without provisioning keys or source rights.

    Parameters
    ----------
    profile : Path
        Private operator JSON with memory_url, model_url, token_file,
        signing_key_file, evidence_directory, evidence_directory_identity and prepared_file.
    """

    def __init__(self, profile: Path) -> None:
        """Resolve trusted file references while preserving source identity values."""
        config = decode(private_bytes(profile))
        members(
            config,
            {
                "memory_url",
                "model_url",
                "token_file",
                "signing_key_file",
                "evidence_directory",
                "evidence_directory_identity",
                "prepared_file",
            },
        )
        self.memory = HttpEndpoint(
            text(config["memory_url"]), Path(text(config["token_file"]))
        )
        self.provider = HttpEndpoint(text(config["model_url"]), timeout=None)
        if self.provider.host not in {"localhost", "127.0.0.1", "::1"}:
            raise FeedbackInvalidError()
        self.key = Path(text(config["signing_key_file"]))
        identity = object_value(config["evidence_directory_identity"])
        members(identity, {"device", "inode"})
        device, inode = identity["device"], identity["inode"]
        if not isinstance(device, int) or not isinstance(inode, int):
            raise FeedbackInvalidError()
        self.evidence = EvidenceRoot(
            Path(text(config["evidence_directory"])), device, inode
        )
        prepared = decode(private_bytes(Path(text(config["prepared_file"]))))
        metadata = {"operator_scope", "prepared_at", "selected_model", "source_file"}
        members(
            prepared, {"input", "provenance", "evidence_id", "submission_id"}, metadata
        )
        self.inputs = object_value(prepared.get("input"))
        self.provenance = object_value(prepared.get("provenance"))
        for name in metadata & prepared.keys():
            text(prepared[name])
        if "prepared_at" in prepared:
            validate_timestamp(prepared["prepared_at"])
        if "selected_model" in prepared and prepared[
            "selected_model"
        ] != self.provenance.get("model"):
            raise FeedbackInvalidError()
        self.evidence_id = text(prepared.get("evidence_id"))
        self.submission_id = text(prepared.get("submission_id"))
        self.directory = (
            self.evidence.path
            / hashlib.sha256(self.evidence_id.encode("utf-8")).hexdigest()
        )

    def resolve(self) -> Originals:
        """Obtain authorised original source/event bytes without a model request.

        Returns
        -------
        Originals
            Current original source and actual served event with strict bindings.
        """
        with durable_directory(self.evidence.path, self.evidence):
            return resolve(self.memory, self.inputs, self.provenance)

    def evaluate(self) -> FeedbackOutcome | None:
        """Evaluate once, retain original A/B, then request a durable event ACK.

        Returns
        -------
        FeedbackOutcome or None
            Strict writeback state, or None for an actual model abstention.
        """
        originals = self.resolve()
        submission = evaluate(
            self.provider,
            originals,
            self.evidence_id,
            self.submission_id,
            self.key,
            self.evidence,
        )
        return self._submit(submission)

    def import_response(self, original_response: Path) -> FeedbackOutcome | None:
        """Admit retained original response bytes without repeating model execution.

        Parameters
        ----------
        original_response : Path
            Original private provider response; importing does not prove execution.

        Returns
        -------
        FeedbackOutcome or None
            Strict writeback state, or None for the retained original abstention.
        """
        originals = self.resolve()
        submission = retain_review(
            originals,
            private_bytes(original_response),
            self.evidence_id,
            self.submission_id,
            self.key,
            self.evidence,
        )
        return self._submit(submission)

    def _submit(self, submission: ReviewSubmission | None) -> FeedbackOutcome | None:
        if submission is None:
            return None
        with durable_directory(self.directory, self.evidence) as custody:
            result = submit(
                self.memory, submission.event_id, submission.request, custody
            )
            if result.ack is not None:
                try:
                    custody.retain("ack.json", canonical(result.ack))
                except (FeedbackInvalidError, OSError) as exc:
                    raise FeedbackWriteUnknownError() from exc
            return result

    def replay(self) -> FeedbackOutcome:
        """Reconcile the same immutable submission under freshly checked current rights.

        Returns
        -------
        FeedbackOutcome
            Original ACK on valid replay, refusal, or a still uncertain write.
        """
        originals = self.resolve()
        with durable_directory(self.directory, self.evidence, create=False) as custody:
            request = decode(private_bytes(Path("request.json"), custody.descriptor))
            validate_request(request)
            if (
                request["feedback_submission_id"] != self.submission_id
                or object_value(request["evidence_ref"])["evidence_id"]
                != self.evidence_id
            ):
                raise FeedbackInvalidError()
            review = private_bytes(Path("review.json"), custody.descriptor)
            if (
                hashlib.sha256(review).hexdigest()
                != object_value(request["evidence_ref"])["evidence_sha256"]
            ):
                raise FeedbackInvalidError()
            carrier = private_bytes(
                Path(
                    "carrier-"
                    + payload_digest(text(self.inputs["recall_event_id"]), request)
                    + ".json"
                ),
                custody.descriptor,
            )
            validate_submission(
                originals, request, review, carrier, self.evidence_id, self.key
            )
            prior: Object | None = None
            if custody.exists("ack.json"):
                prior = decode(private_bytes(Path("ack.json"), custody.descriptor))
                validate_ack(prior, text(self.inputs["recall_event_id"]), request)
            result = submit(
                self.memory,
                text(self.inputs["recall_event_id"]),
                request,
                custody,
                prior,
            )
            if result.ack is not None:
                try:
                    custody.retain("ack.json", canonical(result.ack))
                except (FeedbackInvalidError, OSError) as exc:
                    raise FeedbackWriteUnknownError() from exc
            return result
