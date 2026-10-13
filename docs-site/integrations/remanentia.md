<!-- SPDX-License-Identifier: BUSL-1.1 -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- Director-AI — Source-bound event feedback integration -->

# Source-bound REMANENTIA feedback

The advanced tier exposes an event-feedback SDK and command for an operator-registered source-grounded review procedure. It evaluates an answer from an actual served recall event, retains the complete original review and signed request, and checks the complete durable event-feedback acknowledgement. This acknowledgement records feedback about that event. It does not authorise an action, accept a workflow result, or prove that an entire result has been retained in memory.

Install director-ai-pro alongside the matching core package. The implementation uses the Python standard library; REMANENTIA remains an independently deployed service. Ordinary answering and legacy query-feedback APIs retain their existing behaviour.

## Operator profile

Provision the memory service's current principal/source/producer/procedure/model registration and private bearer/signing keys through its supported operator interface first. Client labels cannot create these rights. Select a private evidence directory already authorised by that registration. Keys and original evidence must be owner-only regular files; final symlinks are refused. The service admission boundary for complete original A/B review objects is 1 MiB. Oversized objects are refused rather than truncated.

Store these exact profile members in an owner-only JSON file. Paths refer to operator-controlled local files, not values received from the recall response:

    {
      "memory_url": "http://127.0.0.1:8765",
      "model_url": "http://127.0.0.1:8085/v1",
      "token_file": "/operator/private/memory-token",
      "signing_key_file": "/operator/private/producer-signing-key",
      "evidence_directory": "/operator/private/original-reviews",
      "evidence_directory_identity": {"device": 0, "inode": 1},
      "prepared_file": "/operator/private/prepared-review.json"
    }

The device/inode numbers above illustrate the shape only. Provision the actual pre-existing private root first, obtain its st_dev and st_ino through a no-follow descriptor, and record those measured integers in the private profile. They bind this registration to that directory across process restarts. A missing binding, replaced root or symbolic ancestor is refused; the client never refreshes the binding or recreates the registered root automatically.

The client also retains each review directory's original device/inode in an immutable root-level .review-directory-* custody record before contacting the model. These records and invocation reservations are original custody, not disposable temporary files. Replacing a review directory cannot clear its reservation: a later process refuses the conflicting inode. Preserve the registered root, these records and every original directory together; relocation or recovery needs an explicit operator procedure, not automatic reenrolment.

The memory endpoint requires verified HTTPS or explicit loopback HTTP. The model endpoint is local loopback, with no remote or CPU fallback, automatic retry, sampling override, output cap or response deadline. Operators must select and qualify the actual model artifact and default serving profile; an alias alone does not qualify a deployment.

The prepared file contains input, provenance, evidence_id and submission_id. Input has these exact members:

- event_encoding and response_encoding: sorted-compact-utf8-json-v1.
- recall_event_id and original_query: the actual server-minted event and original query.
- served_event_sha256 and served_response_sha256: SHA-256 of complete sorted compact UTF-8 JSON objects.
- candidate_id and candidate_text: a stable identifier and an answer actually present in the served response.
- source_snapshots: exactly one original source_id, source_version and full content_sha256.
- supplied_snippets: ordered, nonoverlapping UTF-8 byte slices present in both original source and served response. Each has source_snapshot, source_start, source_end, text, sha256 and served. Served selectors support context, summary, result with item/field, and semantic with item; each declares byte start/end.

Provenance has exact producer_id, actor_type (ai_project), principal (principal_id, tenant_id, project_id), adjudicator_id, procedure_id, procedure_version, provider and model members. Every value must match the existing operator registration. Review and submission IDs remain unchanged throughout recovery.

The SDK fetches the actual original event and exact source version under current rights. It checks complete digests, source response length/digest/ETag, principal bindings and source/served byte selectors before contacting the model or submitting feedback. Prepared metadata never substitutes for original bytes.

## Evaluate or import an original response

    director-ai event-feedback evaluate --profile /operator/private/profile.json

Evaluation reserves the original invocation identity before requesting the model. It retains the exact request, original source/event, original response, completion/usage metadata, decision, complete review object and separate whole signed carrier. A malformed, refused, truncated or abstaining response does not create a correctness label. A genuine abstention stops without submitting feedback. The client makes no automatic repair call; an existing invocation reservation or original response prevents another call under the same review identity.

The producer walks every review-directory component relative to an already open parent descriptor, with no-follow semantics. It checks each parent/child inode binding and holds the same leaf and ancestry descriptors through original custody, exclusive reservation, synchronisation, provider contact and response retention. It synchronises the original leaf and containing entries on that filesystem and rechecks their identities before transport. An observed ancestry replacement, permission change or sync failure refuses the call rather than following a replacement path.

The operator must preserve the registered root and private custody entries during an active operation and qualify the filesystem's durability semantics. Descriptor identity checks do not revoke a trusted host operator's ability to mutate files after a check; they do not claim adversarial kernel/host isolation or a destructive power-loss test. Cold recovery also checks the fixed root and durable per-review inode bindings. SDK writeback and replay hold that registered review handle too: transport attempts are created relative to it, and response/ACK retention checks the same ancestry. An observed custody change after a possible feedback POST remains WRITE_UNKNOWN rather than success or an inferred no-write refusal. Unsupported no-follow/directory-sync primitives are refused; there is no global sync, filesystem fallback or automatic model retry.

Importing a response is useful when the actual model invocation and original custody already exist:

    director-ai event-feedback import-response \
      --profile /operator/private/profile.json \
      --original-response /operator/private/provider-response.raw

Import validates and retains the original response and current source bindings. Import itself does not prove that the model ran; retain independent execution/provenance evidence with the original procedure.

## Recover an uncertain write

    director-ai event-feedback replay --profile /operator/private/profile.json

Replay reads the original retained request/review, verifies its bindings against prepared input, fetches current originals again and submits the same event/request/submission identity. It makes no model call. A saved local ACK cannot bypass current credential, source, producer or key rights. A qualified replay must retain the complete original ACK, including its outcome identity and timestamp.

Success prints only allowlisted receipt metadata, for example:

    {
      "state": "FEEDBACK_RECORDED",
      "http_status": 200,
      "feedback_ack_qualified": true,
      "outcome_id": "server-minted-outcome",
      "feedback_submission_id": "original-submission"
    }

ABSTAINED means no correctness label was submitted. REFUSED means pre-submit validation failed or the actual service authoritatively refused a write. WRITE_UNKNOWN means a write may have occurred but its complete durable receipt cannot be qualified, including disconnects, malformed responses and post-write custody failures. A false feedback_ack_qualified does not assert that no write occurred. Preserve original IDs and use replay to reconcile uncertainty.

The command exits with 0 for a qualified event ACK, 3 for a completed abstention, 4 for a service refusal or validation failure, and 5 for an uncertain write or failed post-write custody. Private queries, provider responses, keys and paths are not printed.

Import retains the current original source, served event, input and provenance alongside the response and signed review. It does not create an invocation reservation or prove model execution. Replay requires existing custody, checks the original decision and complete signed carrier against the exact request, and never creates a missing review identity. Prepared input files require input, provenance, evidence_id and submission_id; the original optional audit fields operator_scope, prepared_at, selected_model and source_file are also supported. These are nonblank strings, prepared_at must be a complete timestamp, and selected_model must match provenance.model. source_file is never followed, operator_scope is not a grant, and prepared_at cannot extend rights or reset a task deadline. Other members refuse. The served event's source snapshot status is retained as source metadata; it does not grant rights or certify the source's truth.

HTTP response custody admits at most 1 MiB, independently of the model's complete-on-response policy. An oversized or incomplete body is refused and retained only as an explicitly incomplete bounded prefix. Ambiguous headers preserve their original parsed pairs and received body; a possibly submitted write remains uncertain. No such response is accepted, automatically retried or converted to an incorrect verdict.

## SDK

    from pathlib import Path
    from director_ai.agentic.event_feedback.session import EventFeedbackSession

    session = EventFeedbackSession(Path("/operator/private/profile.json"))
    result = session.replay()
    if result.ack is not None:
        outcome_id = result.ack["outcome_id"]

The evaluate(), import_response(path) and replay() methods use the same registered implementation as the CLI. Evaluation/import return None for abstention. FeedbackOutcome carries the exact state, HTTP status, complete qualified ACK when present, and private transport-attempt custody location. FeedbackWriteUnknownError preserves post-write uncertainty if local receipt custody fails. Applications must retain this distinction when presenting status or scheduling recovery.
