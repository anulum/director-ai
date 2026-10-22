# SPDX-License-Identifier: LicenseRef-BUSL-1.1-ANULUM-TDM
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Event feedback protocol tests
"""Exercise registered product SDK and CLI with explicit protocol controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from director_ai.agentic.event_feedback.cli import main
from director_ai.agentic.event_feedback.codec import (
    FeedbackInvalidError,
    FeedbackWriteUnknownError,
    Object,
    canonical,
    decode,
    object_value,
)
from director_ai.agentic.event_feedback.contract import payload_digest
from director_ai.agentic.event_feedback.custody import retain
from director_ai.agentic.event_feedback.originals import Originals
from director_ai.agentic.event_feedback.session import EventFeedbackSession

from .conftest import protocol_control, provider_response, registered_root
from .test_originals import source_control


@pytest.mark.parametrize(
    "fault", ["none", "operator_scope", "prepared_at", "selected_model", "source_file"]
)
def test_original_preparation_audit_fields_are_supported_without_authority(
    originals: Originals, tmp_path: Path, fault: str
) -> None:
    """Keep the genuine original preparation shape while refusing changed metadata."""
    control = source_control(originals)
    control.acknowledge_writes = True
    with protocol_control(control) as endpoint:
        profile = profile_file(
            tmp_path, f"http://{endpoint.host}:{endpoint.port}", originals
        )
        prepared_file = tmp_path / "prepared.json"
        prepared = decode(prepared_file.read_bytes())
        prepared.update(
            {
                "operator_scope": "isolated source/event metadata only",
                "prepared_at": "2026-10-04T17:16:07.123456+00:00",
                "selected_model": originals.provenance["model"],
                "source_file": str(tmp_path / "never-read-source-file"),
            }
        )
        if fault != "none":
            prepared[fault] = (
                "different-model"
                if fault == "selected_model"
                else "not-a-timestamp"
                if fault == "prepared_at"
                else None
            )
        prepared_file.write_bytes(canonical(prepared))
        if fault == "none":
            raw = tmp_path / "response"
            retain(raw, provider_response())
            result = EventFeedbackSession(profile).import_response(raw)
            assert result is not None and result.state == "FEEDBACK_RECORDED"
            assert not (tmp_path / "never-read-source-file").exists()
        else:
            with pytest.raises(FeedbackInvalidError):
                EventFeedbackSession(profile)
            assert control.requests == []


@pytest.mark.parametrize(
    "field",
    [
        "schema",
        "input",
        "provenance",
        "label_target",
        "provider_response_b64",
        "request-query",
        "request-verdict",
        "request-verifier",
        "request-procedure",
    ],
)
def test_replay_validates_full_original_bindings_even_with_matching_filenames(
    originals: Originals, tmp_path: Path, field: str
) -> None:
    """Rehashing and renaming tampered custody does not recreate its authority."""
    control = source_control(originals)
    control.acknowledge_writes = True
    with protocol_control(control) as endpoint:
        profile = profile_file(
            tmp_path, f"http://{endpoint.host}:{endpoint.port}", originals
        )
        raw = tmp_path / "response"
        retain(raw, provider_response())
        session = EventFeedbackSession(profile)
        session.import_response(raw)
        (session.directory / "ack.json").unlink()
        review_path = session.directory / "review.json"
        request_path = session.directory / "request.json"
        review, request = (
            decode(review_path.read_bytes()),
            decode(request_path.read_bytes()),
        )
        carrier = next(session.directory.glob("carrier-*.json"))
        if field in {"input", "provenance"}:
            object_value(review[field])["new_authority"] = "untrusted"
        elif field == "request-query":
            request["original_query"] = "Different question"
        elif field == "request-verdict":
            request["was_correct"] = False
        elif field == "request-verifier":
            request["verifier_by"] = "different"
        elif field == "request-procedure":
            object_value(request["evidence_ref"])["procedure_id"] = "different"
        else:
            review[field] = "invalid"
        review_path.write_bytes(canonical(review))
        object_value(request["evidence_ref"])["evidence_sha256"] = hashlib.sha256(
            review_path.read_bytes()
        ).hexdigest()
        request_path.write_bytes(canonical(request))
        destination = session.directory / (
            "carrier-" + payload_digest("event", request) + ".json"
        )
        if destination != carrier:
            carrier.rename(destination)
        with pytest.raises(FeedbackInvalidError):
            EventFeedbackSession(profile).replay()
        assert len([r for r in control.requests if r[0] == "POST"]) == 1


def test_replay_refuses_a_changed_original_directory_marker(
    originals: Originals, tmp_path: Path
) -> None:
    """The read-only recovery path must require its retained original inode."""
    control = source_control(originals)
    control.acknowledge_writes = True
    with protocol_control(control) as endpoint:
        profile = profile_file(
            tmp_path, f"http://{endpoint.host}:{endpoint.port}", originals
        )
        raw = tmp_path / "response"
        retain(raw, provider_response())
        session = EventFeedbackSession(profile)
        session.import_response(raw)
        marker = session.evidence.path / (".review-directory-" + session.directory.name)
        identity = decode(marker.read_bytes())
        identity["inode"] = 1
        marker.write_bytes(canonical(identity))
        with pytest.raises(FeedbackInvalidError):
            session.replay()
        assert len([r for r in control.requests if r[0] == "POST"]) == 1


def test_import_keeps_source_and_event_without_claiming_invocation(
    originals: Originals, tmp_path: Path
) -> None:
    """Retain current originals on the real import path without a fake reservation."""
    control = source_control(originals)
    control.acknowledge_writes = True
    with protocol_control(control) as endpoint:
        profile = profile_file(
            tmp_path, f"http://{endpoint.host}:{endpoint.port}", originals
        )
        raw = tmp_path / "response"
        retain(raw, provider_response())
        session = EventFeedbackSession(profile)
        session.import_response(raw)
    assert (session.directory / "source.original").read_bytes() == originals.source
    assert decode((session.directory / "event.json").read_bytes()) == originals.event
    assert decode((session.directory / "input.json").read_bytes()) == originals.inputs
    assert (
        decode((session.directory / "provenance.json").read_bytes())
        == originals.provenance
    )
    assert not (session.directory / "invocation-reserved").exists()


@pytest.mark.parametrize(
    "tamper",
    [
        "request-verdict",
        "carrier-signature",
        "carrier-envelope",
        "missing-carrier",
        "review-verdict",
    ],
)
def test_replay_refuses_changed_original_decision_and_carrier_before_post(
    originals: Originals, tmp_path: Path, tamper: str
) -> None:
    """A permissive response control cannot turn local tampering into a replay."""
    control = source_control(originals)
    control.acknowledge_writes = True
    with protocol_control(control) as endpoint:
        profile = profile_file(
            tmp_path, f"http://{endpoint.host}:{endpoint.port}", originals
        )
        raw = tmp_path / "response"
        retain(raw, provider_response())
        session = EventFeedbackSession(profile)
        session.import_response(raw)
        (session.directory / "ack.json").unlink()
        request_path = session.directory / "request.json"
        request = decode(request_path.read_bytes())
        carrier_path = next(session.directory.glob("carrier-*.json"))
        if tamper == "request-verdict":
            request["was_correct"] = False
            request_path.write_bytes(canonical(request))
        elif tamper == "missing-carrier":
            carrier_path.unlink()
        elif tamper == "review-verdict":
            review_path = session.directory / "review.json"
            review = decode(review_path.read_bytes())
            review["was_correct"] = False
            review_path.write_bytes(canonical(review))
            object_value(request["evidence_ref"])["evidence_sha256"] = hashlib.sha256(
                review_path.read_bytes()
            ).hexdigest()
            request_path.write_bytes(canonical(request))
            carrier_path.rename(
                session.directory
                / ("carrier-" + payload_digest("event", request) + ".json")
            )
        else:
            carrier = decode(carrier_path.read_bytes())
            if tamper == "carrier-signature":
                carrier["signature"] = "0" * 64
            else:
                object_value(object_value(carrier["carrier"])["envelope"])[
                    "recall_event_id"
                ] = "different"
            carrier_path.write_bytes(canonical(carrier))
        with pytest.raises(FeedbackInvalidError):
            EventFeedbackSession(profile).replay()
        assert (
            len([request for request in control.requests if request[0] == "POST"]) == 1
        )
        assert not any(
            request[1] == "/chat/completions" for request in control.requests
        )


def test_unknown_replay_does_not_create_custody_or_identity(
    originals: Originals, tmp_path: Path
) -> None:
    """A recovery read cannot reserve a never-created review identity."""
    control = source_control(originals)
    with protocol_control(control) as endpoint:
        profile = profile_file(
            tmp_path, f"http://{endpoint.host}:{endpoint.port}", originals
        )
        session = EventFeedbackSession(profile)
        before = set(session.evidence.path.iterdir())
        with pytest.raises(FeedbackInvalidError):
            session.replay()
        assert set(session.evidence.path.iterdir()) == before
        assert not session.directory.exists()
        assert not any(request[0] == "POST" for request in control.requests)


def test_prepared_extensions_refuse_before_contact(
    originals: Originals, tmp_path: Path
) -> None:
    """Undeclared prepared authority metadata cannot enter a session silently."""
    profile = profile_file(tmp_path, "http://localhost", originals)
    prepared = tmp_path / "prepared.json"
    value = decode(prepared.read_bytes())
    value["implicit_grant"] = "operator"
    prepared.write_bytes(canonical(value))
    with pytest.raises(FeedbackInvalidError):
        EventFeedbackSession(profile)


@pytest.mark.parametrize(
    "path",
    [
        "profile",
        "token_file",
        "signing_key_file",
        "evidence_directory",
        "prepared_file",
    ],
)
def test_cli_nul_paths_are_fixed_refusals(
    originals: Originals, tmp_path: Path, capsys: pytest.CaptureFixture[str], path: str
) -> None:
    """Malformed operator paths never escape as private Python diagnostics."""
    control = source_control(originals)
    control.acknowledge_writes = True
    with protocol_control(control) as endpoint:
        profile = profile_file(
            tmp_path, f"http://{endpoint.host}:{endpoint.port}", originals
        )
        raw = tmp_path / "response"
        retain(raw, provider_response())
        EventFeedbackSession(profile).import_response(raw)
        if path == "profile":
            argument = "\x00"
        else:
            config = decode(profile.read_bytes())
            config[path] = str(tmp_path / "private") + "\x00"
            profile.write_bytes(canonical(config))
            argument = str(profile)
        with pytest.raises(SystemExit) as stopped:
            main(["replay", "--profile", argument])
        assert stopped.value.code == 4
        output = capsys.readouterr()
        assert json.loads(output.out) == {
            "state": "REFUSED",
            "feedback_ack_qualified": False,
            "error": "Feedback evidence could not be validated.",
        }
        assert output.err == ""


def test_cli_service_refusal_has_its_own_exit_code(
    originals: Originals, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An authoritative refusal is distinct from abstention and write uncertainty."""
    control = source_control(originals)
    control.status = 401
    control.body = b'{"detail":"authentication required"}'
    with protocol_control(control) as endpoint:
        profile = profile_file(
            tmp_path, f"http://{endpoint.host}:{endpoint.port}", originals
        )
        raw = tmp_path / "response"
        retain(raw, provider_response())
        with pytest.raises(SystemExit) as stopped:
            main(
                [
                    "import-response",
                    "--profile",
                    str(profile),
                    "--original-response",
                    str(raw),
                ]
            )
        assert stopped.value.code == 4
        assert json.loads(capsys.readouterr().out)["state"] == "REFUSED"


def profile_file(tmp_path: Path, url: str, originals: Originals) -> Path:
    """Retain operator-controlled references without provisioning external rights."""
    token = tmp_path / "token"
    key = tmp_path / "key"
    prepared = tmp_path / "prepared.json"
    profile = tmp_path / "profile.json"
    root = registered_root(tmp_path / "reviews")
    retain(token, b"operator-token")
    retain(key, b"k" * 32)
    retain(
        prepared,
        canonical(
            {
                "input": originals.inputs,
                "provenance": originals.provenance,
                "evidence_id": "review",
                "submission_id": "submission",
            }
        ),
    )
    retain(
        profile,
        canonical(
            {
                "memory_url": url,
                "model_url": url,
                "token_file": str(token),
                "signing_key_file": str(key),
                "evidence_directory": str(root.path),
                "evidence_directory_identity": {
                    "device": root.device,
                    "inode": root.inode,
                },
                "prepared_file": str(prepared),
            }
        ),
    )
    return profile


def test_sdk_import_and_replay_use_identical_request(
    originals: Originals, tmp_path: Path
) -> None:
    """A new SDK instance rechecks current originals and replays the same submission."""
    control = source_control(originals)
    control.acknowledge_writes = True
    with protocol_control(control) as endpoint:
        profile = profile_file(
            tmp_path, f"http://{endpoint.host}:{endpoint.port}", originals
        )
        raw = tmp_path / "original-response"
        retain(raw, provider_response())
        session = EventFeedbackSession(profile)
        first = session.import_response(raw)
        again = EventFeedbackSession(profile).replay()
        assert first is not None and first.state == again.state == "FEEDBACK_RECORDED"
        assert first.ack == again.ack
        writes = [r[2] for r in control.requests if r[0] == "POST"]
        assert writes[0] == writes[1]
        control.routes["/recall/events/event"] = (
            401,
            b'{"detail":"authentication required"}',
            [],
        )
        with pytest.raises(FeedbackInvalidError):
            EventFeedbackSession(profile).replay()
        assert len([r for r in control.requests if r[0] == "POST"]) == 2


def test_cli_outputs_only_allowlisted_receipt_fields(
    originals: Originals,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Successful import emits receipt IDs without private query, key or rationale."""
    control = source_control(originals)
    control.acknowledge_writes = True
    with protocol_control(control) as endpoint:
        profile = profile_file(
            tmp_path, f"http://{endpoint.host}:{endpoint.port}", originals
        )
        raw = tmp_path / "original-response"
        retain(raw, provider_response())
        main(
            [
                "import-response",
                "--profile",
                str(profile),
                "--original-response",
                str(raw),
            ]
        )
        output = json.loads(capsys.readouterr().out)
        assert set(output) == {
            "state",
            "http_status",
            "feedback_ack_qualified",
            "outcome_id",
            "feedback_submission_id",
        }
        assert output["feedback_ack_qualified"] is True
        main(["replay", "--profile", str(profile)])
        assert json.loads(capsys.readouterr().out) == output


def test_cli_abstention_does_not_post_feedback(
    originals: Originals,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Preserve actual abstention without returning a correctness boolean."""
    control = source_control(originals)
    with protocol_control(control) as endpoint:
        profile = profile_file(
            tmp_path, f"http://{endpoint.host}:{endpoint.port}", originals
        )
        raw = tmp_path / "original-response"
        retain(raw, provider_response("ABSTAIN"))
        with pytest.raises(SystemExit) as stopped:
            main(
                [
                    "import-response",
                    "--profile",
                    str(profile),
                    "--original-response",
                    str(raw),
                ]
            )
        assert stopped.value.code == 3
        assert json.loads(capsys.readouterr().out) == {
            "state": "ABSTAINED",
            "feedback_ack_qualified": False,
        }
    assert not any(r[0] == "POST" for r in control.requests)


def test_cli_invalid_private_profile_has_safe_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Hide private filesystem details in caller-facing errors."""
    with pytest.raises(SystemExit) as stopped:
        main(["replay", "--profile", str(tmp_path / "missing")])
    assert stopped.value.code == 4
    assert "missing" not in capsys.readouterr().out


@pytest.mark.parametrize("fault", ["provider-content", "prepared-snippet"])
def test_cli_escaped_invalid_unicode_has_safe_refusal(
    originals: Originals,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fault: str,
) -> None:
    """Reject escaped surrogate data without a traceback or feedback write."""
    control = source_control(originals)
    with protocol_control(control) as endpoint:
        profile = profile_file(
            tmp_path, f"http://{endpoint.host}:{endpoint.port}", originals
        )
        response = json.loads(provider_response())
        if fault == "provider-content":
            response["choices"][0]["message"]["content"] = "\ud800"
        else:
            prepared_file = tmp_path / "prepared.json"
            prepared = json.loads(prepared_file.read_bytes())
            prepared["input"]["supplied_snippets"][0]["text"] = "\ud800"
            prepared_file.write_bytes(
                json.dumps(prepared, ensure_ascii=True).encode("ascii")
            )
        raw = tmp_path / "original-response"
        retain(raw, json.dumps(response, ensure_ascii=True).encode("ascii"))
        with pytest.raises(SystemExit) as stopped:
            main(
                [
                    "import-response",
                    "--profile",
                    str(profile),
                    "--original-response",
                    str(raw),
                ]
            )
        assert stopped.value.code == 4
        output = capsys.readouterr()
        assert json.loads(output.out) == {
            "state": "REFUSED",
            "feedback_ack_qualified": False,
            "error": "Feedback evidence could not be validated.",
        }
        assert output.err == ""
        assert not any(request[0] == "POST" for request in control.requests)


@pytest.mark.parametrize(
    "argv",
    [
        ["import-response", "--profile", "profile"],
        ["replay", "--profile", "profile", "--original-response", "raw"],
        ["evaluate", "--profile", "profile", "--original-response", "raw"],
    ],
)
def test_cli_rejects_ambiguous_action_arguments(argv: list[str]) -> None:
    """No implicit model call or import is selected by malformed arguments."""
    with pytest.raises(SystemExit) as stopped:
        main(argv)
    assert stopped.value.code == 2


def test_sdk_replay_refuses_missing_or_drifted_originals(
    originals: Originals, tmp_path: Path
) -> None:
    """Existing ACK cannot bypass retained review/request bindings."""
    control = source_control(originals)
    control.acknowledge_writes = True
    with protocol_control(control) as endpoint:
        profile = profile_file(
            tmp_path, f"http://{endpoint.host}:{endpoint.port}", originals
        )
        raw = tmp_path / "raw"
        retain(raw, provider_response())
        session = EventFeedbackSession(profile)
        session.import_response(raw)
        request_file = session.directory / "request.json"
        request = decode(request_file.read_bytes())
        request["feedback_submission_id"] = "different"
        request_file.write_bytes(canonical(request))
        with pytest.raises(FeedbackInvalidError):
            session.replay()
        assert len([r for r in control.requests if r[0] == "POST"]) == 1


def test_sdk_and_cli_evaluate_registered_transport(
    originals: Originals,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise the registered evaluation path using a declared provider protocol control."""
    control = source_control(originals)
    control.routes["/chat/completions"] = (200, provider_response(), [])
    control.acknowledge_writes = True
    with protocol_control(control) as endpoint:
        profile = profile_file(
            tmp_path, f"http://{endpoint.host}:{endpoint.port}", originals
        )
        main(["evaluate", "--profile", str(profile)])
    output = json.loads(capsys.readouterr().out)
    assert output["state"] == "FEEDBACK_RECORDED"
    assert len([r for r in control.requests if r[1] == "/chat/completions"]) == 1


@pytest.mark.parametrize("action", ["import-response", "replay"])
def test_cli_postwrite_receipt_custody_failure_preserves_uncertainty(
    originals: Originals,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    action: str,
) -> None:
    """A real local receipt collision after the HTTP write cannot become REFUSED."""
    control = source_control(originals)
    control.acknowledge_writes = True
    with protocol_control(control) as endpoint:
        profile = profile_file(
            tmp_path, f"http://{endpoint.host}:{endpoint.port}", originals
        )
        raw = tmp_path / "original-response"
        retain(raw, provider_response())
        session = EventFeedbackSession(profile)
        if action == "replay":
            control.acknowledge_writes = False
            control.status = 503
            control.body = b'{"contract_version":"source-bound-recall-v3","recording":"write_unknown","error":"ledger_unavailable"}'
            first = session.import_response(raw)
            assert first is not None and first.ack is None
            control.acknowledge_writes = True
        control.receipt_collision = session.directory / "ack.json"
        argv = [action, "--profile", str(profile)]
        if action == "import-response":
            argv += ["--original-response", str(raw)]
        with pytest.raises(SystemExit) as stopped:
            main(argv)
        assert stopped.value.code == 5
        assert json.loads(capsys.readouterr().out) == {
            "state": "WRITE_UNKNOWN",
            "feedback_ack_qualified": False,
        }


def test_unknown_then_cold_recovery_does_not_require_local_ack(
    originals: Originals,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reconcile possible writes through the service while preserving original IDs."""
    control = source_control(originals)
    control.status = 503
    control.body = b'{"contract_version":"source-bound-recall-v3","recording":"write_unknown","error":"ledger_unavailable"}'
    with protocol_control(control) as endpoint:
        profile = profile_file(
            tmp_path, f"http://{endpoint.host}:{endpoint.port}", originals
        )
        raw = tmp_path / "raw"
        retain(raw, provider_response())
        with pytest.raises(SystemExit) as stopped:
            main(
                [
                    "import-response",
                    "--profile",
                    str(profile),
                    "--original-response",
                    str(raw),
                ]
            )
        assert stopped.value.code == 5
        assert json.loads(capsys.readouterr().out)["state"] == "WRITE_UNKNOWN"
        control.acknowledge_writes = True
        main(["replay", "--profile", str(profile)])
        assert json.loads(capsys.readouterr().out)["feedback_ack_qualified"] is True


def test_remote_model_profile_is_not_selected(
    tmp_path: Path, originals: Originals
) -> None:
    """Refuse a private operator profile that attempts a remote provider fallback."""
    profile = profile_file(tmp_path, "http://localhost", originals)
    config = decode(profile.read_bytes())
    config["model_url"] = "https://example.com"
    profile.write_bytes(canonical(config))
    with pytest.raises(FeedbackInvalidError):
        EventFeedbackSession(profile)


@pytest.mark.parametrize("tamper", ["review-bytes", "review-input"])
def test_review_custody_and_prepared_input_are_bound(
    originals: Originals,
    tmp_path: Path,
    tamper: str,
) -> None:
    """The prior receipt cannot authorise changed original A bytes or review inputs."""
    control = source_control(originals)
    control.acknowledge_writes = True
    with protocol_control(control) as endpoint:
        profile = profile_file(
            tmp_path, f"http://{endpoint.host}:{endpoint.port}", originals
        )
        raw = tmp_path / "raw"
        retain(raw, provider_response())
        session = EventFeedbackSession(profile)
        session.import_response(raw)
        review_file = session.directory / "review.json"
        review = decode(review_file.read_bytes())
        object_value(review["input"])["candidate_id"] = "different"
        review_file.write_bytes(canonical(review))
        if tamper == "review-input":
            request_file = session.directory / "request.json"
            request = decode(request_file.read_bytes())
            object_value(request["evidence_ref"])["evidence_sha256"] = hashlib.sha256(
                review_file.read_bytes()
            ).hexdigest()
            request_file.write_bytes(canonical(request))
        with pytest.raises(FeedbackInvalidError):
            session.replay()
        assert len([r for r in control.requests if r[0] == "POST"]) == 1


def test_current_service_refusal_overrides_saved_receipt(
    originals: Originals, tmp_path: Path
) -> None:
    """A saved ACK cannot bypass a current producer/evidence write refusal."""
    control = source_control(originals)
    control.acknowledge_writes = True
    with protocol_control(control) as endpoint:
        profile = profile_file(
            tmp_path, f"http://{endpoint.host}:{endpoint.port}", originals
        )
        raw = tmp_path / "raw"
        retain(raw, provider_response())
        session = EventFeedbackSession(profile)
        first = session.import_response(raw)
        assert first is not None and first.ack is not None
        control.acknowledge_writes = False
        control.status = 503
        control.body = b'{"contract_version":"source-bound-recall-v3","recording":"not_recorded","error":"evidence_unavailable"}'
        refused = EventFeedbackSession(profile).replay()
        assert refused.state == "REFUSED" and refused.ack is None
        assert (session.directory / "ack.json").exists()


@pytest.mark.parametrize(
    "identity",
    [
        {},
        {"device": "1", "inode": 1},
        {"device": 1, "inode": "1"},
        {"device": True, "inode": 1},
    ],
)
def test_unbound_operator_custody_profile_refuses(
    tmp_path: Path, originals: Originals, identity: Object
) -> None:
    """A cold process never invents or refreshes the registered root binding."""
    profile = profile_file(tmp_path, "http://localhost", originals)
    config = decode(profile.read_bytes())
    config["evidence_directory_identity"] = identity
    profile.write_bytes(canonical(config))
    with pytest.raises(FeedbackInvalidError):
        EventFeedbackSession(profile)


def test_root_replacement_refuses_before_any_public_http(
    tmp_path: Path, originals: Originals
) -> None:
    """Evaluation, import and replay cannot adopt a root replaced after provisioning."""
    control = source_control(originals)
    with protocol_control(control) as endpoint:
        profile = profile_file(
            tmp_path, f"http://{endpoint.host}:{endpoint.port}", originals
        )
        root = tmp_path / "reviews"
        root.rename(tmp_path / "original-root")
        root.mkdir(mode=0o700)
        session = EventFeedbackSession(profile)
        for action in (
            session.evaluate,
            session.replay,
            lambda: session.import_response(tmp_path / "unused"),
        ):
            with pytest.raises(FeedbackInvalidError):
                action()
    assert control.requests == []


def test_postwrite_root_replacement_preserves_uncertainty(
    tmp_path: Path, originals: Originals
) -> None:
    """A real HTTP response cannot mask replaced local custody after a possible write."""
    control = source_control(originals)
    control.acknowledge_writes = True
    control.postwrite_relocate = tmp_path / "reviews"
    with protocol_control(control) as endpoint:
        profile = profile_file(
            tmp_path, f"http://{endpoint.host}:{endpoint.port}", originals
        )
        raw = tmp_path / "response"
        retain(raw, provider_response())
        with pytest.raises(FeedbackWriteUnknownError):
            EventFeedbackSession(profile).import_response(raw)
        writes = [r for r in control.requests if r[0] == "POST"]
        assert len(writes) == 1
        control.postwrite_relocate = None
        with pytest.raises(FeedbackInvalidError):
            EventFeedbackSession(profile).replay()
        assert len([r for r in control.requests if r[0] == "POST"]) == 1
    moved = tmp_path / "reviews-moved" / hashlib.sha256(b"review").hexdigest()
    assert len(list(moved.glob("attempts/*/request.json"))) == 1
    assert (moved / "review.json").exists()
    assert list((tmp_path / "reviews").iterdir()) == []
