# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Event feedback protocol tests
"""Verify completed-response interpretation and whole independently signed review custody."""

from __future__ import annotations

import base64
import hashlib
import hmac
from pathlib import Path

import pytest

from director_ai.agentic.event_feedback.codec import (
    FeedbackInvalidError,
    Json,
    array,
    canonical,
    decode,
    object_value,
)
from director_ai.agentic.event_feedback.originals import Originals
from director_ai.agentic.event_feedback.producer import (
    evaluate,
    interpret_response,
    retain_review,
)
from director_ai.agentic.event_feedback.transport import HttpEndpoint

from .conftest import (
    ProtocolControl,
    protocol_control,
    provider_response,
    registered_root,
)


@pytest.mark.parametrize("oversized", [False, True])
def test_invalid_received_provider_response_is_retained_without_a_verdict(
    originals: Originals, tmp_path: Path, oversized: bool
) -> None:
    """Retain actual ambiguous response evidence or a labelled oversized prefix."""
    raw = b"x" * (2 * 1048576) if oversized else provider_response()
    control = ProtocolControl(
        body=raw, headers=[] if oversized else [("X-Dup", "a"), ("X-Dup", "b")]
    )
    root = registered_root(tmp_path / "reviews")
    key = tmp_path / "key"
    key.write_bytes(b"k" * 32)
    key.chmod(0o600)
    with protocol_control(control) as endpoint:
        endpoint.timeout = None
        with pytest.raises(FeedbackInvalidError):
            evaluate(endpoint, originals, "review", "submission", key, root)
    folder = root.path / hashlib.sha256(b"review").hexdigest()
    name = "provider_response.prefix" if oversized else "provider_response.raw"
    assert (folder / name).read_bytes() == raw[:1048576]
    assert (
        decode((folder / "provider_transport.json").read_bytes())["complete"]
        is not oversized
    )
    assert (folder / "invocation-reserved").is_file()
    assert not (folder / "review.json").exists()
    assert len(control.requests) == 1


@pytest.mark.parametrize("verdict", ["SUPPORTED", "UNSUPPORTED", "ABSTAIN"])
def test_completed_decisions_remain_distinct(verdict: str) -> None:
    """Do not convert genuine abstention shape into a false correctness label."""
    assert (
        interpret_response(provider_response(verdict), "qwen-model", 1)["decision"]
        == verdict
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "other"),
        ("choices", []),
        ("usage", {"prompt_tokens": True, "completion_tokens": 3, "total_tokens": 4}),
        ("usage", {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 6}),
        ("usage", {"prompt_tokens": -1, "completion_tokens": 3, "total_tokens": 2}),
    ],
)
def test_provider_identity_finish_usage_are_required(field: str, value: Json) -> None:
    """Refuse drifted model, missing completion and coerced or inconsistent usage."""
    response = decode(provider_response())
    response[field] = value
    with pytest.raises(FeedbackInvalidError):
        interpret_response(canonical(response), "qwen-model", 1)


@pytest.mark.parametrize(
    "field,value",
    [("finish_reason", "length"), ("message", {"content": "{}", "refusal": "refused"})],
)
def test_truncated_or_refused_completion_is_not_a_review(
    field: str, value: Json
) -> None:
    """Incomplete or refused model responses retain their actual failure."""
    response = decode(provider_response())
    object_value(array(response["choices"])[0])[field] = value
    with pytest.raises(FeedbackInvalidError):
        interpret_response(canonical(response), "qwen-model", 1)


@pytest.mark.parametrize("verdict", ["wrong", [], True])
def test_verdict_shape_is_checked_before_membership(verdict: Json) -> None:
    """An untrusted list or boolean verdict gives a safe protocol error."""
    response = decode(provider_response())
    message = object_value(object_value(array(response["choices"])[0])["message"])
    decision = decode(str(message["content"]).encode())
    decision["decision"] = verdict
    message["content"] = canonical(decision).decode()
    with pytest.raises(FeedbackInvalidError):
        interpret_response(canonical(response), "qwen-model", 1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("support", "wrong"),
        ("snippet_indexes", [True]),
        ("snippet_indexes", [1]),
        ("snippet_indexes", [0, 0]),
        ("snippet_indexes", []),
        ("claim", ""),
        ("explanation", ""),
    ],
)
def test_material_claim_binding_is_required(field: str, value: Json) -> None:
    """Validate each supplied snippet reference without making a truth judgement."""
    response = decode(provider_response())
    message = object_value(object_value(array(response["choices"])[0])["message"])
    decision = decode(str(message["content"]).encode())
    object_value(array(decision["claim_checks"])[0])[field] = value
    message["content"] = canonical(decision).decode()
    with pytest.raises(FeedbackInvalidError):
        interpret_response(canonical(response), "qwen-model", 1)


def test_consistent_whole_verdict_is_required() -> None:
    """Mixed abstention and incomplete unsupported verdicts never yield a boolean."""
    for verdict, support in (
        ("SUPPORTED", "UNSUPPORTED"),
        ("UNSUPPORTED", "SUPPORTED"),
        ("UNSUPPORTED", "ABSTAIN"),
    ):
        response = decode(provider_response())
        message = object_value(object_value(array(response["choices"])[0])["message"])
        decision = decode(str(message["content"]).encode())
        decision["decision"] = verdict
        object_value(array(decision["claim_checks"])[0])["support"] = support
        message["content"] = canonical(decision).decode()
        with pytest.raises(FeedbackInvalidError):
            interpret_response(canonical(response), "qwen-model", 1)


def test_full_original_and_separate_carrier_are_signed(
    originals: Originals, tmp_path: Path
) -> None:
    """Independently verify every original byte digest and the whole HMAC carrier."""
    key = tmp_path / "signing"
    key.write_bytes(b"k" * 32)
    key.chmod(0o600)
    raw = provider_response()
    submission = retain_review(
        originals,
        raw,
        "review",
        "submission",
        key,
        registered_root(tmp_path / "reviews"),
    )
    assert submission is not None
    a = (submission.directory / "review.json").read_bytes()
    review = decode(a)
    assert base64.b64decode(str(review["provider_response_b64"])) == raw
    reference = object_value(submission.request["evidence_ref"])
    assert reference["evidence_sha256"] == hashlib.sha256(a).hexdigest()
    files = list(submission.directory.glob("carrier-*.json"))
    assert len(files) == 1
    wrapper = decode(files[0].read_bytes())
    carrier = object_value(wrapper["carrier"])
    assert (
        carrier["input"] == originals.inputs
        and carrier["provenance"] == originals.provenance
    )
    assert (
        wrapper["signature"]
        == hmac.new(
            b"k" * 32, b"remanentia-review-carrier-v1\n" + canonical(carrier), "sha256"
        ).hexdigest()
    )
    assert (
        retain_review(
            originals,
            raw,
            "review",
            "submission",
            key,
            registered_root(tmp_path / "reviews"),
        )
        == submission
    )
    with pytest.raises(FeedbackInvalidError):
        retain_review(
            originals,
            provider_response("UNSUPPORTED"),
            "review",
            "submission",
            key,
            registered_root(tmp_path / "reviews"),
        )


def test_abstention_has_no_signed_feedback(
    originals: Originals, tmp_path: Path
) -> None:
    """Retain the actual abstaining response without provisioning a label or key."""
    result = retain_review(
        originals,
        provider_response("ABSTAIN"),
        "review",
        "submission",
        tmp_path / "absent-key",
        registered_root(tmp_path / "reviews"),
    )
    assert result is None
    folder = (
        registered_root(tmp_path / "reviews").path
        / hashlib.sha256(b"review").hexdigest()
    )
    assert (
        not (folder / "review.json").exists() and not (folder / "request.json").exists()
    )
    assert decode((folder / "disposition.json").read_bytes())["state"] == "ABSTAINED"


def test_short_signing_key_refuses(originals: Originals, tmp_path: Path) -> None:
    """Refuse an unqualified signing key before producing an admissible carrier."""
    key = tmp_path / "key"
    key.write_bytes(b"short")
    key.chmod(0o600)
    with pytest.raises(FeedbackInvalidError):
        retain_review(
            originals,
            provider_response(),
            "review",
            "submission",
            key,
            registered_root(tmp_path / "reviews"),
        )


def test_reserved_invocation_is_not_automatically_repeated(
    originals: Originals, tmp_path: Path
) -> None:
    """A controlled provider exercises request custody without claiming genuine AI."""
    key = tmp_path / "key"
    key.write_bytes(b"k" * 32)
    key.chmod(0o600)
    control = ProtocolControl(body=provider_response())
    with protocol_control(control) as endpoint:
        endpoint.timeout = None
        submission = evaluate(
            endpoint,
            originals,
            "review",
            "submission",
            key,
            registered_root(tmp_path / "reviews"),
        )
        with pytest.raises(FeedbackInvalidError):
            evaluate(
                endpoint,
                originals,
                "review",
                "submission",
                key,
                registered_root(tmp_path / "reviews"),
            )
    assert submission is not None and len(control.requests) == 1
    request = decode(control.requests[0][2])
    assert set(request) == {"model", "stream", "messages"}
    assert request["model"] == "qwen-model" and request["stream"] is False
    messages = array(request["messages"])
    assert object_value(messages[0])["role"] == "system"
    user = decode(str(object_value(messages[1])["content"]).encode())
    assert (
        object_value(user["original_source"])["sha256"]
        == hashlib.sha256(originals.source).hexdigest()
    )
    assert (submission.directory / "invocation-reserved").is_file()


def test_provider_failure_reserves_original_invocation(
    originals: Originals, tmp_path: Path
) -> None:
    """A failed HTTP completion keeps its original response and prevents repair calls."""
    key = tmp_path / "key"
    key.write_bytes(b"k" * 32)
    key.chmod(0o600)
    control = ProtocolControl(body=b"refused", status=503)
    with protocol_control(control) as endpoint:
        endpoint.timeout = None
        with pytest.raises(FeedbackInvalidError):
            evaluate(
                endpoint,
                originals,
                "review",
                "submission",
                key,
                registered_root(tmp_path / "reviews"),
            )
        with pytest.raises(FeedbackInvalidError):
            evaluate(
                endpoint,
                originals,
                "review",
                "submission",
                key,
                registered_root(tmp_path / "reviews"),
            )
    assert len(control.requests) == 1
    folder = (
        registered_root(tmp_path / "reviews").path
        / hashlib.sha256(b"review").hexdigest()
    )
    assert (folder / "provider_response.raw").read_bytes() == b"refused"


def test_remote_or_timed_model_profile_refuses_before_call(
    originals: Originals, tmp_path: Path
) -> None:
    """No remote fallback or response deadline replaces the selected local procedure."""
    for endpoint in (
        HttpEndpoint("https://example.com", timeout=None),
        HttpEndpoint("http://localhost"),
    ):
        with pytest.raises(FeedbackInvalidError):
            evaluate(
                endpoint,
                originals,
                "review",
                "submission",
                tmp_path / "key",
                registered_root(tmp_path / "reviews"),
            )


def test_empty_claim_set_is_not_a_completed_judgement() -> None:
    """A response with no material claim checks cannot mint a boolean."""
    response = decode(provider_response())
    message = object_value(object_value(array(response["choices"])[0])["message"])
    decision = decode(str(message["content"]).encode())
    decision["claim_checks"] = []
    message["content"] = canonical(decision).decode()
    with pytest.raises(FeedbackInvalidError):
        interpret_response(canonical(response), "qwen-model", 1)


def test_oversize_original_review_refuses_without_truncation(
    originals: Originals, tmp_path: Path
) -> None:
    """Preserve the complete provider response and refuse oversized A/B admission."""
    response = decode(provider_response())
    message = object_value(object_value(array(response["choices"])[0])["message"])
    decision = decode(str(message["content"]).encode())
    object_value(array(decision["claim_checks"])[0])["explanation"] = "x" * 800000
    message["content"] = canonical(decision).decode()
    raw = canonical(response)
    key = tmp_path / "key"
    key.write_bytes(b"k" * 32)
    key.chmod(0o600)
    with pytest.raises(FeedbackInvalidError):
        retain_review(
            originals,
            raw,
            "review",
            "submission",
            key,
            registered_root(tmp_path / "reviews"),
        )
    folder = (
        registered_root(tmp_path / "reviews").path
        / hashlib.sha256(b"review").hexdigest()
    )
    assert (folder / "provider_response.raw").read_bytes() == raw
    assert not (folder / "review.json").exists()


def test_incomplete_reservation_and_unqualified_inputs_do_not_call_provider(
    originals: Originals,
    tmp_path: Path,
) -> None:
    """Refuse orphaned invocation reservations, bad keys and invalid source encodings."""
    endpoint = HttpEndpoint("http://localhost", timeout=None)
    key = tmp_path / "key"
    key.write_bytes(b"short")
    key.chmod(0o600)
    with pytest.raises(FeedbackInvalidError):
        evaluate(
            endpoint,
            originals,
            "review",
            "submission",
            key,
            registered_root(tmp_path / "reviews"),
        )
    key.write_bytes(b"k" * 32)
    broken = Originals(originals.inputs, originals.provenance, originals.event, b"\xff")
    with pytest.raises(FeedbackInvalidError):
        evaluate(
            endpoint,
            broken,
            "review",
            "submission",
            key,
            registered_root(tmp_path / "reviews"),
        )
    folder = (
        registered_root(tmp_path / "reviews").path
        / hashlib.sha256(b"review").hexdigest()
    )
    folder.mkdir(mode=0o700, parents=True)
    (folder / "invocation-reserved").write_bytes(b"reserved")
    with pytest.raises(FeedbackInvalidError):
        evaluate(
            endpoint,
            originals,
            "review",
            "submission",
            key,
            registered_root(tmp_path / "reviews"),
        )
