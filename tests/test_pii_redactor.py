# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — PII redaction report tests

"""Production redaction coverage for enterprise PII handling."""

from __future__ import annotations

from director_ai.core.redactor import (
    PIIRedactionFinding,
    PIIRedactor,
    _select_non_overlapping,
)


def test_select_non_overlapping_drops_invalid_and_overlapping_spans() -> None:
    # The earliest, longest valid span wins; a degenerate span (end <= start) is
    # rejected outright and a later span overlapping a pick is dropped.
    findings = [
        PIIRedactionFinding(
            detector="regex", category="bad", start=12, end=12, replacement="!"
        ),
        PIIRedactionFinding(
            detector="regex", category="email", start=0, end=10, replacement="X"
        ),
        PIIRedactionFinding(
            detector="regex", category="phone", start=5, end=15, replacement="Y"
        ),
        PIIRedactionFinding(
            detector="regex", category="ssn", start=20, end=29, replacement="Z"
        ),
    ]
    selected = _select_non_overlapping(findings)
    assert [(f.start, f.end) for f in selected] == [(0, 10), (20, 29)]


def test_redact_with_report_masks_stable_categories_and_counts_findings() -> None:
    text = (
        "Email jane.doe@example.com, call +1 415 555 0101, "
        "card 4111-1111-1111-1111, ssn 123-45-6789, "
        "iban CH9300762011623852957, host 10.0.0.5, MRN A-12345."
    )

    report = PIIRedactor(prefer_rust=False).redact_with_report(text)

    assert report.redacted_text == (
        "Email [EMAIL], call [PHONE], card [CARD], ssn [SSN], "
        "iban [IBAN], host [IPV4], [PHI]."
    )
    assert report.category_counts == {
        "card": 1,
        "email": 1,
        "iban": 1,
        "ipv4": 1,
        "phi": 1,
        "phone": 1,
        "ssn": 1,
    }
    assert report.redacted is True


def test_report_payload_is_tenant_safe_and_excludes_raw_pii_values() -> None:
    report = PIIRedactor(prefer_rust=False).redact_with_report(
        "Reach Miroslav at miroslav@example.com or 415-555-0101."
    )

    payload = report.to_dict()

    assert payload["redacted"] is True
    assert payload["redacted_text"] == "Reach Miroslav at [EMAIL] or [PHONE]."
    assert payload["privacy"] == {
        "payload_classification": "tenant_safe",
        "raw_payload_included": False,
    }
    assert "miroslav@example.com" not in repr(payload)
    assert "415-555-0101" not in repr(payload)
    assert payload["findings"] == [
        {
            "category": "email",
            "detector": "pii_regex",
            "end": 38,
            "replacement": "[EMAIL]",
            "score": 1.0,
            "start": 18,
        },
        {
            "category": "phone",
            "detector": "pii_regex",
            "end": 54,
            "replacement": "[PHONE]",
            "score": 1.0,
            "start": 42,
        },
    ]


def test_disabled_redactor_returns_empty_tenant_safe_report() -> None:
    report = PIIRedactor(enabled=False).redact_with_report("Email a@example.com")

    assert report.redacted_text == "Email a@example.com"
    assert report.findings == ()
    assert report.category_counts == {}
    assert report.to_dict()["privacy"]["raw_payload_included"] is False


def test_overlapping_matches_keep_longest_first_without_double_redaction() -> None:
    report = PIIRedactor(prefer_rust=False).redact_with_report(
        "Use 4111-1111-1111-1111."
    )

    assert report.redacted_text == "Use [CARD]."
    assert report.category_counts == {"card": 1}
    assert len(report.findings) == 1


def test_enterprise_redactor_shim_reexports_core() -> None:
    # Back-compat: the old import path still resolves to the moved core class.
    from director_ai.enterprise.redactor import PIIRedactor as ShimRedactor

    assert ShimRedactor is PIIRedactor


def test_redactor_call_builds_default_detector_and_redacts() -> None:
    # No explicit detectors -> _resolve_detectors builds the default; __call__
    # delegates to redact.
    redactor = PIIRedactor(prefer_rust=False)
    out = redactor("Reach me at alice@example.com please")
    assert "alice@example.com" not in out


def test_explicit_detectors_are_used_without_building_defaults() -> None:
    # An injected detector tuple is returned as-is, never replaced by the default.
    from director_ai.core.safety.moderation.pii import RegexPIIDetector

    detector = RegexPIIDetector(prefer_rust=False)
    redactor = PIIRedactor(prefer_rust=False, detectors=(detector,))

    assert redactor._resolve_detectors() == (detector,)
    report = redactor.redact_with_report("Email bob@example.com")
    assert report.redacted_text == "Email [EMAIL]"
