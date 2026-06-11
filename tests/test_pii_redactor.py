# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — PII redaction report tests

"""Production redaction coverage for enterprise PII handling."""

from __future__ import annotations

from director_ai.enterprise.redactor import PIIRedactor


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
