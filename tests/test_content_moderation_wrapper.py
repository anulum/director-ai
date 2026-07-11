# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — content moderation wrapper tests

"""Production wrapper tests for PII redaction plus toxicity moderation."""

from __future__ import annotations

from director_ai.enterprise.moderation import ContentModerator, ModerationAction


def test_moderator_redacts_pii_and_reports_tenant_safe_metadata() -> None:
    moderator = ContentModerator(prefer_rust=False)

    result = moderator.moderate("Contact jane@example.com or 415-555-0101.")
    payload = result.to_dict()

    assert result.action is ModerationAction.REDACT
    assert result.blocked is False
    assert result.safe_text == "Contact [EMAIL] or [PHONE]."
    assert result.category_counts == {"email": 1, "phone": 1}
    assert payload["privacy"] == {
        "payload_classification": "tenant_safe",
        "raw_input_included": False,
    }
    assert "jane@example.com" not in repr(payload)
    assert "415-555-0101" not in repr(payload)


def test_moderator_blocks_toxicity_after_redacting_pii() -> None:
    moderator = ContentModerator(prefer_rust=False)

    result = moderator.moderate("Email jane@example.com, then go kill yourself.")

    assert result.action is ModerationAction.BLOCK
    assert result.blocked is True
    assert result.safe_text == "Email [EMAIL], then go kill yourself."
    assert result.category_counts["email"] == 1
    assert result.category_counts["self_harm_encouragement"] >= 1
    assert any(f.detector == "toxicity_keyword" for f in result.findings)


def test_moderator_warn_mode_reports_findings_without_blocking() -> None:
    moderator = ContentModerator(
        toxicity_action=ModerationAction.WARN,
        prefer_rust=False,
    )

    result = moderator.moderate("I will kill you next time.")

    assert result.action is ModerationAction.WARN
    assert result.blocked is False
    assert result.safe_text == "I will kill you next time."
    assert result.category_counts["threat"] >= 1


def test_moderator_allows_clean_text_with_empty_report() -> None:
    result = ContentModerator(prefer_rust=False).moderate("The deployment is healthy.")

    assert result.action is ModerationAction.ALLOW
    assert result.blocked is False
    assert result.safe_text == "The deployment is healthy."
    assert result.findings == ()
    assert result.category_counts == {}


def test_disabled_pii_detector_still_applies_toxicity_policy() -> None:
    moderator = ContentModerator(enable_pii=False, prefer_rust=False)

    result = moderator.moderate("Email jane@example.com, I will kill you.")

    assert result.action is ModerationAction.BLOCK
    assert result.blocked is True
    assert "jane@example.com" in result.safe_text
    assert "email" not in result.category_counts
    assert result.category_counts["threat"] >= 1


def test_moderate_returns_the_content_moderation_contract() -> None:
    from director_ai.enterprise.moderation import (
        ContentModerationFinding,
        ContentModerationResult,
    )

    result = ContentModerator(prefer_rust=False).moderate(
        "Contact jane@example.com about the invoice."
    )

    assert isinstance(result, ContentModerationResult)
    assert result.action is ModerationAction.REDACT
    assert "jane@example.com" not in result.safe_text
    assert result.findings
    finding = result.findings[0]
    assert isinstance(finding, ContentModerationFinding)
    assert finding.category == "email"
    assert finding.end > finding.start >= 0
