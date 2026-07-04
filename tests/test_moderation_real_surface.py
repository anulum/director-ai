# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Real production-surface coverage for moderation detector policy wiring."""

from __future__ import annotations

import pytest

from director_ai.core.safety.moderation import (
    KeywordToxicityDetector,
    RegexPIIDetector,
)
from director_ai.core.safety.policy import Policy
from director_ai.enterprise.moderation import ContentModerator, ModerationAction


def test_policy_with_real_moderation_detectors_reports_pii_and_toxicity() -> None:
    """Policy.with_moderation should consume shipped detector implementations."""
    policy = Policy().with_moderation(
        [
            RegexPIIDetector(prefer_rust=False),
            KeywordToxicityDetector(prefer_rust=False),
        ],
    )

    violations = policy.check("Contact jane@example.com because I will kill you.")

    assert {violation.rule for violation in violations} == {
        "moderation:pii_regex:email",
        "moderation:toxicity_keyword:threat",
    }
    assert {violation.detail for violation in violations} == {"8-24", "33-48"}


def test_content_moderator_returns_tenant_safe_block_report() -> None:
    """ContentModerator should redact first and then block unsafe clean text."""
    moderator = ContentModerator(prefer_rust=False)

    result = moderator.moderate(
        "Send receipt to jane@example.com and I will kill you.",
    )
    payload = result.to_dict()

    assert result.action is ModerationAction.BLOCK
    assert result.blocked is True
    assert result.safe_text == "Send receipt to [EMAIL] and I will kill you."
    assert result.category_counts == {"email": 1, "threat": 1}
    assert payload["privacy"] == {
        "payload_classification": "tenant_safe",
        "raw_input_included": False,
    }
    assert "jane@example.com" not in repr(payload)


def test_content_moderator_clean_text_uses_empty_public_report() -> None:
    """ContentModerator should allow clean text without synthetic findings."""
    result = ContentModerator(prefer_rust=False).moderate(
        "The deployment checklist is complete.",
    )

    assert result.action is ModerationAction.ALLOW
    assert result.blocked is False
    assert result.safe_text == "The deployment checklist is complete."
    assert result.findings == ()
    assert result.category_counts == {}


def test_content_moderator_rejects_invalid_public_action() -> None:
    """ContentModerator should fail fast for unsupported toxicity actions."""
    with pytest.raises(ValueError, match="toxicity_action must be block or warn"):
        ContentModerator(toxicity_action=ModerationAction.REDACT)
