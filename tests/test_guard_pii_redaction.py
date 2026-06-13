# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — SDK-path PII redaction parity tests

from __future__ import annotations

from director_ai.core.config import DirectorConfig
from director_ai.core.types import CoherenceScore
from director_ai.guard import ProductionGuard


class _CapturingScorer:
    """Records the (prompt, response) the guard actually scores."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, str]] = []

    def review(self, prompt: str, response: str, *args, **kwargs):
        self.seen.append((prompt, response))
        return True, CoherenceScore(
            score=0.9, approved=True, h_logical=0.1, h_factual=0.1
        )


def _guard(redact: bool) -> tuple[ProductionGuard, _CapturingScorer]:
    guard = ProductionGuard(config=DirectorConfig(redact_pii=redact))
    cap = _CapturingScorer()
    guard._scorer = cap  # capture what reaches the scorer
    return guard, cap


def test_sdk_redacts_pii_before_scoring():
    guard, cap = _guard(redact=True)
    result = guard.check(
        "email me at john.doe@example.com",
        "your number is 555-123-4567",
    )
    scored_prompt, scored_response = cap.seen[-1]
    # the scorer must never see the raw PII — parity with the REST /v1/review path
    assert "john.doe@example.com" not in scored_prompt
    assert "[EMAIL]" in scored_prompt
    assert "555-123-4567" not in scored_response
    assert "[PHONE]" in scored_response
    assert result.approved is True


def test_sdk_no_redaction_when_disabled():
    guard, cap = _guard(redact=False)
    guard.check("email me at john.doe@example.com", "plain response")
    scored_prompt, _ = cap.seen[-1]
    # default behaviour unchanged — PII passes through when redact_pii is off
    assert "john.doe@example.com" in scored_prompt


def test_redactor_enabled_follows_config():
    assert ProductionGuard(config=DirectorConfig(redact_pii=True))._redactor.enabled
    assert not ProductionGuard(
        config=DirectorConfig(redact_pii=False)
    )._redactor.enabled
