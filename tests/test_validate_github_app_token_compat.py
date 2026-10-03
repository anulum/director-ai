# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — GitHub App token compatibility validator tests

"""Check token-pattern compatibility rules and the current repository scan."""

from __future__ import annotations

import re

import tools.validate_github_app_token_compat as validator


def _hits(pattern: re.Pattern[str], text: str) -> bool:
    """Check whether the supplied compiled pattern matches the sample text."""
    return bool(pattern.search(text))


def test_flags_fixed_length_ghs_regexes() -> None:
    """Reject the supplied token pattern that assumes forty characters."""
    sample = "token_pattern = r'ghs_[A-Za-z0-9]{40}'"
    assert any(_hits(regex, sample) for regex, _ in validator.FAIL_PATTERNS)


def test_flags_length_40_assertion() -> None:
    """Reject the supplied token-length comparison against forty."""
    sample = "if len(token) == 40: return True"
    assert any(_hits(regex, sample) for regex, _ in validator.FAIL_PATTERNS)


def test_accepts_recommended_dual_format_regex() -> None:
    """Allow the supplied variable-length pattern covering both token formats."""
    sample = r"ghs_[A-Za-z0-9\._]{36,}"
    assert not any(_hits(regex, sample) for regex, _ in validator.FAIL_PATTERNS)


def test_repository_has_no_brittle_assumptions() -> None:
    """Require the real repository scan to report no token-format violations."""
    violations = validator.validate()
    assert violations == []
