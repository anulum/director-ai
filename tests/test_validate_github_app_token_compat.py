# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — GitHub App token compatibility validator tests

from __future__ import annotations

import re

import tools.validate_github_app_token_compat as validator


def _hits(pattern: re.Pattern[str], text: str) -> bool:
    return bool(pattern.search(text))


def test_flags_fixed_length_ghs_regexes() -> None:
    sample = "token_pattern = r'ghs_[A-Za-z0-9]{40}'"
    assert any(_hits(regex, sample) for regex, _ in validator.FAIL_PATTERNS)


def test_flags_length_40_assertion() -> None:
    sample = "if len(token) == 40: return True"
    assert any(_hits(regex, sample) for regex, _ in validator.FAIL_PATTERNS)


def test_accepts_recommended_dual_format_regex() -> None:
    sample = r"ghs_[A-Za-z0-9\._]{36,}"
    assert not any(_hits(regex, sample) for regex, _ in validator.FAIL_PATTERNS)


def test_repository_has_no_brittle_assumptions() -> None:
    violations = validator.validate()
    assert violations == []
