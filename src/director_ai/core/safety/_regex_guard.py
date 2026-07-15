# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ReDoS guard for operator-supplied regexes

"""Validate operator-supplied regexes before compilation (KIMI-B).

Policy YAML patterns and sanitiser ``extra_patterns``/``allowlist``
entries come from deployment configuration — a semi-trusted source.
A pattern like ``(a+)+$`` compiles fine but backtracks catastrophically
on crafted input, so every such pattern passes through
:func:`ensure_safe_pattern`, which rejects the classic ReDoS shape:
an unbounded repeat nested inside another unbounded repeat. Bound the
inner repetition (``(?:a{1,64})+`` style) or restructure the pattern
to clear the guard; the error says which construct was rejected.
"""

from __future__ import annotations

import re

# The package floor is Python 3.11, where the sre parser lives at
# re._parser (private but stable across 3.11-3.13; the guard's tests
# exercise it on every supported version in CI).
from re import _parser as _sre_parser  # type: ignore[attr-defined]

__all__ = ["ensure_safe_pattern"]

#: Operator patterns have no business being longer than this.
_MAX_PATTERN_LENGTH = 4096

#: A bounded repeat above this count is treated as unbounded — ``{2,4000}``
#: backtracks just as catastrophically as ``+`` when nested.
_UNBOUNDED_FLOOR = 256


def ensure_safe_pattern(
    pattern: str,
    *,
    source: str,
    flags: int = 0,
) -> re.Pattern[str]:
    """Validate and compile one operator-supplied regex.

    Parameters
    ----------
    pattern : str
        The regex text from configuration.
    source : str
        Human-readable origin (policy name, sanitiser category) used in
        error messages.
    flags : int
        Flags forwarded to :func:`re.compile`.

    Returns
    -------
    re.Pattern[str]
        The compiled pattern.

    Raises
    ------
    ValueError
        If the pattern does not compile, exceeds the length bound, or
        contains an unbounded repeat nested inside another unbounded
        repeat (the catastrophic-backtracking shape).
    """
    if len(pattern) > _MAX_PATTERN_LENGTH:
        raise ValueError(
            f"regex for {source} is {len(pattern)} chars long "
            f"(limit {_MAX_PATTERN_LENGTH})"
        )
    try:
        parsed = _sre_parser.parse(pattern, flags)
    except re.error as exc:
        raise ValueError(f"invalid regex for {source}: {exc}") from exc
    if _has_nested_unbounded_repeat(parsed, inside_unbounded=False):
        raise ValueError(
            f"regex for {source} nests an unbounded repeat inside another "
            "unbounded repeat (catastrophic backtracking, e.g. '(a+)+'); "
            "bound the repetition, e.g. '{1,64}'"
        )
    return re.compile(pattern, flags)


def _has_nested_unbounded_repeat(parsed: object, *, inside_unbounded: bool) -> bool:
    """Walk one parsed subpattern for the nested-unbounded-repeat shape."""
    for op, arg in parsed:  # type: ignore[attr-defined]
        name = str(op)
        if name in ("MAX_REPEAT", "MIN_REPEAT", "POSSESSIVE_REPEAT"):
            _min_count, max_count, subpattern = arg
            unbounded = (
                max_count == _sre_parser.MAXREPEAT or max_count >= _UNBOUNDED_FLOOR
            )
            if unbounded and inside_unbounded:
                return True
            if _has_nested_unbounded_repeat(
                subpattern, inside_unbounded=inside_unbounded or unbounded
            ):
                return True
        elif name == "SUBPATTERN":
            if _has_nested_unbounded_repeat(arg[3], inside_unbounded=inside_unbounded):
                return True
        elif name == "BRANCH":
            for alternative in arg[1]:
                if _has_nested_unbounded_repeat(
                    alternative, inside_unbounded=inside_unbounded
                ):
                    return True
        elif name in ("ASSERT", "ASSERT_NOT"):
            if _has_nested_unbounded_repeat(arg[1], inside_unbounded=inside_unbounded):
                return True
        elif name == "ATOMIC_GROUP":
            if _has_nested_unbounded_repeat(arg, inside_unbounded=inside_unbounded):
                return True
    return False
