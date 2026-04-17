# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — compiled rule data model

"""Compiled-rule dataclass shared by the extractor, compiler, and
registry. Kept in its own module so both :mod:`extractor` and
:mod:`compiler` can import it without a circular dependency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RuleKind = Literal["forbidden", "pattern", "max_length", "required_citations"]
RuleAction = Literal["block", "warn", "redact"]

_VALID_KINDS: frozenset[str] = frozenset(
    ("forbidden", "pattern", "max_length", "required_citations")
)
_VALID_ACTIONS: frozenset[str] = frozenset(("block", "warn", "redact"))


@dataclass(frozen=True)
class CompiledRule:
    """One rule extracted from a compliance document.

    Parameters
    ----------
    id :
        Stable identifier — SHA-256 digest of (kind, value, source)
        when produced by :class:`StubExtractor`. Callers that plug
        in their own extractor are free to use any stable string.
    kind :
        One of ``forbidden`` / ``pattern`` / ``max_length`` /
        ``required_citations``. Validated at construction.
    value :
        Rule payload. Phrase for ``forbidden``, regex for
        ``pattern``, integer-as-string for ``max_length`` and
        ``required_citations``.
    name :
        Human-readable handle that shows up in :class:`Violation`
        entries. Keep under 64 characters.
    action :
        ``block`` (default), ``warn``, or ``redact``. Validated.
    threshold :
        Calibrated cut-off for channels that need one (e.g. a
        classifier confidence). ``None`` means the rule is
        deterministic (the default for the four built-in kinds).
    source :
        Free-form pointer to the document chunk the rule came
        from — useful for audit trails. Kept as a string so
        callers do not have to model documents in a particular way.
    """

    id: str
    kind: RuleKind
    value: str
    name: str
    action: RuleAction = "block"
    threshold: float | None = None
    source: str = ""

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(
                f"CompiledRule.kind must be one of {sorted(_VALID_KINDS)}; "
                f"got {self.kind!r}"
            )
        if self.action not in _VALID_ACTIONS:
            raise ValueError(
                f"CompiledRule.action must be one of {sorted(_VALID_ACTIONS)}; "
                f"got {self.action!r}"
            )
        if self.threshold is not None and not 0.0 <= self.threshold <= 1.0:
            raise ValueError(
                f"CompiledRule.threshold must be in [0, 1] or None; got {self.threshold!r}"
            )
        if not self.id:
            raise ValueError("CompiledRule.id must be non-empty")
        if not self.name:
            raise ValueError("CompiledRule.name must be non-empty")
