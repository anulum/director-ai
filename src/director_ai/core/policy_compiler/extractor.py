# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — rule extractor Protocol + stub

"""Rule extraction boundary.

Concrete extractors (LLM-backed, grammar-based, human-curated)
plug in via :class:`RuleExtractor`. The :class:`StubExtractor`
that ships here parses the common compliance phrasings with a
small regex set so the compiler is testable without a model —
it is also a reasonable bootstrap for documents that already use
imperative "must not / shall not / maximum / at least" style.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from .rule import CompiledRule

_FORBIDDEN_PATTERN = re.compile(
    r"(?:must not|shall not|may not|do not|never)\s+([^.\n]{3,120})",
    re.IGNORECASE,
)
_MAX_LENGTH_PATTERN = re.compile(
    r"(?:maximum|at most|no more than|up to)(?:\s+of)?\s+(\d{2,6})\s+characters?",
    re.IGNORECASE,
)
_MIN_CITATIONS_PATTERN = re.compile(
    r"(?:at least|minimum of|must cite)\s+(\d{1,3})\s+(?:sources?|citations?|references?)",
    re.IGNORECASE,
)
_PATTERN_RULE = re.compile(
    r"block\s+(?:the\s+)?pattern\s+`([^`]+)`",
    re.IGNORECASE,
)


@runtime_checkable
class RuleExtractor(Protocol):
    """Contract for anything that turns a compliance document into
    a list of :class:`CompiledRule`."""

    def extract(self, document: str) -> list[CompiledRule]: ...


class StubExtractor:
    """Deterministic regex-based extractor.

    Recognises four phrasings used by most compliance style guides:

    * "must not / shall not / never X" → ``forbidden`` rule with X
      as the phrase.
    * "maximum N characters" → ``max_length`` rule.
    * "at least N citations / sources / references" →
      ``required_citations`` rule.
    * "block pattern \\`REGEX\\`" → ``pattern`` rule with the
      backticked regex.

    The stub is intentionally narrow — it trades coverage for
    determinism so unit tests and bootstrap deployments do not
    need an LLM. Rule IDs are stable SHA-256 hashes of
    ``(kind, value, source)`` so recompilation of the same
    document yields identical IDs (important for hot-swap diffs).
    """

    def __init__(self, *, source: str = "stub") -> None:
        self._source = source

    def extract(self, document: str) -> list[CompiledRule]:
        if not document or not document.strip():
            return []
        rules: list[CompiledRule] = []
        for m in _FORBIDDEN_PATTERN.finditer(document):
            phrase = m.group(1).strip().rstrip(",;:.")
            if phrase:
                rules.append(
                    CompiledRule(
                        id=self._hash("forbidden", phrase),
                        kind="forbidden",
                        value=phrase,
                        name=f"forbid:{phrase[:32]}",
                        source=self._source,
                    )
                )
        for m in _MAX_LENGTH_PATTERN.finditer(document):
            value = m.group(1)
            rules.append(
                CompiledRule(
                    id=self._hash("max_length", value),
                    kind="max_length",
                    value=value,
                    name="max_length",
                    source=self._source,
                )
            )
        for m in _MIN_CITATIONS_PATTERN.finditer(document):
            value = m.group(1)
            rules.append(
                CompiledRule(
                    id=self._hash("required_citations", value),
                    kind="required_citations",
                    value=value,
                    name="required_citations",
                    source=self._source,
                )
            )
        for m in _PATTERN_RULE.finditer(document):
            pattern = m.group(1)
            rules.append(
                CompiledRule(
                    id=self._hash("pattern", pattern),
                    kind="pattern",
                    value=pattern,
                    name=f"pattern:{pattern[:32]}",
                    source=self._source,
                )
            )
        return _dedup(rules)

    def _hash(self, kind: str, value: str) -> str:
        return hashlib.sha256(
            f"{kind}|{value}|{self._source}".encode()
        ).hexdigest()[:16]


def _dedup(rules: Iterable[CompiledRule]) -> list[CompiledRule]:
    """Deduplicate by ID while preserving insertion order — the same
    phrase occurring twice in a document should yield one rule."""
    seen: set[str] = set()
    out: list[CompiledRule] = []
    for r in rules:
        if r.id in seen:
            continue
        seen.add(r.id)
        out.append(r)
    return out
