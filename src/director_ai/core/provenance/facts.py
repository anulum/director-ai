# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — CitationFact

"""One citation: a source id, the content it cited, a timestamp,
and a SHA-256 hash of the canonical content for tamper detection.

The fact intentionally stores the raw ``content`` string as well
as the hash — the hash is the stable handle used by every
downstream data structure, but operators who need to display the
content back to a human need the original text too.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field


class FactVerificationError(ValueError):
    """Raised when a citation fact fails its integrity check."""


@dataclass(frozen=True)
class CitationFact:
    """One cited fact.

    ``source_id`` and ``content`` are caller-supplied. ``timestamp``
    defaults to ``time.time()`` at construction. ``content_hash``
    is derived at construction from ``source_id || \\x1f ||
    content`` so two facts from different sources can share the
    same content without colliding.
    """

    source_id: str
    content: str
    timestamp: float = field(default_factory=lambda: time.time())
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must be non-empty")
        if not self.content:
            raise ValueError("content must be non-empty")
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative")
        canonical = _canonical(self.source_id, self.content)
        expected_hash = hashlib.sha256(canonical).hexdigest()
        if not self.content_hash:
            # Using object.__setattr__ because the dataclass is frozen.
            object.__setattr__(self, "content_hash", expected_hash)
        elif self.content_hash != expected_hash:
            raise FactVerificationError(
                f"content_hash {self.content_hash!r} does not match "
                f"digest of source_id + content {expected_hash!r}"
            )

    def verify_integrity(self) -> None:
        """Raise :class:`FactVerificationError` when the stored
        hash no longer matches the canonical content. Callers
        run this after deserialising from an untrusted store."""
        canonical = _canonical(self.source_id, self.content)
        expected = hashlib.sha256(canonical).hexdigest()
        if self.content_hash != expected:
            raise FactVerificationError(
                f"content_hash {self.content_hash!r} != expected {expected!r}"
            )


def _canonical(source_id: str, content: str) -> bytes:
    """``source_id || 0x1f || content`` encoded UTF-8 — the unit
    separator rules out collisions between ``("ab", "cd")`` and
    ``("abc", "d")``."""
    return f"{source_id}\x1f{content}".encode("utf-8")
