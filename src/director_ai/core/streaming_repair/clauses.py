# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Streaming repair clause splitting

"""Split a generated answer into clauses that reassemble losslessly.

The repairer rewrites *only* the unsafe clause, so the split must be reversible:
joining the segments back together must reproduce the original text byte for
byte, including the sentence terminators and the whitespace between sentences.
Each segment therefore carries its own trailing terminator and spacing.
"""

from __future__ import annotations

import re

__all__ = ["join_clauses", "split_clauses"]

# A clause is a run of text up to and including a sentence terminator, plus any
# whitespace that follows it. The final segment may have no terminator. The
# pattern is exhaustive over the input, so the segments always rejoin exactly.
_CLAUSE = re.compile(r".*?[.!?]+(?:\s+|$)|.+$", re.DOTALL)


def split_clauses(text: str) -> list[str]:
    """Split ``text`` into clause segments that rejoin to the original.

    ``"".join(split_clauses(text)) == text`` for any input.
    """
    if not text:
        return []
    return _CLAUSE.findall(text)


def join_clauses(clauses: list[str]) -> str:
    """Reassemble clause segments into a single string."""
    return "".join(clauses)
