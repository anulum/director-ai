# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Shared Heuristic Patterns
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""Negation words and entity regex shared by lite_scorer.py and scorer.py."""

from __future__ import annotations

import re

NEGATION_WORDS = frozenset(
    {
        "not",
        "no",
        "never",
        "neither",
        "nobody",
        "nothing",
        "nowhere",
        "nor",
        "cannot",
        "can't",
        "don't",
        "doesn't",
        "didn't",
        "won't",
        "wouldn't",
        "shouldn't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "hasn't",
        "haven't",
        "hadn't",
    }
)

WORD_RE = re.compile(r"\b\w+\b")
ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
