# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Â© Concepts 1996â€“2026 Miroslav Ĺ otek. All rights reserved.
# Â© Code 2020â€“2026 Miroslav Ĺ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI â€” Enterprise PII Redaction

"""Enterprise PII (Personally Identifiable Information) Redaction.

Detects and masks sensitive information (SSN, emails, phones, credit cards, PHI)
before processing or logging.
"""

from __future__ import annotations

import re


class PIIRedactor:
    """Enterprise redaction pipeline for sensitive string values."""

    _PATTERNS = [
        (re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"), "[EMAIL]"),
        (re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b"), "[CARD]"),
        # SSN before phone: 3-2-4 digit pattern with dash/dot separators
        (re.compile(r"\b\d{3}[-]\d{2}[-]\d{4}\b"), "[SSN]"),
        (
            re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
            "[PHONE]",
        ),
        (re.compile(r"\b(?:MRN|DOB|NHS)[\s:]*[A-Z0-9-]+\b", re.IGNORECASE), "[PHI]"),
    ]

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def redact(self, text: str) -> str:
        """Redact known PII patterns from the text string."""
        if not self.enabled or not text:
            return text

        for pat, repl in self._PATTERNS:
            text = pat.sub(repl, text)
        return text

    def __call__(self, text: str) -> str:
        return self.redact(text)
