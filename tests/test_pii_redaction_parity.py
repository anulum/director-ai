# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — PII detector Rust/Python parity

from __future__ import annotations

import pytest

from director_ai.core.safety.moderation.pii import RegexPIIDetector

_SAMPLES = [
    "Email alice@example.com or call (555) 123-4567 today.",
    "SSN 123-45-6789, card 4111 1111 1111 1111, host 10.0.0.1.",
    "There is no personal data in this sentence at all.",
    "IBAN GB82WEST12345698765432 and record MRN: AB12345 noted.",
    "Passport B2345678 with phone +1-555-987-6543 on file.",
]


def _spans(detector, text):
    return sorted((m.category, m.start, m.end) for m in detector.analyse(text).matches)


def test_rust_and_python_pii_detection_are_identical():
    rust = RegexPIIDetector(prefer_rust=True)
    if rust.backend != "rust":
        pytest.skip("backfire_kernel not built — Rust PII path unavailable")
    python = RegexPIIDetector(prefer_rust=False)
    for text in _SAMPLES:
        assert _spans(rust, text) == _spans(python, text), text


def test_empty_text_parity():
    rust = RegexPIIDetector(prefer_rust=True)
    if rust.backend != "rust":
        pytest.skip("backfire_kernel not built — Rust PII path unavailable")
    python = RegexPIIDetector(prefer_rust=False)
    assert _spans(rust, "") == _spans(python, "") == []
