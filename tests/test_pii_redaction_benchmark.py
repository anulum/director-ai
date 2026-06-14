# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — PII redaction benchmark logic tests

from __future__ import annotations

import pytest

from benchmarks.pii_redaction import _gold_spans, _prf, evaluate
from director_ai.core.safety.moderation.pii import RegexPIIDetector


def test_gold_spans_locates_each_substring():
    spans = _gold_spans(
        "mail a@b.com and ip 10.0.0.1 done",
        [("email", "a@b.com"), ("ipv4", "10.0.0.1")],
    )
    assert ("email", 5, 12) in spans
    assert ("ipv4", 20, 28) in spans


def test_prf_perfect_recall_and_precision():
    m = _prf(tp=4, fp=0, fn=0)
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0


def test_prf_empty_is_neutral():
    # nothing to find and nothing found = perfect by convention
    m = _prf(tp=0, fp=0, fn=0)
    assert m["precision"] == 1.0 and m["recall"] == 1.0


def test_prf_counts_partial():
    m = _prf(tp=2, fp=1, fn=1)
    assert m["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["recall"] == pytest.approx(2 / 3, abs=1e-4)


def test_evaluate_detects_and_scores_against_gold():
    detector = RegexPIIDetector(prefer_rust=False)
    samples = [
        ("Reach me at carol@example.org soon", [("email", "carol@example.org")]),
        ("Card 4111 1111 1111 1111 stored", [("credit_card", "4111 1111 1111 1111")]),
        ("Nothing sensitive in this line", []),
    ]
    result = evaluate(detector, samples)
    assert result["overall"]["recall"] == 1.0
    assert result["per_category"]["email"]["tp"] == 1
    assert result["per_category"]["credit_card"]["tp"] == 1
