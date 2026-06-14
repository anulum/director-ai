# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — contradiction recall benchmark plumbing tests

from __future__ import annotations

from benchmarks import contradiction_recall as cr


class _FlipDetectingScorer:
    """Stub: high contradiction when the claim was meaning-flipped."""

    threshold = 0.2

    def contradiction_batch(self, pairs):
        out = []
        for _premise, claim in pairs:
            low = claim.lower()
            flipped = " not " in low or "fell" in low or low.startswith("no ")
            out.append(0.9 if flipped else 0.05)
        return out


def test_recall_run_plumbing(monkeypatch):
    supported = [
        {
            "doc": "The merger was approved by the board fully.",
            "claim": "The merger was approved by the board fully.",
            "label": "1",
        },
        {
            "doc": "Revenue rose sharply this year overall.",
            "claim": "Revenue rose sharply this year overall.",
            "label": "1",
        },
    ]
    monkeypatch.setattr(cr, "_load_supported", lambda n: supported)
    monkeypatch.setattr(
        cr.ContradictionScorer,
        "from_pretrained",
        lambda *a, **k: _FlipDetectingScorer(),
    )

    res = cr.run(device=-1, granularity="document")

    assert res["n_supported_seen"] == 2
    assert res["n_injectable"] == 2
    # row1 -> negation + antonym; row2 -> antonym
    assert res["n_injected_variants"] == 3
    assert res["injection_strategy_counts"]["antonym"] == 2
    assert res["injection_strategy_counts"]["negation"] == 1

    thr = res["overall"]["thresholds"]["0.2"]
    # originals are unflipped -> never false-halt
    assert thr["false_halt_rate"] == 0.0
    # flipped variants are caught (negation + one antonym "fell")
    assert thr["recall"] > 0.0
    assert res["recall_by_strategy"]["negation"]["recall@0.2"] == 1.0


def test_recall_run_skips_unflippable_claims(monkeypatch):
    supported = [
        {"doc": "Birds sang outside.", "claim": "Birds sang outside.", "label": "1"},
    ]
    monkeypatch.setattr(cr, "_load_supported", lambda n: supported)
    monkeypatch.setattr(
        cr.ContradictionScorer,
        "from_pretrained",
        lambda *a, **k: _FlipDetectingScorer(),
    )
    res = cr.run(device=-1, granularity="document")
    assert res["n_injectable"] == 0
    assert res["n_injected_variants"] == 0
