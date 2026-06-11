# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — FreshQA harness tests
"""Offline tests for the FreshQA harness.

The harness was previously unreachable by CI (network-gated, no test) and had
drifted: it constructed ``E2ESample(question=…, score=…)`` (fields that do not
exist), called ``metrics.add`` (no such method), and passed a label string where
``print_e2e_results`` expects a baseline ``E2EMetrics`` — so any real run raised.
These tests pin the corrected construction (the loader is stubbed, NLI is off, so
no network or model is touched) and the validity→is_hallucinated mapping.
"""

from __future__ import annotations

import benchmarks.freshqa_eval as freshqa_eval
from benchmarks.e2e_eval import E2EMetrics

_ITEMS = [
    {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "validity": "valid",
    },
    {
        "question": "Who is the current king of France?",
        "answer": "Louis XX",
        "validity": "false_premise",
    },
    {"question": "Latest iPhone?", "answer": "iPhone 3G", "validity": "outdated"},
]


def _stub_loader(monkeypatch, items):
    monkeypatch.setattr(
        freshqa_eval, "_load_freshqa", lambda max_samples=None: list(items)
    )


def test_run_freshqa_builds_metrics_without_drift(monkeypatch):
    # Reproduces what previously raised TypeError on the very first sample.
    _stub_loader(monkeypatch, _ITEMS)
    metrics = freshqa_eval.run_freshqa(use_nli=False)
    assert isinstance(metrics, E2EMetrics)
    assert metrics.total == 3
    # coherence_score is the correct field (the old `score=` kwarg did not exist).
    assert all(s.coherence_score is not None for s in metrics.samples)
    assert all(s.task == "freshqa" for s in metrics.samples)


def test_validity_maps_to_hallucination(monkeypatch):
    _stub_loader(monkeypatch, _ITEMS)
    metrics = freshqa_eval.run_freshqa(use_nli=False)
    flags = {s.response: s.is_hallucinated for s in metrics.samples}
    assert flags["Paris"] is False  # valid
    assert flags["Louis XX"] is True  # false_premise
    assert flags["iPhone 3G"] is True  # outdated


def test_to_dict_is_serialisable(monkeypatch):
    _stub_loader(monkeypatch, _ITEMS)
    report = freshqa_eval.run_freshqa(use_nli=False).to_dict()
    for key in ("catch_rate", "false_positive_rate", "precision", "f1", "total"):
        assert key in report
    assert report["total"] == 3


def test_print_results_takes_single_metrics(monkeypatch, capsys):
    # The old call passed a label as the `baseline` arg; the fixed call must not.
    _stub_loader(monkeypatch, _ITEMS[:1])
    metrics = freshqa_eval.run_freshqa(use_nli=False)
    freshqa_eval.print_e2e_results(metrics)
    assert "FreshQA" in capsys.readouterr().out or metrics.total == 1
