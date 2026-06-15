# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ragtruth eval regression tests

from __future__ import annotations

import types

from benchmarks import ragtruth_eval


def test_load_ragtruth_fallback_order(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def _fake_load_dataset(dataset_id: str, *, split: str):
        calls.append((dataset_id, split))
        if dataset_id == "wandb/RAGTruth-processed":
            raise RuntimeError("primary unavailable")
        return [{"context": "ctx", "query": "q", "output": "r", "label": 0}]

    monkeypatch.setitem(
        __import__("sys").modules,
        "datasets",
        types.SimpleNamespace(load_dataset=_fake_load_dataset),
    )
    items = ragtruth_eval._load_ragtruth(max_samples=1)
    assert len(items) == 1
    assert calls[0] == ("wandb/RAGTruth-processed", "test")
    assert calls[1] == ("flowaicom/RAGTruth_test", "qa")


def test_run_ragtruth_maps_labels_and_appends_samples(monkeypatch) -> None:
    monkeypatch.setattr(
        ragtruth_eval,
        "_load_ragtruth",
        lambda _max: [
            {
                "context": "ctx",
                "query": "q",
                "output": "r",
                "hallucination_labels_processed": {
                    "evident_conflict": 1,
                    "baseless_info": 0,
                },
            }
        ],
    )

    class _Store:
        def ingest(self, _docs):
            return None

    class _Score:
        score = 0.3

    class _Scorer:
        def __init__(self, **_kwargs):
            pass

        def review(self, _prompt: str, _response: str):
            return (False, _Score())

    monkeypatch.setitem(
        __import__("sys").modules,
        "director_ai.core.vector_store",
        types.SimpleNamespace(VectorGroundTruthStore=_Store),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "director_ai.core.scorer",
        types.SimpleNamespace(CoherenceScorer=_Scorer),
    )

    metrics = ragtruth_eval.run_ragtruth(max_samples=1, use_nli=False)
    assert metrics.total == 1
    assert metrics.samples[0].is_hallucinated is True
    assert metrics.samples[0].approved is False


def test_evaluate_decomposed_flags_low_coverage_responses() -> None:
    # Decompose-then-aggregate: a response is flagged when grounded-claim coverage
    # drops below min_coverage. Stub coverage_fn keys off a marker so no model is
    # needed; verify the confusion matrix maps coverage -> approved correctly.
    rows = [
        # hallucinated label, low coverage -> should be caught (not approved)
        {"context": "c", "output": "bad claim", "hallucination_labels": [{"x": 1}]},
        # grounded label, full coverage -> should pass (approved)
        {"context": "c", "output": "good claim", "hallucination_labels": []},
        # hallucinated label but full coverage -> a miss (approved despite label)
        {"context": "c", "output": "good claim", "hallucination_labels": [{"x": 1}]},
    ]

    def coverage_fn(_context: str, response: str) -> float:
        return 0.5 if "bad" in response else 1.0

    metrics = ragtruth_eval.evaluate_decomposed(rows, coverage_fn, min_coverage=1.0)
    assert metrics.total == 3
    # row 0: hallucinated + coverage 0.5 < 1.0 -> not approved -> caught (tp)
    assert metrics.tp == 1
    # row 1: grounded + coverage 1.0 -> approved -> true negative
    assert metrics.tn == 1
    # row 2: hallucinated + coverage 1.0 -> approved -> false negative (missed)
    assert metrics.fn == 1
    assert metrics.fp == 0


def test_evaluate_decomposed_skips_empty_response() -> None:
    rows = [{"context": "c", "output": "", "hallucination_labels": []}]
    metrics = ragtruth_eval.evaluate_decomposed(
        rows, lambda _c, _r: 1.0, min_coverage=1.0
    )
    assert metrics.total == 0


def test_evaluate_decomposed_min_coverage_threshold_tunes_sensitivity() -> None:
    rows = [{"context": "c", "output": "x", "hallucination_labels": [{"x": 1}]}]
    # coverage 0.8: with min_coverage 1.0 it is flagged; with 0.5 it passes.
    strict = ragtruth_eval.evaluate_decomposed(
        rows, lambda _c, _r: 0.8, min_coverage=1.0
    )
    lenient = ragtruth_eval.evaluate_decomposed(
        rows, lambda _c, _r: 0.8, min_coverage=0.5
    )
    assert strict.tp == 1  # flagged
    assert lenient.fn == 1  # passed (missed)
