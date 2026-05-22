# SPDX-License-Identifier: AGPL-3.0-or-later
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
