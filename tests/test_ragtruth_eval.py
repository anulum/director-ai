# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ragtruth eval regression tests

from __future__ import annotations

import json
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


class TestRowLabel:
    """``_row_label`` must read the real ``wandb/RAGTruth-processed`` schema.

    The raw ``hallucination_labels`` field arrives as a JSON *string* (``"[]"``
    for grounded rows), so a plain truthiness test treats every grounded row as
    hallucinated — the bug that made the whole test split look 100% positive.
    """

    def test_grounded_json_string_is_negative(self) -> None:
        # "[]" is a non-empty string but an empty span list -> grounded.
        assert ragtruth_eval._row_label({"hallucination_labels": "[]"}) is False

    def test_hallucinated_json_string_is_positive(self) -> None:
        row = {"hallucination_labels": '[{"start": 0, "end": 4}]'}
        assert ragtruth_eval._row_label(row) is True

    def test_processed_counts_drive_label(self) -> None:
        conflict = {"hallucination_labels_processed": {"evident_conflict": 1}}
        baseless = {"hallucination_labels_processed": {"baseless_info": 1}}
        clean = {"hallucination_labels_processed": {"evident_conflict": 0}}
        assert ragtruth_eval._row_label(conflict) is True
        assert ragtruth_eval._row_label(baseless) is True
        assert ragtruth_eval._row_label(clean) is False

    def test_explicit_bool_label_takes_precedence(self) -> None:
        # A falsey explicit label must win over a populated span list.
        row = {"label": 0, "hallucination_labels": '[{"x": 1}]'}
        assert ragtruth_eval._row_label(row) is False
        assert ragtruth_eval._row_label({"is_hallucinated": 1}) is True

    def test_native_list_span_annotation(self) -> None:
        assert ragtruth_eval._row_label({"hallucination_labels": [{"x": 1}]}) is True
        assert ragtruth_eval._row_label({"hallucination_labels": []}) is False


class TestHasSpanAnnotation:
    def test_malformed_json_falls_back_to_string_emptiness(self) -> None:
        assert ragtruth_eval._has_span_annotation("not-json") is True
        assert ragtruth_eval._has_span_annotation("   ") is False
        assert ragtruth_eval._has_span_annotation("null") is False
        assert ragtruth_eval._has_span_annotation("{}") is False

    def test_parsed_empty_and_nonempty(self) -> None:
        assert ragtruth_eval._has_span_annotation("[]") is False
        assert ragtruth_eval._has_span_annotation('[{"a": 1}]') is True
        assert ragtruth_eval._has_span_annotation(None) is False


def _stub_metrics() -> ragtruth_eval.E2EMetrics:
    samples = [
        ragtruth_eval.E2ESample(
            task="qa",
            context="ctx",
            response="hallucinated answer",
            is_hallucinated=True,
            coherence_score=0.1,
            approved=False,
        ),
        ragtruth_eval.E2ESample(
            task="qa",
            context="ctx",
            response="grounded answer",
            is_hallucinated=False,
            coherence_score=0.9,
            approved=True,
        ),
    ]
    return ragtruth_eval.E2EMetrics(samples=samples)


def test_main_out_writes_explicit_path_with_provenance(monkeypatch, tmp_path) -> None:
    """--out + --git-sha satisfy the GPU runner contract with full provenance."""
    calls: dict[str, object] = {}

    def fake_run_ragtruth(max_samples=None, use_nli=False):
        calls["max_samples"] = max_samples
        calls["use_nli"] = use_nli
        return _stub_metrics()

    monkeypatch.setattr(ragtruth_eval, "run_ragtruth", fake_run_ragtruth)
    out = tmp_path / "artefact.json"
    sha = "f" * 40
    assert ragtruth_eval.main(["--nli", "--out", str(out), "--git-sha", sha]) == 0
    payload = json.loads(out.read_text())
    assert payload["provenance"]["git_sha"] == sha
    assert len(payload["rows"]) == 2
    assert calls == {"max_samples": None, "use_nli": True}


def test_main_default_path_uses_save_results(monkeypatch) -> None:
    """Without --out the legacy results filename and save_results path hold."""
    captured: dict[str, object] = {}

    def fake_save_results(payload, filename):
        captured["payload"] = payload
        captured["filename"] = filename

    monkeypatch.setattr(
        ragtruth_eval,
        "run_ragtruth",
        lambda max_samples=None, use_nli=False: _stub_metrics(),
    )
    monkeypatch.setattr(ragtruth_eval, "save_results", fake_save_results)
    assert ragtruth_eval.main([]) == 0
    assert captured["filename"] == "ragtruth_results.json"
    payload = captured["payload"]
    assert payload["rows"]
    assert payload["provenance"]["git_sha"] not in (None, "", "unknown")


def test_main_decomposed_branch_forwards_arguments(monkeypatch) -> None:
    """--decomposed routes to the decomposed runner with its own filename."""
    calls: dict[str, object] = {}

    def fake_decomposed(max_samples=None, min_coverage=1.0):
        calls["max_samples"] = max_samples
        calls["min_coverage"] = min_coverage
        return _stub_metrics()

    captured: dict[str, object] = {}
    monkeypatch.setattr(ragtruth_eval, "run_ragtruth_decomposed", fake_decomposed)
    monkeypatch.setattr(
        ragtruth_eval,
        "save_results",
        lambda payload, filename: captured.update(filename=filename),
    )
    argv = ["--decomposed", "--max-samples", "7", "--min-coverage", "0.8"]
    assert ragtruth_eval.main(argv) == 0
    assert calls == {"max_samples": 7, "min_coverage": 0.8}
    assert captured["filename"] == "ragtruth_decomposed_results.json"
