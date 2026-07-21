# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — FEVER benchmark tests

"""FEVER dev benchmark: artefact-shape + label-space-guard unit tests plus
slow, model-backed runs.

The two model-backed tests moved out of ``benchmarks/fever_eval.py`` so the
benchmark module no longer imports pytest at module scope (KIMI3/#66); they are
guarded on the presence of a 3-class NLI model so offline CI skips them. FEVER
dev is a 3-class benchmark, so the runs use the held-out hallucination model,
never the 2-class FactCG production default.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import benchmarks.fever_eval as fe
from benchmarks._common import NLIMetrics
from benchmarks.fever_eval import build_fever_artefact, run_fever_benchmark
from director_ai.core.model_revisions import DEFAULT_NLI_MODEL

_FEVER_MODEL = "training/output/deberta-v3-large-hallucination"
_model_ok = Path(_FEVER_MODEL).is_dir()
# NLIMetrics.to_dict() computes P/R/F1 via a lazy sklearn import, and sklearn is
# a benchmark-only dependency absent from the core CI test extras. Tests that
# build the artefact (which calls to_dict) skip there; the label-space guard
# test needs neither sklearn nor datasets and always runs.
_has_sklearn = importlib.util.find_spec("sklearn") is not None
_needs_sklearn = pytest.mark.skipif(
    not _has_sklearn, reason="sklearn (benchmark dep) not in core CI extras"
)


class _FakePredictor:
    """Stand-in for NLIPredictor that reports a chosen label count."""

    def __init__(self, model_name: str | None = None, *, num_labels: int, **_: object):
        self.model_name = model_name or "fake-model"
        config = type("Config", (), {"num_labels": num_labels})()
        self.model = type("Model", (), {"config": config})()


@_needs_sklearn
def test_build_fever_artefact_carries_rows_model_and_provenance():
    metrics = NLIMetrics()
    metrics.y_true = [0, 1, 2]
    metrics.y_pred = [0, 1, 1]
    artefact = build_fever_artefact(metrics, git_sha="a" * 40, model_name="my/model")

    assert artefact["benchmark"] == "FEVER_dev"
    assert artefact["model"] == "my/model"
    assert len(artefact["rows"]) == 3
    assert artefact["rows"][0] == {"index": 0, "label": 0, "predicted": 0}
    assert artefact["rows"][2] == {"index": 2, "label": 2, "predicted": 1}
    assert artefact["provenance"]["git_sha"] == "a" * 40


def test_run_fever_benchmark_rejects_non_3class_model(monkeypatch):
    """A 2-class model cannot produce a valid FEVER verdict — fail loudly."""
    monkeypatch.setattr(
        fe,
        "NLIPredictor",
        lambda model_name=None, **kw: _FakePredictor(model_name, num_labels=2),
    )
    with pytest.raises(ValueError, match="3-class"):
        run_fever_benchmark(max_samples=1, model_name="some/2class-model")


@_needs_sklearn
def test_main_writes_artefact_with_model_and_provenance(tmp_path, monkeypatch):
    metrics = NLIMetrics()
    metrics.y_true = [0, 2]
    metrics.y_pred = [0, 2]
    monkeypatch.setattr(
        fe, "run_fever_benchmark", lambda max_samples=None, model_name=None: metrics
    )
    out = tmp_path / "fever.json"
    argv = ["--max-samples", "2", "--out", str(out), "--git-sha", "b" * 40]
    argv += ["--model", "held-out/3class"]
    assert fe.main(argv) == 0

    payload = json.loads(out.read_text())
    assert payload["benchmark"] == "FEVER_dev"
    assert payload["model"] == "held-out/3class"
    assert len(payload["rows"]) == 2
    assert payload["provenance"]["git_sha"] == "b" * 40


@_needs_sklearn
def test_main_default_records_default_model(tmp_path, monkeypatch):
    metrics = NLIMetrics()
    metrics.y_true = [1]
    metrics.y_pred = [1]
    monkeypatch.setattr(
        fe, "run_fever_benchmark", lambda max_samples=None, model_name=None: metrics
    )
    monkeypatch.delenv("DIRECTOR_NLI_MODEL", raising=False)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        fe,
        "save_results",
        lambda payload, filename: captured.update(payload=payload, filename=filename),
    )
    assert fe.main([]) == 0
    assert captured["filename"] == "fever_results.json"
    assert captured["payload"]["model"] == DEFAULT_NLI_MODEL  # type: ignore[index]


@pytest.mark.slow
@pytest.mark.skipif(not _model_ok, reason="3-class FEVER NLI model not available")
def test_fever_dev_sample():
    metrics = run_fever_benchmark(max_samples=200, model_name=_FEVER_MODEL)
    assert metrics.total > 0
    assert metrics.accuracy > 0.60


@pytest.mark.slow
@pytest.mark.skipif(not _model_ok, reason="3-class FEVER NLI model not available")
def test_fever_dev_full():
    metrics = run_fever_benchmark(model_name=_FEVER_MODEL)
    assert metrics.total > 0
    assert metrics.accuracy > 0.60
