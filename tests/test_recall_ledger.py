# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for reading REMANENTIA's recall ledger and cold-starting calibration.

Covers the JSONL merge (query + outcome lines into one RecallQuery, the two
labels independent, latest-outcome-per-field wins), every robustness skip (blank,
malformed, non-object, missing/blank event_id, outcome-before-query), the absent
file and env-override path resolution, score/segment coercion, and the cold-start
replay: correctness-labelled records prime both the adaptive predictor (only when
a retrieval score is present) and the miscoverage monitor segmented by project,
while usage-only and unlabelled records contribute nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from director_ai.core.calibration.adaptive_conformal import AdaptiveConformalPredictor
from director_ai.core.calibration.miscoverage import MiscoverageMonitor
from director_ai.core.calibration.recall_ledger import (
    DEFAULT_LEDGER_PATH,
    LEDGER_PATH_ENV,
    ColdStartSummary,
    RecallQuery,
    cold_start_from_ledger,
    default_ledger_path,
    read_recall_ledger,
)


def _query_line(event_id: str, **over: object) -> str:
    record: dict[str, object] = {
        "kind": "query",
        "event_id": event_id,
        "ts": 1.0,
        "by": "remanentia",
        "query": f"q-{event_id}",
        "top_k": 3,
        "project": "director-ai",
        "returned_ids": ["semantic:doc1", "episodic:trace2"],
        "found": True,
        "score": 0.8,
        "abstained": False,
    }
    record.update(over)
    return json.dumps(record)


def _outcome_line(event_id: str, **over: object) -> str:
    record: dict[str, object] = {"kind": "outcome", "event_id": event_id, "ts": 2.0}
    record.update(over)
    return json.dumps(record)


def _write(path: Path, *lines: str) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- merge semantics --------------------------------------------------------


def test_merges_query_and_outcome(tmp_path: Path) -> None:
    ledger = _write(
        tmp_path / "l.jsonl",
        _query_line("e1"),
        _outcome_line("e1", was_used=True, was_correct=False),
    )
    (record,) = read_recall_ledger(ledger)
    assert record == RecallQuery(
        event_id="e1",
        ts=1.0,
        by="remanentia",
        query="q-e1",
        top_k=3,
        project="director-ai",
        returned_ids=("semantic:doc1", "episodic:trace2"),
        found=True,
        score=0.8,
        abstained=False,
        was_used=True,
        was_correct=False,
    )


def test_labels_are_independent(tmp_path: Path) -> None:
    """was_used and was_correct may arrive on separate outcome lines."""
    ledger = _write(
        tmp_path / "l.jsonl",
        _query_line("e1"),
        _outcome_line("e1", ts=2.0, was_used=True),
        _outcome_line("e1", ts=3.0, was_correct=True),
    )
    (record,) = read_recall_ledger(ledger)
    assert record.was_used is True
    assert record.was_correct is True


def test_latest_outcome_per_field_wins(tmp_path: Path) -> None:
    ledger = _write(
        tmp_path / "l.jsonl",
        _query_line("e1"),
        _outcome_line("e1", ts=5.0, was_correct=True),
        _outcome_line("e1", ts=2.0, was_correct=False),  # older, must not clobber
    )
    (record,) = read_recall_ledger(ledger)
    assert record.was_correct is True


def test_query_without_outcome_has_none_labels(tmp_path: Path) -> None:
    ledger = _write(tmp_path / "l.jsonl", _query_line("e1"))
    (record,) = read_recall_ledger(ledger)
    assert record.was_used is None
    assert record.was_correct is None


def test_null_score_and_missing_project(tmp_path: Path) -> None:
    ledger = _write(
        tmp_path / "l.jsonl",
        _query_line("e1", score=None, project="", returned_ids="bad", abstained=True),
    )
    (record,) = read_recall_ledger(ledger)
    assert record.score is None
    assert record.project == "default"
    assert record.returned_ids == ()
    assert record.abstained is True


@pytest.mark.parametrize("bad_top_k", ["three", None, True])
def test_non_integer_top_k_defaults_to_zero(tmp_path: Path, bad_top_k: object) -> None:
    """A missing, non-integer, or boolean top_k coerces to 0, not a crash."""
    ledger = _write(tmp_path / "l.jsonl", _query_line("e1", top_k=bad_top_k))
    (record,) = read_recall_ledger(ledger)
    assert record.top_k == 0


# --- robustness skips -------------------------------------------------------


def test_skips_blank_malformed_and_orphan_lines(tmp_path: Path) -> None:
    ledger = _write(
        tmp_path / "l.jsonl",
        "",
        "   ",
        "{not json}",
        json.dumps([1, 2, 3]),  # non-object
        json.dumps({"kind": "query"}),  # missing event_id
        json.dumps({"kind": "query", "event_id": ""}),  # blank event_id
        _outcome_line("orphan", was_correct=True),  # outcome before any query
        _query_line("e1"),
        _outcome_line("e1", was_correct=True),
    )
    records = read_recall_ledger(ledger)
    assert [r.event_id for r in records] == ["e1"]
    assert records[0].was_correct is True


def test_non_bool_label_is_ignored(tmp_path: Path) -> None:
    """A non-boolean was_correct value does not set the label."""
    ledger = _write(
        tmp_path / "l.jsonl",
        _query_line("e1"),
        _outcome_line("e1", was_correct="yes"),
    )
    (record,) = read_recall_ledger(ledger)
    assert record.was_correct is None


def test_unknown_kind_line_is_ignored(tmp_path: Path) -> None:
    """A line with a valid event_id but neither query nor outcome kind is skipped."""
    ledger = _write(
        tmp_path / "l.jsonl",
        json.dumps({"kind": "heartbeat", "event_id": "e0"}),
        _query_line("e1"),
        _outcome_line("e1", was_correct=True),
    )
    records = read_recall_ledger(ledger)
    assert [r.event_id for r in records] == ["e1"]


# --- path resolution --------------------------------------------------------


def test_missing_file_returns_empty(tmp_path: Path) -> None:
    assert read_recall_ledger(tmp_path / "absent.jsonl") == []


def test_default_ledger_path_uses_constant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LEDGER_PATH_ENV, raising=False)
    assert default_ledger_path() == DEFAULT_LEDGER_PATH


def test_default_ledger_path_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LEDGER_PATH_ENV, "/tmp/shared/recall.jsonl")
    assert default_ledger_path() == Path("/tmp/shared/recall.jsonl")


def test_blank_env_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LEDGER_PATH_ENV, "   ")
    assert default_ledger_path() == DEFAULT_LEDGER_PATH


def test_read_uses_default_path_when_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger = _write(tmp_path / "d.jsonl", _query_line("e1"))
    monkeypatch.setenv(LEDGER_PATH_ENV, str(ledger))
    records = read_recall_ledger()
    assert [r.event_id for r in records] == ["e1"]


# --- cold start -------------------------------------------------------------


def _primed() -> tuple[AdaptiveConformalPredictor, MiscoverageMonitor]:
    return (
        AdaptiveConformalPredictor(coverage=0.9, gamma=0.1, min_samples=2),
        MiscoverageMonitor(target_alpha=0.1, min_samples=2),
    )


def test_cold_start_primes_predictor_and_monitor(tmp_path: Path) -> None:
    ledger = _write(
        tmp_path / "l.jsonl",
        _query_line("e1", project="alpha", score=0.9),
        _outcome_line("e1", was_correct=True),
        _query_line("e2", project="alpha", score=0.3),
        _outcome_line("e2", was_correct=False),
        _query_line("e3", project="beta", score=0.7),
        _outcome_line("e3", was_correct=True),
    )
    predictor, monitor = _primed()
    summary = cold_start_from_ledger(predictor, monitor, path=ledger)

    assert summary == ColdStartSummary(
        records=3, labelled=3, calibrated=3, segments=("alpha", "beta")
    )
    # Predictor received one residual per labelled+scored record.
    assert predictor.predict(0.5).calibration_size == 3
    # Monitor saw one miss in two alpha observations.
    assert monitor.miscoverage("alpha") == pytest.approx(0.5)
    assert monitor.miscoverage("beta") == pytest.approx(0.0)


def test_cold_start_skips_unlabelled_records(tmp_path: Path) -> None:
    ledger = _write(
        tmp_path / "l.jsonl",
        _query_line("e1", score=0.9),
        _outcome_line("e1", was_used=True),  # usage only, no correctness
        _query_line("e2", score=0.5),  # no outcome at all
        _query_line("e3", score=0.6),
        _outcome_line("e3", was_correct=True),
    )
    predictor, monitor = _primed()
    summary = cold_start_from_ledger(predictor, monitor, path=ledger)
    assert summary.records == 3
    assert summary.labelled == 1
    assert summary.calibrated == 1
    assert predictor.predict(0.5).calibration_size == 1


def test_cold_start_labelled_without_score_not_calibrated(tmp_path: Path) -> None:
    """A correctness label with no retrieval score primes coverage, not residuals."""
    ledger = _write(
        tmp_path / "l.jsonl",
        _query_line("e1", score=None),
        _outcome_line("e1", was_correct=True),
    )
    predictor, monitor = _primed()
    summary = cold_start_from_ledger(predictor, monitor, path=ledger)
    assert summary.labelled == 1
    assert summary.calibrated == 0
    assert predictor.predict(0.5).calibration_size == 0
    assert monitor.miscoverage("director-ai") == pytest.approx(0.0)


def test_cold_start_accepts_in_memory_records() -> None:
    records = [
        RecallQuery(
            event_id="e1",
            ts=1.0,
            by="b",
            query="q",
            top_k=3,
            project="p",
            returned_ids=(),
            found=True,
            score=0.4,
            abstained=False,
            was_used=None,
            was_correct=True,
        )
    ]
    predictor, monitor = _primed()
    summary = cold_start_from_ledger(predictor, monitor, records=records)
    assert summary == ColdStartSummary(
        records=1, labelled=1, calibrated=1, segments=("p",)
    )


def test_cold_start_empty_ledger_is_noop(tmp_path: Path) -> None:
    predictor, monitor = _primed()
    summary = cold_start_from_ledger(predictor, monitor, path=tmp_path / "absent.jsonl")
    assert summary == ColdStartSummary(records=0, labelled=0, calibrated=0, segments=())


def test_public_surface_reexports_ledger_api() -> None:
    from director_ai.core import (
        ColdStartSummary as ExportedSummary,
    )
    from director_ai.core import (
        cold_start_from_ledger as exported_cold_start,
    )
    from director_ai.core import (
        read_recall_ledger as exported_read,
    )

    assert ExportedSummary is ColdStartSummary
    assert exported_cold_start is cold_start_from_ledger
    assert exported_read is read_recall_ledger
