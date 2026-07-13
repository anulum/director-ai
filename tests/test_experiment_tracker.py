# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Experiment Tracker Tests

"""Multi-angle tests for the file-backed training experiment tracker."""

from __future__ import annotations

import json
import time

import pytest

from director_ai.core.training.experiment_tracker import (
    ExperimentRun,
    ExperimentTracker,
)
from director_ai.core.training.jobs import (
    LocalTrainingBackend,
    TrainingJobSpec,
    TrainingJobSubmission,
)


def _local_spec(tmp_path, **overrides) -> TrainingJobSpec:
    dataset = tmp_path / "train.jsonl"
    if not dataset.exists():
        dataset.write_text(
            '{"premise": "a", "hypothesis": "b", "label": 1}\n',
            encoding="utf-8",
        )
    values = {
        "display_name": "tracked-training",
        "dataset_uri": str(dataset),
        "output_uri": str(tmp_path / "out"),
    }
    values.update(overrides)
    return TrainingJobSpec(**values)


def _submission(job_id: str = "local-abc123", **overrides) -> TrainingJobSubmission:
    values = {
        "backend": "local",
        "job_id": job_id,
        "state": "dry_run",
        "dry_run": True,
        "request": {},
        "submitted_at": time.time(),
    }
    values.update(overrides)
    return TrainingJobSubmission(**values)


def _record(tracker, tmp_path, job_id="local-abc123", **kwargs) -> ExperimentRun:
    spec = _local_spec(tmp_path)
    return tracker.record_submission(
        _submission(job_id),
        spec,
        fingerprint=spec.dataset_fingerprint(),
        **kwargs,
    )


class TestRecordSubmission:
    def test_run_carries_full_lineage(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "runs")
        spec = _local_spec(tmp_path)
        fingerprint = spec.dataset_fingerprint()

        run = tracker.record_submission(
            _submission(),
            spec,
            fingerprint=fingerprint,
            tags={"sweep": "s1"},
            notes="first",
        )

        assert run.backend == "local"
        assert run.job_id == "local-abc123"
        assert run.state == "dry_run"
        assert run.config_hash == spec.config_hash
        assert run.dataset_fingerprint == fingerprint.to_dict()
        assert run.dataset_fingerprint["hash_source"] == "content"
        assert run.spec == spec.to_redacted_dict()
        assert run.tags == {"sweep": "s1"}
        assert run.notes == "first"
        assert run.metrics == {}
        assert run.artifact_uri == ""
        assert run.created_at == run.updated_at > 0.0

    def test_run_is_persisted_as_pretty_json(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "runs")
        run = _record(tracker, tmp_path)

        stored = json.loads(
            (tmp_path / "runs" / f"{run.run_id}.json").read_text(encoding="utf-8"),
        )
        assert stored == run.to_dict()

    def test_resubmission_allocates_distinct_run_ids(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "runs")
        first = _record(tracker, tmp_path)
        second = _record(tracker, tmp_path)
        third = _record(tracker, tmp_path)

        assert first.run_id == "local-abc123"
        assert second.run_id == "local-abc123-r2"
        assert third.run_id == "local-abc123-r3"

    def test_job_id_with_path_separators_is_sanitised(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "runs")
        run = _record(
            tracker,
            tmp_path,
            job_id="projects/p/locations/us/customJobs/123",
        )

        assert "/" not in run.run_id
        assert tracker.get(run.run_id).job_id == (
            "projects/p/locations/us/customJobs/123"
        )

    def test_empty_job_id_falls_back_to_run(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "runs")
        run = _record(tracker, tmp_path, job_id="///")

        assert run.run_id == "run"

    def test_local_backend_submission_round_trip(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "runs")
        spec = _local_spec(tmp_path)
        submission = LocalTrainingBackend().submit(spec, dry_run=True)

        run = tracker.record_submission(
            submission,
            spec,
            fingerprint=spec.dataset_fingerprint(),
        )

        assert run.job_id == submission.job_id
        assert run.dataset_fingerprint == (submission.request["dataset_fingerprint"])


class TestUpdateRun:
    def test_update_merges_metrics_and_state(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "runs")
        run = _record(tracker, tmp_path)

        updated = tracker.update_run(
            run.run_id,
            state="completed",
            metrics={"balanced_accuracy": 0.79},
            artifact_uri="file:///artifacts/model",
            notes="harvested",
        )

        assert updated.state == "completed"
        assert updated.metrics == {"balanced_accuracy": 0.79}
        assert updated.artifact_uri == "file:///artifacts/model"
        assert updated.notes == "harvested"
        assert updated.updated_at >= run.updated_at
        assert tracker.get(run.run_id) == updated

    def test_update_preserves_unspecified_fields(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "runs")
        run = _record(tracker, tmp_path, notes="original")
        tracker.update_run(run.run_id, metrics={"loss": 0.5})

        updated = tracker.update_run(run.run_id, metrics={"loss": 0.4, "f1": 0.8})

        assert updated.state == "dry_run"
        assert updated.notes == "original"
        assert updated.artifact_uri == ""
        assert updated.metrics == {"loss": 0.4, "f1": 0.8}

    def test_update_unknown_run_raises_key_error(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "runs")
        with pytest.raises(KeyError, match="unknown experiment run"):
            tracker.update_run("absent", state="completed")


class TestQueries:
    def test_get_unknown_run_raises_key_error(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "runs")
        with pytest.raises(KeyError, match="unknown experiment run"):
            tracker.get("absent")

    def test_list_runs_is_ordered_and_filterable(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "runs")
        first = _record(tracker, tmp_path, job_id="job-a", tags={"lane": "x"})
        second = _record(tracker, tmp_path, job_id="job-b", tags={"lane": "y"})
        tracker.update_run(second.run_id, state="completed")

        every = tracker.list_runs()
        assert [run.run_id for run in every] == [first.run_id, second.run_id]
        assert [run.run_id for run in tracker.list_runs(backend="local")] == [
            first.run_id,
            second.run_id,
        ]
        assert tracker.list_runs(backend="vertex") == []
        assert [run.run_id for run in tracker.list_runs(state="completed")] == [
            second.run_id
        ]
        assert [run.run_id for run in tracker.list_runs(tags={"lane": "x"})] == [
            first.run_id
        ]
        assert tracker.list_runs(tags={"lane": "x", "extra": "no"}) == []

    def test_compare_and_best_run_rank_by_metric(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "runs")
        low = _record(tracker, tmp_path, job_id="job-low")
        high = _record(tracker, tmp_path, job_id="job-high")
        untouched = _record(tracker, tmp_path, job_id="job-none")
        tracker.update_run(low.run_id, metrics={"balanced_accuracy": 0.70})
        tracker.update_run(high.run_id, metrics={"balanced_accuracy": 0.79})

        ranking = tracker.compare("balanced_accuracy")
        assert ranking == [(high.run_id, 0.79), (low.run_id, 0.70)]
        assert untouched.run_id not in [run_id for run_id, _ in ranking]

        best = tracker.best_run("balanced_accuracy")
        assert best is not None and best.run_id == high.run_id

    def test_lower_is_better_ranking(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "runs")
        low = _record(tracker, tmp_path, job_id="job-low")
        high = _record(tracker, tmp_path, job_id="job-high")
        tracker.update_run(low.run_id, metrics={"loss": 0.2})
        tracker.update_run(high.run_id, metrics={"loss": 0.9})

        best = tracker.best_run("loss", higher_is_better=False)
        assert best is not None and best.run_id == low.run_id

    def test_best_run_without_metric_is_none(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "runs")
        _record(tracker, tmp_path)

        assert tracker.best_run("balanced_accuracy") is None

    def test_root_property_reports_storage_directory(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "nested" / "runs")

        assert tracker.root == tmp_path / "nested" / "runs"
        assert tracker.root.is_dir()


class TestRunSerialisation:
    def test_round_trip_preserves_every_field(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "runs")
        run = _record(tracker, tmp_path, tags={"sweep": "s1"}, notes="n")
        updated = tracker.update_run(
            run.run_id,
            state="completed",
            metrics={"balanced_accuracy": 0.79},
            artifact_uri="file:///artifacts/model",
        )

        rebuilt = ExperimentRun.from_dict(
            json.loads(json.dumps(updated.to_dict())),
        )

        assert rebuilt == updated

    def test_from_dict_defaults_optional_fields(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "runs")
        run = _record(tracker, tmp_path)
        payload = run.to_dict()
        payload.pop("artifact_uri")
        payload.pop("notes")

        rebuilt = ExperimentRun.from_dict(payload)

        assert rebuilt.artifact_uri == ""
        assert rebuilt.notes == ""
