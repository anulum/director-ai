# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Train Tracking CLI Tests

"""Multi-angle tests for the `train runs` and `train registry` CLI commands."""

from __future__ import annotations

import json

import pytest

from director_ai.cli import main
from director_ai.core.training.experiment_tracker import ExperimentTracker
from director_ai.core.training.jobs import LocalTrainingBackend, TrainingJobSpec
from director_ai.core.training.trained_model_registry import TrainedModelRegistry


def _local_spec(tmp_path) -> TrainingJobSpec:
    dataset = tmp_path / "train.jsonl"
    if not dataset.exists():
        dataset.write_text(
            '{"premise": "a", "hypothesis": "b", "label": 1}\n',
            encoding="utf-8",
        )
    return TrainingJobSpec(
        display_name="tracked-training",
        dataset_uri=str(dataset),
        output_uri=str(tmp_path / "out"),
    )


def _tracked_run(tmp_path, runs_dir):
    spec = _local_spec(tmp_path)
    submission = LocalTrainingBackend().submit(spec, dry_run=True)
    return ExperimentTracker(runs_dir).record_submission(
        submission,
        spec,
        fingerprint=spec.dataset_fingerprint(),
    )


def _json_output(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


class TestTrainRunsCommand:
    def test_runs_requires_dir(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(["train", "runs"])
        assert excinfo.value.code == 1
        assert "--dir is required" in capsys.readouterr().out

    def test_runs_rejects_unknown_option(self, capsys):
        with pytest.raises(SystemExit):
            main(["train", "runs", "--bogus"])
        assert "Unknown train runs option" in capsys.readouterr().out

    def test_runs_lists_empty_directory(self, tmp_path, capsys):
        main(["train", "runs", "--dir", str(tmp_path / "runs")])

        assert _json_output(capsys) == {"runs": []}

    def test_runs_lists_tracked_runs(self, tmp_path, capsys):
        run = _tracked_run(tmp_path, tmp_path / "runs")

        main(["train", "runs", "--dir", str(tmp_path / "runs")])

        payload = _json_output(capsys)
        assert [entry["run_id"] for entry in payload["runs"]] == [run.run_id]
        assert payload["runs"][0]["dataset_fingerprint"]["hash_source"] == "content"

    def test_runs_filters_by_backend_and_state(self, tmp_path, capsys):
        _tracked_run(tmp_path, tmp_path / "runs")

        main(
            [
                "train",
                "runs",
                "--dir",
                str(tmp_path / "runs"),
                "--backend",
                "vertex",
                "--state",
                "dry_run",
            ]
        )

        assert _json_output(capsys) == {"runs": []}

    def test_runs_ranks_by_metric(self, tmp_path, capsys):
        runs_dir = tmp_path / "runs"
        first = _tracked_run(tmp_path, runs_dir)
        second = _tracked_run(tmp_path, runs_dir)
        tracker = ExperimentTracker(runs_dir)
        tracker.update_run(first.run_id, metrics={"balanced_accuracy": 0.70})
        tracker.update_run(second.run_id, metrics={"balanced_accuracy": 0.79})

        main(
            [
                "train",
                "runs",
                "--dir",
                str(runs_dir),
                "--metric",
                "balanced_accuracy",
            ]
        )

        payload = _json_output(capsys)
        assert payload["best"]["run_id"] == second.run_id
        assert [entry["run_id"] for entry in payload["ranking"]] == [
            second.run_id,
            first.run_id,
        ]

    def test_runs_metric_without_scores_reports_no_best(self, tmp_path, capsys):
        _tracked_run(tmp_path, tmp_path / "runs")

        main(
            [
                "train",
                "runs",
                "--dir",
                str(tmp_path / "runs"),
                "--metric",
                "balanced_accuracy",
            ]
        )

        payload = _json_output(capsys)
        assert payload["best"] is None
        assert payload["ranking"] == []


class TestSubmitTracksRuns:
    def test_submit_with_experiment_dir_records_run(self, tmp_path, capsys):
        dataset = tmp_path / "train.jsonl"
        dataset.write_text('{"label": 1}\n', encoding="utf-8")

        main(
            [
                "train",
                "submit",
                "--backend",
                "local",
                "--dataset-uri",
                str(dataset),
                "--output-uri",
                str(tmp_path / "out"),
                "--experiment-dir",
                str(tmp_path / "runs"),
            ]
        )

        output = capsys.readouterr().out
        assert "Tracked run:" in output
        runs = ExperimentTracker(tmp_path / "runs").list_runs()
        assert len(runs) == 1
        assert runs[0].dataset_fingerprint["hash_source"] == "content"

    def test_submit_without_experiment_dir_tracks_nothing(self, tmp_path, capsys):
        dataset = tmp_path / "train.jsonl"
        dataset.write_text('{"label": 1}\n', encoding="utf-8")

        main(
            [
                "train",
                "submit",
                "--backend",
                "local",
                "--dataset-uri",
                str(dataset),
                "--output-uri",
                str(tmp_path / "out"),
            ]
        )

        assert "Tracked run:" not in capsys.readouterr().out

    def test_train_help_mentions_tracking_subcommands(self, capsys):
        main(["train", "help"])

        output = capsys.readouterr().out
        assert "runs" in output
        assert "registry" in output
        assert "--experiment-dir" in output


class TestTrainRegistryCommand:
    def test_registry_requires_dir(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(["train", "registry"])
        assert excinfo.value.code == 1
        assert "--dir is required" in capsys.readouterr().out

    def test_registry_rejects_unknown_option(self, capsys):
        with pytest.raises(SystemExit):
            main(["train", "registry", "--bogus"])
        assert "Unknown train registry option" in capsys.readouterr().out

    def test_registry_lists_empty_store(self, tmp_path, capsys):
        main(["train", "registry", "--dir", str(tmp_path / "registry")])

        assert _json_output(capsys) == {"models": {}}

    def test_register_from_tracked_run_carries_lineage(self, tmp_path, capsys):
        run = _tracked_run(tmp_path, tmp_path / "runs")

        main(
            [
                "train",
                "registry",
                "--dir",
                str(tmp_path / "registry"),
                "--register",
                "--name",
                "domain-nli",
                "--artifact",
                "file:///artifacts/domain-nli",
                "--from-run",
                run.run_id,
                "--runs-dir",
                str(tmp_path / "runs"),
                "--base-revision",
                "0430e3509dbd28d2dff7a117c0eae25359ff3e80",
            ]
        )

        record = _json_output(capsys)["registered"]
        assert record["version"] == 1
        assert record["run_id"] == run.run_id
        assert record["config_hash"] == run.config_hash
        assert record["dataset_digest"] == run.dataset_fingerprint["digest"]
        assert record["dataset_hash_source"] == "content"
        assert record["base_model_id"] == "yaxili96/FactCG-DeBERTa-v3-Large"
        assert record["base_model_revision"] == (
            "0430e3509dbd28d2dff7a117c0eae25359ff3e80"
        )

    def test_register_falls_back_to_base_model_field(self, tmp_path, capsys):
        run = _tracked_run(tmp_path, tmp_path / "runs")
        run_file = tmp_path / "runs" / f"{run.run_id}.json"
        payload = json.loads(run_file.read_text(encoding="utf-8"))
        payload["spec"].pop("resolved_base_model")
        run_file.write_text(json.dumps(payload), encoding="utf-8")

        main(
            [
                "train",
                "registry",
                "--dir",
                str(tmp_path / "registry"),
                "--register",
                "--name",
                "domain-nli",
                "--artifact",
                "file:///artifacts/domain-nli",
                "--from-run",
                run.run_id,
                "--runs-dir",
                str(tmp_path / "runs"),
            ]
        )

        record = _json_output(capsys)["registered"]
        assert record["base_model_id"] == "factcg-deberta-v3-large"
        assert record["base_model_revision"] == ""

    @pytest.mark.parametrize(
        ("missing", "message"),
        [
            ("--runs-dir", "--runs-dir is required"),
            ("--from-run", "--from-run is required"),
            ("--name", "--name is required"),
            ("--artifact", "--artifact is required"),
        ],
    )
    def test_register_requires_lineage_options(
        self, tmp_path, capsys, missing, message
    ):
        run = _tracked_run(tmp_path, tmp_path / "runs")
        args = {
            "--name": "domain-nli",
            "--artifact": "file:///artifacts/domain-nli",
            "--from-run": run.run_id,
            "--runs-dir": str(tmp_path / "runs"),
        }
        args.pop(missing)
        argv = ["train", "registry", "--dir", str(tmp_path / "registry"), "--register"]
        for option, value in args.items():
            argv.extend([option, value])

        with pytest.raises(SystemExit) as excinfo:
            main(argv)
        assert excinfo.value.code == 1
        assert message in capsys.readouterr().out

    def test_register_unknown_run_reports_error(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(
                [
                    "train",
                    "registry",
                    "--dir",
                    str(tmp_path / "registry"),
                    "--register",
                    "--name",
                    "domain-nli",
                    "--artifact",
                    "file:///a",
                    "--from-run",
                    "absent",
                    "--runs-dir",
                    str(tmp_path / "runs"),
                ]
            )
        assert excinfo.value.code == 1
        assert "unknown experiment run" in capsys.readouterr().out


class TestRegistryLifecycleViaCli:
    def _registered(self, tmp_path, capsys) -> None:
        run = _tracked_run(tmp_path, tmp_path / "runs")
        tracker = ExperimentTracker(tmp_path / "runs")
        tracker.update_run(run.run_id, metrics={"balanced_accuracy": 0.79})
        main(
            [
                "train",
                "registry",
                "--dir",
                str(tmp_path / "registry"),
                "--register",
                "--name",
                "domain-nli",
                "--artifact",
                "file:///artifacts/domain-nli",
                "--from-run",
                run.run_id,
                "--runs-dir",
                str(tmp_path / "runs"),
            ]
        )
        capsys.readouterr()

    def test_show_model_versions_and_production(self, tmp_path, capsys):
        self._registered(tmp_path, capsys)

        main(
            [
                "train",
                "registry",
                "--dir",
                str(tmp_path / "registry"),
                "--model",
                "domain-nli",
            ]
        )

        payload = _json_output(capsys)
        assert payload["name"] == "domain-nli"
        assert [entry["version"] for entry in payload["versions"]] == [1]
        assert payload["production"] is None

    def test_show_single_version(self, tmp_path, capsys):
        self._registered(tmp_path, capsys)

        main(
            [
                "train",
                "registry",
                "--dir",
                str(tmp_path / "registry"),
                "--model",
                "domain-nli",
                "--version",
                "1",
            ]
        )

        assert _json_output(capsys)["model"]["version"] == 1

    def test_show_unknown_version_reports_error(self, tmp_path, capsys):
        self._registered(tmp_path, capsys)

        with pytest.raises(SystemExit) as excinfo:
            main(
                [
                    "train",
                    "registry",
                    "--dir",
                    str(tmp_path / "registry"),
                    "--model",
                    "domain-nli",
                    "--version",
                    "9",
                ]
            )
        assert excinfo.value.code == 1
        assert "unknown trained model" in capsys.readouterr().out

    def test_promote_requires_evidence_option(self, tmp_path, capsys):
        self._registered(tmp_path, capsys)

        with pytest.raises(SystemExit) as excinfo:
            main(
                [
                    "train",
                    "registry",
                    "--dir",
                    str(tmp_path / "registry"),
                    "--promote",
                    "--model",
                    "domain-nli",
                    "--version",
                    "1",
                ]
            )
        assert excinfo.value.code == 1
        assert "--promote requires --evidence" in capsys.readouterr().out

    def test_promote_requires_model_option(self, tmp_path, capsys):
        evidence = tmp_path / "evidence.json"
        evidence.write_text(
            json.dumps({"metric": "acc", "candidate": 1, "baseline": 0}),
            encoding="utf-8",
        )

        with pytest.raises(SystemExit) as excinfo:
            main(
                [
                    "train",
                    "registry",
                    "--dir",
                    str(tmp_path / "registry"),
                    "--promote",
                    "--evidence",
                    str(evidence),
                ]
            )
        assert excinfo.value.code == 1
        assert "--model is required" in capsys.readouterr().out

    def test_promote_with_valid_evidence_reaches_production(self, tmp_path, capsys):
        self._registered(tmp_path, capsys)
        evidence = tmp_path / "evidence.json"
        evidence.write_text(
            json.dumps(
                {
                    "metric": "balanced_accuracy",
                    "candidate": 0.79,
                    "baseline": 0.758,
                }
            ),
            encoding="utf-8",
        )

        main(
            [
                "train",
                "registry",
                "--dir",
                str(tmp_path / "registry"),
                "--promote",
                "--model",
                "domain-nli",
                "--version",
                "1",
                "--evidence",
                str(evidence),
            ]
        )

        payload = _json_output(capsys)
        assert payload["promoted"]["stage"] == "production"

        main(
            [
                "train",
                "registry",
                "--dir",
                str(tmp_path / "registry"),
                "--model",
                "domain-nli",
            ]
        )
        assert _json_output(capsys)["production"]["version"] == 1

    def test_promote_regression_is_refused(self, tmp_path, capsys):
        self._registered(tmp_path, capsys)
        evidence = tmp_path / "evidence.json"
        evidence.write_text(
            json.dumps(
                {
                    "metric": "balanced_accuracy",
                    "candidate": 0.70,
                    "baseline": 0.758,
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(SystemExit) as excinfo:
            main(
                [
                    "train",
                    "registry",
                    "--dir",
                    str(tmp_path / "registry"),
                    "--promote",
                    "--model",
                    "domain-nli",
                    "--version",
                    "1",
                    "--evidence",
                    str(evidence),
                ]
            )
        assert excinfo.value.code == 1
        assert "anti-regression gate" in capsys.readouterr().out

    def test_promote_with_unreadable_evidence_reports_error(self, tmp_path, capsys):
        self._registered(tmp_path, capsys)
        evidence = tmp_path / "evidence.json"
        evidence.write_text("{not json", encoding="utf-8")

        with pytest.raises(SystemExit) as excinfo:
            main(
                [
                    "train",
                    "registry",
                    "--dir",
                    str(tmp_path / "registry"),
                    "--promote",
                    "--model",
                    "domain-nli",
                    "--version",
                    "1",
                    "--evidence",
                    str(evidence),
                ]
            )
        assert excinfo.value.code == 1
        assert "cannot read benchmark evidence" in capsys.readouterr().out

    def test_promote_with_non_object_evidence_reports_error(self, tmp_path, capsys):
        self._registered(tmp_path, capsys)
        evidence = tmp_path / "evidence.json"
        evidence.write_text("[1, 2]", encoding="utf-8")

        with pytest.raises(SystemExit) as excinfo:
            main(
                [
                    "train",
                    "registry",
                    "--dir",
                    str(tmp_path / "registry"),
                    "--promote",
                    "--model",
                    "domain-nli",
                    "--version",
                    "1",
                    "--evidence",
                    str(evidence),
                ]
            )
        assert excinfo.value.code == 1
        assert "must be a JSON object" in capsys.readouterr().out

    def test_retire_moves_version_to_retired(self, tmp_path, capsys):
        self._registered(tmp_path, capsys)

        main(
            [
                "train",
                "registry",
                "--dir",
                str(tmp_path / "registry"),
                "--retire",
                "--model",
                "domain-nli",
                "--version",
                "1",
            ]
        )

        assert _json_output(capsys)["retired"]["stage"] == "retired"
        assert (
            TrainedModelRegistry(tmp_path / "registry").get("domain-nli", 1).stage
            == "retired"
        )

    def test_retire_requires_version_option(self, tmp_path, capsys):
        self._registered(tmp_path, capsys)

        with pytest.raises(SystemExit) as excinfo:
            main(
                [
                    "train",
                    "registry",
                    "--dir",
                    str(tmp_path / "registry"),
                    "--retire",
                    "--model",
                    "domain-nli",
                ]
            )
        assert excinfo.value.code == 1
        assert "--version is required" in capsys.readouterr().out
