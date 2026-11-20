# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — fine-tuning worker contract tests

"""Contract tests for the local fine-tuning worker module.

The worker's behaviour (success, failure, cleanup) is exercised through
the historical ``finetune_api`` surface in ``test_finetune_api.py``; this
file pins the module boundary introduced by the WCB-7 decomposition.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import director_ai._finetune_worker as worker_module
import director_ai.core.training.finetune as finetune_module
import director_ai.finetune_api as finetune_api_module
from director_ai.core.training.finetune import FinetuneConfig, _load_jsonl
from director_ai.core.training.training_provenance import TrainingProvenance
from director_ai.finetune_jobs import FinetuneJob, _JobStore


def test_facade_reexports_the_worker() -> None:
    """Require the historical API facade to expose the actual worker function."""
    assert (
        finetune_api_module._run_training_worker is worker_module._run_training_worker
    )


def test_worker_failure_before_split_cleans_only_the_upload(tmp_path: Path) -> None:
    """Remove the upload when model resolution fails before splitting.

    The unknown model is refused before train/eval paths are assigned. The
    worker's actual failure path preserves the error and removes the upload.

    Parameters
    ----------
    tmp_path : Path
        Isolated fixture directory containing the upload and model directory.
    """
    data_path = tmp_path / "upload.jsonl"
    data_path.write_text('{"premise": "p", "hypothesis": "h", "label": 0}\n')
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    job = FinetuneJob(
        job_id="pre-split-fail",
        config={"base_model": "no-such-model", "epochs": 1, "batch_size": 4},
    )

    worker_module._run_training_worker(job, data_path, models_dir, _JobStore())

    assert job.state == "failed"
    assert "no-such-model" in job.error
    assert not data_path.exists()


def test_worker_module_owns_only_the_worker() -> None:
    """Keep the decomposed worker's declared export limited to its worker."""
    assert worker_module.__all__ == ["_run_training_worker"]


def test_worker_logs_to_the_finetune_api_logger() -> None:
    """Retain the historical API logger name for worker diagnostics."""
    assert worker_module.logger.name == "DirectorAI.FinetuneAPI"


def test_worker_splits_uploads_by_source_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep every source cluster and repeated row on one side of the split.

    Parameters
    ----------
    tmp_path : Path
        Isolated fixture directory containing the upload and model directory.
    monkeypatch : pytest.MonkeyPatch
        Replaces the training call with a recorder of the split files.
    """
    rows = [
        {
            "premise": f"Passage {i} of document {i % 6}.",
            "hypothesis": f"Claim {i}.",
            "label": i % 2,
            "source_cluster": f"doc-{i % 6}",
        }
        for i in range(30)
    ]
    rows.append({"premise": "Repeated passage.", "hypothesis": "Claim.", "label": 1})
    rows.append({"premise": "REPEATED  passage.", "hypothesis": "claim.", "label": 1})
    data_path = tmp_path / "upload.jsonl"
    data_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    seen: dict[str, list[dict[str, str | int]]] = {}

    def record_split(
        train_path: str, *, eval_path: str, config: FinetuneConfig
    ) -> SimpleNamespace:
        seen["train"] = _load_jsonl(train_path)
        seen["eval"] = _load_jsonl(eval_path)
        return SimpleNamespace(
            output_dir=config.output_dir,
            eval_metrics={},
            regression_report={},
            best_balanced_accuracy=0.0,
        )

    monkeypatch.setattr(finetune_module, "finetune_nli", record_split)
    job = FinetuneJob(
        job_id="cluster-split",
        config={"base_model": "factcg-deberta-v3-large", "epochs": 1, "batch_size": 4},
    )

    worker_module._run_training_worker(job, data_path, models_dir, _JobStore())

    assert job.state == "completed", job.error
    assert seen["train"] and seen["eval"]
    assert len(seen["train"]) + len(seen["eval"]) == len(rows)
    assert job.metrics["split_train_rows"] == len(seen["train"])
    assert job.metrics["split_eval_rows"] == len(seen["eval"])
    assert (
        TrainingProvenance.from_rows(seen["train"]).overlapping_rows(seen["eval"]) == 0
    )


def test_worker_fails_when_the_upload_has_one_source_cluster(tmp_path: Path) -> None:
    """Fail the job when no independent evaluation rows can be held out.

    Parameters
    ----------
    tmp_path : Path
        Isolated fixture directory containing the upload and model directory.
    """
    rows = [
        {
            "premise": f"P{i}",
            "hypothesis": f"H{i}",
            "label": i % 2,
            "source_cluster": "only",
        }
        for i in range(8)
    ]
    data_path = tmp_path / "upload.jsonl"
    data_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    job = FinetuneJob(
        job_id="one-cluster",
        config={"base_model": "factcg-deberta-v3-large", "epochs": 1, "batch_size": 4},
    )

    worker_module._run_training_worker(job, data_path, models_dir, _JobStore())

    assert job.state == "failed"
    assert "independent groups" in job.error
    assert not data_path.exists()
