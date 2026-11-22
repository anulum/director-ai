# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Local fine-tuning background worker

"""Background worker that runs a local fine-tuning job to completion.

Owns the training thread body only: dataset split, pipeline invocation,
job-state persistence, and upload cleanup. The REST surface that spawns
the thread lives in :mod:`director_ai.finetune_api`.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from director_ai._finetune_operator_benchmark import snapshot_operator_benchmark
from director_ai.finetune_jobs import FinetuneJob, _JobStore

__all__ = ["_run_training_worker"]

logger = logging.getLogger("DirectorAI.FinetuneAPI")


def _run_training_worker(
    job: FinetuneJob,
    data_path: Path,
    models_dir: Path,
    store: _JobStore,
    general_benchmark: Path | None = None,
) -> None:
    """Run one fine-tuning job in its background thread.

    Parameters
    ----------
    job : FinetuneJob
        The job record, updated and saved as the job progresses.
    data_path : Path
        The uploaded training rows, removed when the job ends.
    models_dir : Path
        The directory that receives the job's model.
    store : _JobStore
        The persistent job store.
    general_benchmark : Path or None
        The operator's general benchmark. When the job benchmarks itself it
        uses a private copy, and its regression report names the copy's
        SHA-256; without one the report says ``reject``.
    """
    train_path: Path | None = None
    eval_path: Path | None = None
    general_path: Path | None = None
    try:
        from director_ai.core.training.finetune import FinetuneConfig, finetune_nli
        from director_ai.core.training.model_registry import resolve_finetune_model

        job.state = "training"
        store.save(job)
        cfg = job.config
        model_profile = resolve_finetune_model(
            cfg.get("base_model", "factcg-deberta-v3-large"),
            allow_experimental=cfg.get("allow_experimental_model", False),
        )

        output_dir = str(models_dir / job.job_id)
        config = FinetuneConfig(
            base_model=model_profile.model_id,
            output_dir=output_dir,
            epochs=cfg.get("epochs", 3),
            batch_size=cfg.get("batch_size", 16),
            learning_rate=cfg.get("learning_rate", 2e-5),
            mix_general_data=cfg.get("mix_general_data", False),
            general_data_ratio=cfg.get("general_data_ratio", 0.2),
            early_stopping_patience=cfg.get("early_stopping_patience", 0),
            class_weighted_loss=cfg.get("class_weighted_loss", False),
            auto_benchmark=cfg.get("auto_benchmark", True),
            auto_onnx_export=cfg.get("auto_onnx_export", False),
        )
        general_sha256 = None
        if config.auto_benchmark and general_benchmark is not None:
            general_path = data_path.parent / f"{job.job_id}_general.jsonl"
            general_sha256 = snapshot_operator_benchmark(
                general_benchmark, general_path
            )
            config.general_benchmark_path = str(general_path)

        from director_ai.core.training.finetune import _load_jsonl
        from director_ai.core.training.training_provenance import (
            split_rows_by_cluster,
        )

        rows = _load_jsonl(data_path)
        # Whole source clusters and repeated content stay on one side, so the
        # evaluation rows are independent of the training rows.
        train_rows, eval_rows = split_rows_by_cluster(rows, eval_fraction=0.1, seed=42)
        logger.info(
            "Job %s split: %d training rows, %d evaluation rows",
            job.job_id,
            len(train_rows),
            len(eval_rows),
        )

        train_path = data_path.parent / f"{job.job_id}_train.jsonl"
        eval_path = data_path.parent / f"{job.job_id}_eval.jsonl"
        for p, r in [(train_path, train_rows), (eval_path, eval_rows)]:
            with open(p, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in r)

        job.total_steps = len(train_rows) // config.batch_size * config.epochs
        store.save(job)

        result = finetune_nli(str(train_path), eval_path=str(eval_path), config=config)

        # Set result fields before state so readers see consistent data
        job.model_path = result.output_dir
        job.metrics = {
            **result.eval_metrics,
            "split_train_rows": len(train_rows),
            "split_eval_rows": len(eval_rows),
        }
        job.regression_report = result.regression_report
        if general_sha256 is not None and isinstance(job.regression_report, dict):
            job.regression_report["general_benchmark_sha256"] = general_sha256
        job.completed_at = time.time()
        job.progress = 1.0
        job.state = "completed"
        store.save(job)

        logger.info(
            "Job %s completed: bal_acc=%.1f%%",
            job.job_id,
            result.best_balanced_accuracy * 100,
        )

    except Exception as exc:
        job.error = str(exc)
        job.state = "failed"
        store.save(job)
        logger.error("Job %s failed: %s", job.job_id, exc)
    finally:
        paths_to_clean: tuple[Path | None, ...] = (
            data_path,
            train_path,
            eval_path,
            general_path,
        )
        for _p in paths_to_clean:
            if _p is not None:
                _p.unlink(missing_ok=True)
