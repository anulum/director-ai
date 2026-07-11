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

from director_ai.finetune_jobs import FinetuneJob, _JobStore

__all__ = ["_run_training_worker"]

logger = logging.getLogger("DirectorAI.FinetuneAPI")


def _run_training_worker(
    job: FinetuneJob, data_path: Path, models_dir: Path, store: _JobStore
) -> None:
    """Background thread that runs the fine-tuning pipeline."""
    train_path: Path | None = None
    eval_path: Path | None = None
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

        import random

        from director_ai.core.training.finetune import _load_jsonl

        rows = _load_jsonl(data_path)
        rng = random.Random(42)
        rng.shuffle(rows)
        n_eval = max(1, int(len(rows) * 0.1))
        eval_rows = rows[:n_eval]
        train_rows = rows[n_eval:]

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
        job.metrics = result.eval_metrics
        job.regression_report = result.regression_report
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
        paths_to_clean: tuple[Path | None, ...] = (data_path, train_path, eval_path)
        for _p in paths_to_clean:
            if _p is not None:
                _p.unlink(missing_ok=True)
