# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Training Experiment Tracker

"""File-backed experiment tracking for managed training runs.

Every submitted training job can be recorded as an :class:`ExperimentRun`
carrying its full lineage — redacted job spec, config hash, and the
content-addressed dataset fingerprint — plus mutable state and metrics that
are updated as the job progresses and its results are harvested. Runs are
stored one JSON file per run under a caller-chosen directory, written
atomically, with no external tracking-service dependency; the directory can
be committed, synced, or exported as-is.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .dataset_fingerprint import DatasetFingerprint
from .jobs import TrainingJobSpec, TrainingJobSubmission

_RUN_ID_SANITISE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_RUN_SUFFIX = ".json"


@dataclass(frozen=True)
class ExperimentRun:
    """One tracked training run with its lineage and outcome.

    Attributes
    ----------
    run_id : str
        Tracker-unique identifier derived from the backend job id.
    backend : str
        Managed training backend name (``local``/``portable``/``vertex``).
    job_id : str
        Backend job identifier from the submission.
    state : str
        Last known job state (``dry_run``, ``submitted``, ``completed``, …).
    config_hash : str
        Redacted-spec hash from :class:`TrainingJobSpec`.
    dataset_fingerprint : dict
        Serialised :class:`DatasetFingerprint` for the training dataset.
    spec : dict
        Redacted job spec at submission time.
    metrics : dict
        Numeric evaluation metrics, filled in as results arrive.
    artifact_uri : str
        Trained artefact location once known.
    tags : dict
        Free-form string labels for querying.
    notes : str
        Human-readable annotations.
    created_at, updated_at : float
        Unix timestamps for creation and last mutation.
    """

    run_id: str
    backend: str
    job_id: str
    state: str
    config_hash: str
    dataset_fingerprint: dict[str, Any]
    spec: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    artifact_uri: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    notes: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of this run."""
        return {
            "run_id": self.run_id,
            "backend": self.backend,
            "job_id": self.job_id,
            "state": self.state,
            "config_hash": self.config_hash,
            "dataset_fingerprint": dict(self.dataset_fingerprint),
            "spec": dict(self.spec),
            "metrics": dict(self.metrics),
            "artifact_uri": self.artifact_uri,
            "tags": dict(self.tags),
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentRun:
        """Rebuild a run from its serialised dictionary shape."""
        return cls(
            run_id=str(payload["run_id"]),
            backend=str(payload["backend"]),
            job_id=str(payload["job_id"]),
            state=str(payload["state"]),
            config_hash=str(payload["config_hash"]),
            dataset_fingerprint=dict(payload["dataset_fingerprint"]),
            spec=dict(payload["spec"]),
            metrics={
                key: float(value) for key, value in dict(payload["metrics"]).items()
            },
            artifact_uri=str(payload.get("artifact_uri", "")),
            tags={key: str(value) for key, value in dict(payload["tags"]).items()},
            notes=str(payload.get("notes", "")),
            created_at=float(payload["created_at"]),
            updated_at=float(payload["updated_at"]),
        )


class ExperimentTracker:
    """Record, update, and query training runs in a local directory."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        """Return the tracker's storage directory."""
        return self._root

    def record_submission(
        self,
        submission: TrainingJobSubmission,
        spec: TrainingJobSpec,
        *,
        fingerprint: DatasetFingerprint,
        tags: dict[str, str] | None = None,
        notes: str = "",
    ) -> ExperimentRun:
        """Persist a new run for *submission* and return it.

        The run id is derived from the backend job id; resubmissions of an
        identical job get distinct ``-r<n>`` suffixes so no run is silently
        overwritten.
        """
        now = time.time()
        run = ExperimentRun(
            run_id=self._allocate_run_id(submission.job_id),
            backend=submission.backend,
            job_id=submission.job_id,
            state=submission.state,
            config_hash=spec.config_hash,
            dataset_fingerprint=fingerprint.to_dict(),
            spec=spec.to_redacted_dict(),
            tags=dict(tags or {}),
            notes=notes,
            created_at=now,
            updated_at=now,
        )
        self._write(run)
        return run

    def update_run(
        self,
        run_id: str,
        *,
        state: str | None = None,
        metrics: dict[str, float] | None = None,
        artifact_uri: str | None = None,
        notes: str | None = None,
    ) -> ExperimentRun:
        """Merge new state, metrics, or annotations into an existing run."""
        run = self.get(run_id)
        merged_metrics = dict(run.metrics)
        for key, value in (metrics or {}).items():
            merged_metrics[key] = float(value)
        updated = replace(
            run,
            state=run.state if state is None else state,
            metrics=merged_metrics,
            artifact_uri=run.artifact_uri if artifact_uri is None else artifact_uri,
            notes=run.notes if notes is None else notes,
            updated_at=time.time(),
        )
        self._write(updated)
        return updated

    def get(self, run_id: str) -> ExperimentRun:
        """Return the run stored under *run_id*.

        Raises
        ------
        KeyError
            When no run with that id exists.
        """
        path = self._run_path(run_id)
        if not path.exists():
            raise KeyError(f"unknown experiment run: {run_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ExperimentRun.from_dict(payload)

    def list_runs(
        self,
        *,
        backend: str | None = None,
        state: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> list[ExperimentRun]:
        """Return runs matching every given filter, oldest first."""
        runs = []
        for path in sorted(self._root.glob(f"*{_RUN_SUFFIX}")):
            run = ExperimentRun.from_dict(
                json.loads(path.read_text(encoding="utf-8")),
            )
            if backend is not None and run.backend != backend:
                continue
            if state is not None and run.state != state:
                continue
            if tags and any(run.tags.get(key) != value for key, value in tags.items()):
                continue
            runs.append(run)
        runs.sort(key=lambda run: (run.created_at, run.run_id))
        return runs

    def best_run(
        self,
        metric: str,
        *,
        higher_is_better: bool = True,
    ) -> ExperimentRun | None:
        """Return the run with the best value for *metric*, if any has it."""
        ranked = self.compare(metric, higher_is_better=higher_is_better)
        if not ranked:
            return None
        return self.get(ranked[0][0])

    def compare(
        self,
        metric: str,
        *,
        higher_is_better: bool = True,
    ) -> list[tuple[str, float]]:
        """Return ``(run_id, value)`` pairs for *metric*, best first."""
        scored = [
            (run.run_id, run.metrics[metric])
            for run in self.list_runs()
            if metric in run.metrics
        ]
        scored.sort(key=lambda item: item[1], reverse=higher_is_better)
        return scored

    def _allocate_run_id(self, job_id: str) -> str:
        base = _RUN_ID_SANITISE_RE.sub("-", job_id).strip("-") or "run"
        if not self._run_path(base).exists():
            return base
        counter = 2
        while self._run_path(f"{base}-r{counter}").exists():
            counter += 1
        return f"{base}-r{counter}"

    def _run_path(self, run_id: str) -> Path:
        safe = _RUN_ID_SANITISE_RE.sub("-", run_id)
        return self._root / f"{safe}{_RUN_SUFFIX}"

    def _write(self, run: ExperimentRun) -> None:
        path = self._run_path(run.run_id)
        staging = path.with_suffix(".tmp")
        staging.write_text(
            json.dumps(run.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(path)
