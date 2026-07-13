# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Trained Model Registry

"""Versioned registry for trained (fine-tuned) model artefacts.

:mod:`.model_registry` answers "which base models may be fine-tuned"; this
module answers "which trained artefacts exist, where did each come from, and
which one is in production". Every registered version carries its lineage —
tracked run id, config hash, content-addressed dataset digest, and base-model
identity — and moves through explicit stages: ``candidate`` at registration,
``production`` only through :meth:`TrainedModelRegistry.promote`, and
``retired`` when superseded or withdrawn.

Promotion enforces the sweep selection policy ("promote only after
anti-regression benchmark"): it requires benchmark evidence naming the metric
with both candidate and baseline values, and refuses any candidate that
regresses below its baseline. Promoting a version retires the previous
production version of the same model, so at most one production version
exists per name.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .dataset_fingerprint import DatasetFingerprint

STAGE_CANDIDATE = "candidate"
STAGE_PRODUCTION = "production"
STAGE_RETIRED = "retired"
_STAGES = (STAGE_CANDIDATE, STAGE_PRODUCTION, STAGE_RETIRED)
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_EVIDENCE_KEYS = ("metric", "candidate", "baseline")


@dataclass(frozen=True)
class TrainedModelRecord:
    """One registered trained-model version with full lineage.

    Attributes
    ----------
    name : str
        Registry model name (lowercase slug).
    version : int
        Monotonically increasing version within the name.
    artifact_uri : str
        Location of the trained artefact.
    run_id : str
        Experiment-tracker run this artefact came from ("" when untracked).
    base_model_id : str
        Base model that was fine-tuned.
    base_model_revision : str
        Immutable base-model revision when known ("" means unpinned).
    config_hash : str
        Training job config hash.
    dataset_digest : str
        Dataset fingerprint digest used for training.
    dataset_hash_source : str
        ``"content"`` or ``"uri-only"`` — what the digest covers.
    metrics : dict
        Evaluation metrics recorded at registration.
    stage : str
        ``candidate``, ``production``, or ``retired``.
    registered_at : float
        Unix timestamp of registration.
    promoted_at : float
        Unix timestamp of promotion; 0.0 while never promoted.
    benchmark_evidence : dict
        Anti-regression evidence supplied at promotion; empty until promoted.
    """

    name: str
    version: int
    artifact_uri: str
    run_id: str
    base_model_id: str
    base_model_revision: str
    config_hash: str
    dataset_digest: str
    dataset_hash_source: str
    metrics: dict[str, float]
    stage: str = STAGE_CANDIDATE
    registered_at: float = 0.0
    promoted_at: float = 0.0
    benchmark_evidence: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Reject stages outside the candidate/production/retired lifecycle."""
        if self.stage not in _STAGES:
            raise ValueError(f"stage must be one of {_STAGES}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of this record."""
        return {
            "name": self.name,
            "version": self.version,
            "artifact_uri": self.artifact_uri,
            "run_id": self.run_id,
            "base_model_id": self.base_model_id,
            "base_model_revision": self.base_model_revision,
            "config_hash": self.config_hash,
            "dataset_digest": self.dataset_digest,
            "dataset_hash_source": self.dataset_hash_source,
            "metrics": dict(self.metrics),
            "stage": self.stage,
            "registered_at": self.registered_at,
            "promoted_at": self.promoted_at,
            "benchmark_evidence": dict(self.benchmark_evidence or {}),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainedModelRecord:
        """Rebuild a record from its serialised dictionary shape."""
        evidence = dict(payload.get("benchmark_evidence") or {})
        return cls(
            name=str(payload["name"]),
            version=int(payload["version"]),
            artifact_uri=str(payload["artifact_uri"]),
            run_id=str(payload.get("run_id", "")),
            base_model_id=str(payload["base_model_id"]),
            base_model_revision=str(payload.get("base_model_revision", "")),
            config_hash=str(payload.get("config_hash", "")),
            dataset_digest=str(payload["dataset_digest"]),
            dataset_hash_source=str(payload["dataset_hash_source"]),
            metrics={
                key: float(value) for key, value in dict(payload["metrics"]).items()
            },
            stage=str(payload["stage"]),
            registered_at=float(payload["registered_at"]),
            promoted_at=float(payload.get("promoted_at", 0.0)),
            benchmark_evidence=evidence or None,
        )


class TrainedModelRegistry:
    """Register, stage, and query trained model artefacts on disk."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        """Return the registry's storage directory."""
        return self._root

    def register(
        self,
        *,
        name: str,
        artifact_uri: str,
        base_model_id: str,
        dataset_fingerprint: DatasetFingerprint,
        metrics: dict[str, float] | None = None,
        run_id: str = "",
        base_model_revision: str = "",
        config_hash: str = "",
    ) -> TrainedModelRecord:
        """Register a new candidate version of *name* and return it."""
        if not _NAME_RE.fullmatch(name):
            raise ValueError(
                "model name must be a lowercase slug ([a-z0-9._-], max 128)"
            )
        if not artifact_uri:
            raise ValueError("artifact_uri is required")
        if not base_model_id:
            raise ValueError("base_model_id is required")
        versions = self.list_versions(name)
        record = TrainedModelRecord(
            name=name,
            version=versions[-1].version + 1 if versions else 1,
            artifact_uri=artifact_uri,
            run_id=run_id,
            base_model_id=base_model_id,
            base_model_revision=base_model_revision,
            config_hash=config_hash,
            dataset_digest=dataset_fingerprint.digest,
            dataset_hash_source=dataset_fingerprint.hash_source,
            metrics={key: float(value) for key, value in dict(metrics or {}).items()},
            registered_at=time.time(),
        )
        self._write(record)
        return record

    def get(self, name: str, version: int) -> TrainedModelRecord:
        """Return one registered version.

        Raises
        ------
        KeyError
            When the name or version does not exist.
        """
        path = self._record_path(name, version)
        if not path.exists():
            raise KeyError(f"unknown trained model: {name} v{version}")
        return TrainedModelRecord.from_dict(
            json.loads(path.read_text(encoding="utf-8")),
        )

    def list_models(self) -> list[str]:
        """Return all registered model names, sorted."""
        return sorted(entry.name for entry in self._root.iterdir() if entry.is_dir())

    def list_versions(self, name: str) -> list[TrainedModelRecord]:
        """Return every version of *name*, oldest first."""
        model_dir = self._root / name
        if not model_dir.is_dir():
            return []
        records = [
            TrainedModelRecord.from_dict(
                json.loads(path.read_text(encoding="utf-8")),
            )
            for path in model_dir.glob("v*.json")
        ]
        records.sort(key=lambda record: record.version)
        return records

    def production(self, name: str) -> TrainedModelRecord | None:
        """Return the production version of *name*, if one exists."""
        for record in self.list_versions(name):
            if record.stage == STAGE_PRODUCTION:
                return record
        return None

    def promote(
        self,
        name: str,
        version: int,
        *,
        benchmark_evidence: dict[str, Any],
    ) -> TrainedModelRecord:
        """Promote a candidate to production behind the anti-regression gate.

        Parameters
        ----------
        name, version : str, int
            The registered version to promote.
        benchmark_evidence : dict
            Must contain ``metric`` (str), ``candidate`` and ``baseline``
            (numeric) from a same-eval-set benchmark. A candidate below its
            baseline is refused.

        Returns
        -------
        TrainedModelRecord
            The promoted record. Any previous production version of the same
            name is moved to ``retired``.
        """
        record = self.get(name, version)
        if record.stage == STAGE_RETIRED:
            raise ValueError(f"cannot promote retired version: {name} v{version}")
        _validate_benchmark_evidence(benchmark_evidence)
        current = self.production(name)
        if current is not None and current.version != version:
            self._write(replace(current, stage=STAGE_RETIRED))
        promoted = replace(
            record,
            stage=STAGE_PRODUCTION,
            promoted_at=time.time(),
            benchmark_evidence=dict(benchmark_evidence),
        )
        self._write(promoted)
        return promoted

    def retire(self, name: str, version: int) -> TrainedModelRecord:
        """Move one version to the ``retired`` stage."""
        record = self.get(name, version)
        retired = replace(record, stage=STAGE_RETIRED)
        self._write(retired)
        return retired

    def _record_path(self, name: str, version: int) -> Path:
        return self._root / name / f"v{version}.json"

    def _write(self, record: TrainedModelRecord) -> None:
        path = self._record_path(record.name, record.version)
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_suffix(".tmp")
        staging.write_text(
            json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(path)


def _validate_benchmark_evidence(evidence: dict[str, Any]) -> None:
    """Reject promotion evidence that is missing or shows a regression."""
    missing = [key for key in _EVIDENCE_KEYS if key not in evidence]
    if missing:
        raise ValueError(
            f"benchmark_evidence is missing required keys: {missing}; "
            "promotion requires a same-eval-set anti-regression benchmark"
        )
    metric = evidence["metric"]
    if not isinstance(metric, str) or not metric:
        raise ValueError("benchmark_evidence metric must be a non-empty string")
    try:
        candidate = float(evidence["candidate"])
        baseline = float(evidence["baseline"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "benchmark_evidence candidate and baseline must be numeric"
        ) from exc
    if candidate < baseline:
        raise ValueError(
            f"anti-regression gate: candidate {metric}={candidate} is below "
            f"baseline {baseline}; promotion refused"
        )
