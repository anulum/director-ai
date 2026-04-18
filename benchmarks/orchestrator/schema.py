# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — orchestrator result schema

"""Dataclasses + JSON schema for the orchestrator's outputs.

Every artefact the orchestrator writes round-trips through
:class:`RunReport.from_json` — a schema mismatch fails the
regression gate rather than corrupting the baseline. The schema
is intentionally strict; adding a new field is a deliberate
change that ripples through :mod:`.regression` and the report
generator.

``ResultStatus`` distinguishes four outcomes — ``passed``
benchmarks contribute to regression deltas, ``failed`` benchmarks
are hard errors, ``skipped`` benchmarks are logged but don't
affect the gate, and ``warned`` benchmarks surface in the report
but don't halt the pipeline. The split matters for Vertex AI
runs where the orchestrator must finish the whole suite even
when a single component errors.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

SCHEMA_VERSION = "1.0.0"
"""Bumped when a backwards-incompatible field is introduced so
old baselines can be detected and migrated explicitly."""

ResultStatus = Literal["passed", "failed", "skipped", "warned"]


@dataclass(frozen=True)
class MetricResult:
    """One numeric measurement from a single benchmark.

    ``name`` is the canonical identifier under the suite entry
    (e.g. ``"balanced_accuracy"``, ``"p99_latency_ms"``). ``value``
    is a float so the regression engine can compute deltas
    uniformly. ``unit`` is free-text for the report generator
    (``"ratio"``, ``"ms"``, ``"samples/s"``). ``higher_is_better``
    drives the direction of the regression check — e.g. a BA
    drop of 2 pp is bad, a latency drop of 20 ms is good.
    """

    name: str
    value: float
    unit: str = ""
    higher_is_better: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("MetricResult.name must be non-empty")
        if not isinstance(self.value, (int, float)):
            raise ValueError(
                f"MetricResult.value must be numeric; got {type(self.value).__name__}"
            )


@dataclass(frozen=True)
class SuiteEntry:
    """One benchmark's output inside a :class:`RunReport`.

    ``kind`` tells the regression engine how to interpret the
    metrics — the four practical categories today are
    ``accuracy``, ``latency``, ``e2e`` (end-to-end guardrail),
    and ``smoke`` (hook / import / invariant). Keeping the
    category explicit avoids the regression gate conflating a
    latency dip with an accuracy regression.
    """

    name: str
    kind: str
    status: ResultStatus
    metrics: tuple[MetricResult, ...]
    wall_clock_seconds: float
    dataset_hash: str = ""
    dataset_size: int = 0
    seed: int = 0
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("SuiteEntry.name must be non-empty")
        if self.kind not in {"accuracy", "latency", "e2e", "smoke"}:
            raise ValueError(
                f"SuiteEntry.kind must be one of "
                f"(accuracy, latency, e2e, smoke); got {self.kind!r}"
            )
        if self.wall_clock_seconds < 0:
            raise ValueError("wall_clock_seconds must be non-negative")

    def metric(self, name: str) -> MetricResult | None:
        """Return the metric named ``name`` or ``None`` if absent."""
        for m in self.metrics:
            if m.name == name:
                return m
        return None


@dataclass(frozen=True)
class EnvironmentRecord:
    """Immutable environment fingerprint captured at run start.

    This is the authoritative answer to "what code and what
    hardware produced this report". Missing or fabricated fields
    make the report unreplicable — the orchestrator refuses to
    write a report with an empty ``git_commit``.
    """

    git_commit: str
    git_dirty: bool
    git_branch: str
    package_version: str
    python_version: str
    platform: str
    cpu_model: str
    cpu_count: int
    ram_gb: float
    gpu_model: str = ""
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    runner: str = "local"

    def __post_init__(self) -> None:
        if not self.git_commit:
            raise ValueError("git_commit must not be empty")
        if self.cpu_count <= 0:
            raise ValueError("cpu_count must be positive")
        if self.ram_gb < 0:
            raise ValueError("ram_gb must be non-negative")
        if self.runner not in {"local", "vertex", "ci", "remote"}:
            raise ValueError(
                f"runner must be one of (local, vertex, ci, remote); "
                f"got {self.runner!r}"
            )


@dataclass(frozen=True)
class RunReport:
    """Top-level report written to disk per orchestrator run.

    JSON format is::

        {
            "schema_version": "1.0.0",
            "run_id": "...",
            "timestamp_utc": "2026-04-18T05:12:03Z",
            "environment": {...},
            "entries": [...],
            "notes": "..."
        }

    Load-then-save is idempotent: the JSON emitted by
    :meth:`to_json` round-trips through :meth:`from_json` without
    change.
    """

    run_id: str
    timestamp_utc: str
    environment: EnvironmentRecord
    entries: tuple[SuiteEntry, ...]
    schema_version: str = SCHEMA_VERSION
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if not self.timestamp_utc.endswith("Z"):
            raise ValueError(
                "timestamp_utc must be ISO-8601 UTC with trailing Z; "
                f"got {self.timestamp_utc!r}"
            )

    @property
    def all_passed(self) -> bool:
        return all(e.status in {"passed", "skipped"} for e in self.entries)

    @property
    def failed_entries(self) -> tuple[SuiteEntry, ...]:
        return tuple(e for e in self.entries if e.status == "failed")

    def entry(self, name: str) -> SuiteEntry | None:
        """Return the suite entry named ``name`` or ``None``."""
        for e in self.entries:
            if e.name == name:
                return e
        return None

    def to_json(self, indent: int | None = 2) -> str:
        """Serialise to canonical JSON. ``indent=None`` produces
        one-line output suitable for Vertex AI log streams."""
        return json.dumps(asdict(self), indent=indent, sort_keys=True)

    def to_file(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json())
        return p

    @classmethod
    def from_json(cls, payload: str | Mapping[str, object]) -> RunReport:
        """Parse a JSON string or dict into a :class:`RunReport`."""
        raw = json.loads(payload) if isinstance(payload, str) else dict(payload)
        validate_report(raw)
        env_raw = raw["environment"]
        if not isinstance(env_raw, Mapping):
            raise ValueError("environment must be an object")
        environment = EnvironmentRecord(**env_raw)
        entries_raw = raw["entries"]
        if not isinstance(entries_raw, Sequence):
            raise ValueError("entries must be a sequence")
        entries = tuple(_parse_entry(e) for e in entries_raw)
        return cls(
            run_id=str(raw["run_id"]),
            timestamp_utc=str(raw["timestamp_utc"]),
            environment=environment,
            entries=entries,
            schema_version=str(raw.get("schema_version", SCHEMA_VERSION)),
            notes=str(raw.get("notes", "")),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> RunReport:
        return cls.from_json(Path(path).read_text())


def _parse_entry(raw: object) -> SuiteEntry:
    if not isinstance(raw, Mapping):
        raise ValueError(f"entry must be a dict, got {type(raw).__name__}")
    metrics_raw = raw.get("metrics", [])
    if not isinstance(metrics_raw, Sequence):
        raise ValueError("entry.metrics must be a sequence")
    metrics = tuple(
        MetricResult(**m) if isinstance(m, Mapping) else _bad_metric(m)
        for m in metrics_raw
    )
    return SuiteEntry(
        name=str(raw["name"]),
        kind=str(raw["kind"]),
        status=_narrow_status(raw.get("status", "passed")),
        metrics=metrics,
        wall_clock_seconds=float(raw["wall_clock_seconds"]),
        dataset_hash=str(raw.get("dataset_hash", "")),
        dataset_size=int(raw.get("dataset_size", 0)),
        seed=int(raw.get("seed", 0)),
        notes=str(raw.get("notes", "")),
    )


def _bad_metric(m: object) -> MetricResult:
    raise ValueError(f"metric must be a dict, got {type(m).__name__}: {m!r}")


def _narrow_status(value: object) -> ResultStatus:
    text = str(value)
    if text not in {"passed", "failed", "skipped", "warned"}:
        raise ValueError(f"status must be passed/failed/skipped/warned; got {text!r}")
    # Runtime membership check above proves the narrowing; ``cast``
    # documents it to the type checker without suppressing anything.
    return cast(ResultStatus, text)


def validate_report(raw: Mapping[str, object]) -> None:
    """Raise :class:`ValueError` when *raw* does not match the
    required top-level shape. Called by :meth:`RunReport.from_json`
    and exposed for callers that want to validate a payload
    before constructing a report (e.g. CI pre-checks)."""
    required = {
        "run_id",
        "timestamp_utc",
        "environment",
        "entries",
    }
    missing = required - set(raw.keys())
    if missing:
        raise ValueError(f"report missing required fields: {sorted(missing)}")
    version = str(raw.get("schema_version", SCHEMA_VERSION))
    major = version.split(".", 1)[0]
    expected_major = SCHEMA_VERSION.split(".", 1)[0]
    if major != expected_major:
        raise ValueError(
            f"schema major version mismatch: got {version!r}, "
            f"expected {SCHEMA_VERSION!r}"
        )
