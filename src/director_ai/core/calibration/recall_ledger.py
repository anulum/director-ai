# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — read REMANENTIA's recall ledger to cold-start calibration

"""Read REMANENTIA's recall ledger and cold-start the conformal predictors.

REMANENTIA writes a JSON-lines ledger with two record kinds: a ``query`` line
per recall (carrying the query, project, retrieval ``score`` and ``found`` flag)
and ``outcome`` lines that attach the two independent labels — ``was_used``
(downstream usage, auto-loop-closure) and ``was_correct`` (Director-AI's
verification verdict, posted via
:mod:`director_ai.core.calibration.recall_correctness_client`). The labels arrive
on separate lines and the latest outcome per field wins, so this module merges
them by ``event_id`` into a :class:`RecallQuery` per recall — mirroring
REMANENTIA's own ``RecallLedger.queries()`` so the two sides agree on the merge.

The point of reading it is **cold start**: a fresh
:class:`~director_ai.core.calibration.adaptive_conformal.AdaptiveConformalPredictor`
and :class:`~director_ai.core.calibration.miscoverage.MiscoverageMonitor` begin
empty and certify nothing until they have observed enough outcomes. Replaying the
ledger's correctness history primes them so the abstention gate is calibrated
from the first live query instead of failing closed for the first few hundred.

The cold start calibrates on ``was_correct`` **only**. ``was_used`` is a usage /
ranking signal — calibrating a coverage guarantee on usage rather than
correctness is the "calibration theatre" the abstention design rules out (see
:mod:`director_ai.core.calibration.recall_correctness`). Records whose
``was_correct`` is still unlabelled are skipped; they carry no correctness signal.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .adaptive_conformal import AdaptiveConformalPredictor
from .miscoverage import MiscoverageMonitor

__all__ = [
    "DEFAULT_LEDGER_PATH",
    "LEDGER_PATH_ENV",
    "ColdStartSummary",
    "RecallQuery",
    "cold_start_from_ledger",
    "default_ledger_path",
    "read_recall_ledger",
]

_logger = logging.getLogger(__name__)

#: Environment variable that overrides the ledger location on both sides.
LEDGER_PATH_ENV = "REMANENTIA_RECALL_LEDGER"

#: REMANENTIA-local default path on this shared workstation. The ledger is
#: REMANENTIA's, gitignored and absent from Director-AI's checkout, so it is read
#: by absolute path; override with :data:`LEDGER_PATH_ENV` for a shared location.
DEFAULT_LEDGER_PATH = Path(
    "/media/anulum/GOTM/aaa_God_of_the_Math_Collection"
    "/03_CODE/REMANENTIA/.coordination/runtime/recall_ledger.jsonl"
)

#: Segment used when a ledger record carries no project tag.
_DEFAULT_SEGMENT = "default"


@dataclass(frozen=True)
class RecallQuery:
    """One recall merged from its query line and any outcome lines.

    Mirrors REMANENTIA's ``RecallQuery``: the query metadata plus the two
    independent outcome labels, each ``None`` until an outcome line sets it.
    """

    event_id: str
    ts: float
    by: str
    query: str
    top_k: int
    project: str
    returned_ids: tuple[str, ...]
    found: bool
    score: float | None
    abstained: bool | None
    was_used: bool | None
    was_correct: bool | None


@dataclass(frozen=True)
class ColdStartSummary:
    """What a cold-start replay consumed from the ledger.

    ``records`` is every merged recall read; ``labelled`` is how many carried a
    ``was_correct`` label and so drove calibration; ``calibrated`` is how many of
    those also had a retrieval ``score`` and entered the conformal residual set.
    ``segments`` is the set of projects primed in the miscoverage monitor.
    """

    records: int
    labelled: int
    calibrated: int
    segments: tuple[str, ...]


def default_ledger_path() -> Path:
    """Return the ledger path, honouring the ``REMANENTIA_RECALL_LEDGER`` env."""
    override = os.environ.get(LEDGER_PATH_ENV, "").strip()
    return Path(override) if override else DEFAULT_LEDGER_PATH


def _coerce_segment(project: object) -> str:
    """Return a non-empty segment key for a record's project tag."""
    if isinstance(project, str) and project.strip():
        return project.strip()
    return _DEFAULT_SEGMENT


@dataclass
class _MergeState:
    """Mutable accumulator for one recall while its outcome lines are applied.

    Holds the query-line fields plus the running labels and, per label, the ts
    at which it was last set — so a later outcome line overrides an earlier one
    while an earlier line can never clobber a newer label. :meth:`freeze` yields
    the immutable :class:`RecallQuery`.
    """

    event_id: str
    ts: float
    by: str
    query: str
    top_k: int
    project: str
    returned_ids: tuple[str, ...]
    found: bool
    score: float | None
    abstained: bool | None
    was_used: bool | None = None
    was_correct: bool | None = None
    label_ts: dict[str, float] = field(default_factory=dict)

    def apply_outcome(self, obj: dict[str, object]) -> None:
        """Merge an outcome line's labels, keeping the latest ts per field."""
        ts = _as_float(obj.get("ts")) or 0.0
        for name in ("was_used", "was_correct"):
            if name not in obj:
                continue
            value = _as_bool(obj.get(name))
            if value is None:
                continue
            if ts >= self.label_ts.get(name, float("-inf")):
                setattr(self, name, value)
                self.label_ts[name] = ts

    def freeze(self) -> RecallQuery:
        """Return the immutable record, dropping the merge-only ts bookkeeping."""
        return RecallQuery(
            event_id=self.event_id,
            ts=self.ts,
            by=self.by,
            query=self.query,
            top_k=self.top_k,
            project=self.project,
            returned_ids=self.returned_ids,
            found=self.found,
            score=self.score,
            abstained=self.abstained,
            was_used=self.was_used,
            was_correct=self.was_correct,
        )


def _merge_lines(lines: Iterable[str]) -> list[RecallQuery]:
    """Merge raw JSONL lines into per-event records (latest outcome wins)."""
    states: dict[str, _MergeState] = {}
    for raw in lines:
        text = raw.strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            _logger.debug("skipping malformed recall-ledger line")
            continue
        if not isinstance(obj, dict):
            continue
        event_id = obj.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            continue
        kind = obj.get("kind")
        if kind == "query":
            states[event_id] = _query_state(obj)
        elif kind == "outcome":
            state = states.get(event_id)
            if state is None:
                # Outcome before its query line — nothing to attach it to.
                continue
            state.apply_outcome(obj)
    return [state.freeze() for state in states.values()]


def _query_state(obj: dict[str, object]) -> _MergeState:
    """Build a merge accumulator from a query line, labels seeded unset."""
    returned = obj.get("returned_ids")
    returned_ids = (
        tuple(str(item) for item in returned) if isinstance(returned, list) else ()
    )
    return _MergeState(
        event_id=str(obj.get("event_id", "")),
        ts=_as_float(obj.get("ts")) or 0.0,
        by=str(obj.get("by", "")),
        query=str(obj.get("query", "")),
        top_k=_as_int(obj.get("top_k")),
        project=_coerce_segment(obj.get("project")),
        returned_ids=returned_ids,
        found=bool(obj.get("found", False)),
        score=_as_float(obj.get("score")),
        abstained=_as_bool(obj.get("abstained")),
    )


def read_recall_ledger(path: Path | None = None) -> list[RecallQuery]:
    """Read and merge the recall ledger; empty when the file is absent.

    ``path`` defaults to :func:`default_ledger_path`. A missing ledger returns an
    empty list (the loop may not have produced any recalls yet) rather than
    raising, so a cold start before the first recall is a no-op, not an error.
    """
    ledger = path or default_ledger_path()
    if not ledger.exists():
        _logger.info("recall ledger not found at %s; nothing to cold-start", ledger)
        return []
    with ledger.open(encoding="utf-8") as handle:
        return _merge_lines(handle)


def cold_start_from_ledger(
    predictor: AdaptiveConformalPredictor,
    monitor: MiscoverageMonitor,
    *,
    path: Path | None = None,
    records: Iterable[RecallQuery] | None = None,
) -> ColdStartSummary:
    """Prime ``predictor`` and ``monitor`` from the ledger's correctness history.

    Replays each merged recall in ledger order. A record with a ``was_correct``
    label feeds the miscoverage monitor (segmented by project) and nudges the
    adaptive predictor's ``alpha_t`` toward target; if it also carries a
    retrieval ``score`` it enters the conformal residual set. ``was_used`` is
    ignored — coverage is calibrated on correctness only. Records without a
    correctness label contribute nothing.

    Pass ``records`` to replay an already-read sequence; otherwise the ledger at
    ``path`` (default :func:`default_ledger_path`) is read. Returns a
    :class:`ColdStartSummary` of what was consumed.
    """
    history = list(records) if records is not None else read_recall_ledger(path)
    labelled = 0
    calibrated = 0
    segments: set[str] = set()
    for record in history:
        if record.was_correct is None:
            continue
        labelled += 1
        segment = record.project or _DEFAULT_SEGMENT
        segments.add(segment)
        monitor.observe(segment, covered=record.was_correct)
        if record.score is not None:
            predictor.add_observation(record.score, correct_label=record.was_correct)
            calibrated += 1
        predictor.update(covered=record.was_correct)
    return ColdStartSummary(
        records=len(history),
        labelled=labelled,
        calibrated=calibrated,
        segments=tuple(sorted(segments)),
    )


def _as_int(value: object) -> int:
    """Return a genuine JSON integer, or ``0`` for null/non-integer/boolean."""
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _as_float(value: object) -> float | None:
    """Coerce a JSON number to float, or ``None`` for null/non-numeric."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _as_bool(value: object) -> bool | None:
    """Return a bool only for a genuine JSON boolean, else ``None``."""
    return value if isinstance(value, bool) else None
