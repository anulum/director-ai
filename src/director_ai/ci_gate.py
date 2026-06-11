# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — CI eval gate
"""CI quality gate: run the guardrail over a labelled set, fail under a threshold.

This is the building block for the GitHub Action and the ``director-ai ci-gate``
command. It mirrors how teams gate prompt/eval suites in CI (Promptfoo,
Braintrust): score each case with the coherence scorer, compare the
approve/reject decision against the expected label, and report pass/fail against
configurable thresholds so a CI job can block a regression.

A case is one ``(prompt, response, expected)`` triple where ``expected`` is
``"approve"`` for a grounded answer that should pass and ``"reject"`` for a
hallucination that the guard should catch. The scorer is injected and only needs
a ``review(prompt, response) -> (bool, score)`` method, so the gate is testable
without a model and works with any backend the user has configured.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

Label = str  # "approve" | "reject"
_APPROVE: Label = "approve"
_REJECT: Label = "reject"
_LABELS = (_APPROVE, _REJECT)


class _Reviewer(Protocol):
    """Minimal scorer surface the gate needs (duck-typed)."""

    def review(self, prompt: str, response: str) -> tuple[bool, object]: ...


@dataclass(frozen=True)
class EvalCase:
    """One labelled gate case.

    ``expected`` is ``"approve"`` when a correct guard should let the response
    through (grounded answer) and ``"reject"`` when it should block it
    (hallucination). ``case_id`` defaults to the 1-based load order.
    """

    prompt: str
    response: str
    expected: Label
    case_id: str = ""

    def __post_init__(self) -> None:
        if self.expected not in _LABELS:
            raise ValueError(
                f"case {self.case_id or '?'}: expected must be one of {_LABELS}, "
                f"got {self.expected!r}"
            )


@dataclass(frozen=True)
class CaseOutcome:
    """The gate's decision for a single case."""

    case_id: str
    expected: Label
    predicted: Label
    score: float | None
    correct: bool


@dataclass(frozen=True)
class GateThresholds:
    """Pass/fail thresholds. ``None`` disables a check."""

    min_accuracy: float = 0.0
    min_catch_rate: float | None = None
    max_false_halt_rate: float | None = None


@dataclass(frozen=True)
class GateReport:
    """Aggregate gate result. ``passed`` is the CI exit signal."""

    total: int
    correct: int
    accuracy: float
    catch_rate: float | None
    false_halt_rate: float | None
    passed: bool
    thresholds: GateThresholds
    failures: tuple[str, ...]
    outcomes: tuple[CaseOutcome, ...] = field(default_factory=tuple)

    def to_dict(self, *, include_outcomes: bool = True) -> dict[str, object]:
        """Render to a JSON-serialisable summary for CI artefacts."""
        data: dict[str, object] = {
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 6),
            "catch_rate": None
            if self.catch_rate is None
            else round(self.catch_rate, 6),
            "false_halt_rate": (
                None if self.false_halt_rate is None else round(self.false_halt_rate, 6)
            ),
            "passed": self.passed,
            "thresholds": {
                "min_accuracy": self.thresholds.min_accuracy,
                "min_catch_rate": self.thresholds.min_catch_rate,
                "max_false_halt_rate": self.thresholds.max_false_halt_rate,
            },
            "failures": list(self.failures),
        }
        if include_outcomes:
            data["outcomes"] = [
                {
                    "case_id": o.case_id,
                    "expected": o.expected,
                    "predicted": o.predicted,
                    "score": o.score,
                    "correct": o.correct,
                }
                for o in self.outcomes
            ]
        return data

    def summary_lines(self) -> list[str]:
        """Human-readable summary for the CI log."""
        catch = "n/a" if self.catch_rate is None else f"{self.catch_rate:.1%}"
        halt = "n/a" if self.false_halt_rate is None else f"{self.false_halt_rate:.1%}"
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"Director-AI CI gate: {status}",
            f"  cases            : {self.total}",
            f"  accuracy         : {self.accuracy:.1%} ({self.correct}/{self.total})",
            f"  hallucination catch rate : {catch}",
            f"  false-halt rate          : {halt}",
        ]
        lines.extend(f"  ✗ {reason}" for reason in self.failures)
        return lines


def load_cases(path: str | Path) -> list[EvalCase]:
    """Load gate cases from a JSONL file (one ``{prompt, response, expected}`` per line).

    Blank lines are skipped. ``id`` is optional; missing ids default to the
    1-based line order. Raises ``ValueError`` with the line number on a malformed
    line so CI logs point at the offending case.
    """
    cases: list[EvalCase] = []
    text = Path(path).read_text(encoding="utf-8")
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{lineno}: expected a JSON object")
        missing = {"prompt", "response", "expected"} - obj.keys()
        if missing:
            raise ValueError(f"{path}:{lineno}: missing field(s) {sorted(missing)}")
        case_id = str(obj.get("id", lineno))
        try:
            cases.append(
                EvalCase(
                    prompt=str(obj["prompt"]),
                    response=str(obj["response"]),
                    expected=str(obj["expected"]),
                    case_id=case_id,
                )
            )
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno}: {exc}") from exc
    if not cases:
        raise ValueError(f"{path}: no cases found")
    return cases


def _score_float(score: object) -> float | None:
    """Extract a float from a scorer's score object (``.score`` attr or numeric)."""
    value = getattr(score, "score", score)
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def run_eval_gate(
    cases: Sequence[EvalCase],
    scorer: _Reviewer,
    thresholds: GateThresholds,
) -> GateReport:
    """Score every case, compare to its label, and apply the thresholds.

    ``accuracy`` is over all cases. ``catch_rate`` is recall on the ``reject``
    (hallucination) cases — the share the guard correctly blocked — and is
    ``None`` when there are no such cases. ``false_halt_rate`` is the share of
    ``approve`` (grounded) cases the guard wrongly blocked, and is ``None`` when
    there are no such cases. The gate ``passed`` only when every configured
    threshold holds; each breach is recorded in ``failures``.
    """
    if not cases:
        raise ValueError("run_eval_gate requires at least one case")

    outcomes: list[CaseOutcome] = []
    reject_total = reject_caught = 0
    approve_total = approve_halted = 0
    for case in cases:
        approved, raw_score = scorer.review(case.prompt, case.response)
        predicted: Label = _APPROVE if approved else _REJECT
        correct = predicted == case.expected
        outcomes.append(
            CaseOutcome(
                case_id=case.case_id,
                expected=case.expected,
                predicted=predicted,
                score=_score_float(raw_score),
                correct=correct,
            )
        )
        if case.expected == _REJECT:
            reject_total += 1
            reject_caught += int(predicted == _REJECT)
        else:
            approve_total += 1
            approve_halted += int(predicted == _REJECT)

    total = len(cases)
    correct_count = sum(o.correct for o in outcomes)
    accuracy = correct_count / total
    catch_rate = reject_caught / reject_total if reject_total else None
    false_halt_rate = approve_halted / approve_total if approve_total else None

    failures = _threshold_failures(thresholds, accuracy, catch_rate, false_halt_rate)
    return GateReport(
        total=total,
        correct=correct_count,
        accuracy=accuracy,
        catch_rate=catch_rate,
        false_halt_rate=false_halt_rate,
        passed=not failures,
        thresholds=thresholds,
        failures=tuple(failures),
        outcomes=tuple(outcomes),
    )


def _threshold_failures(
    thresholds: GateThresholds,
    accuracy: float,
    catch_rate: float | None,
    false_halt_rate: float | None,
) -> list[str]:
    """Return one message per breached threshold (empty == gate passes)."""
    failures: list[str] = []
    if accuracy < thresholds.min_accuracy:
        failures.append(
            f"accuracy {accuracy:.1%} < required {thresholds.min_accuracy:.1%}"
        )
    if thresholds.min_catch_rate is not None:
        if catch_rate is None:
            failures.append(
                "min-catch-rate set but the dataset has no reject-labelled cases"
            )
        elif catch_rate < thresholds.min_catch_rate:
            failures.append(
                f"catch rate {catch_rate:.1%} < required "
                f"{thresholds.min_catch_rate:.1%}"
            )
    if thresholds.max_false_halt_rate is not None:
        if false_halt_rate is None:
            failures.append(
                "max-false-halt-rate set but the dataset has no approve-labelled cases"
            )
        elif false_halt_rate > thresholds.max_false_halt_rate:
            failures.append(
                f"false-halt rate {false_halt_rate:.1%} > allowed "
                f"{thresholds.max_false_halt_rate:.1%}"
            )
    return failures


def gate_from_cases(
    cases: Iterable[EvalCase],
    scorer: _Reviewer,
    *,
    min_accuracy: float = 0.0,
    min_catch_rate: float | None = None,
    max_false_halt_rate: float | None = None,
) -> GateReport:
    """Convenience wrapper building :class:`GateThresholds` from keyword args."""
    return run_eval_gate(
        list(cases),
        scorer,
        GateThresholds(
            min_accuracy=min_accuracy,
            min_catch_rate=min_catch_rate,
            max_false_halt_rate=max_false_halt_rate,
        ),
    )
