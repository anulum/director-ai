# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the WS-3 grader-validation contract (the FPR of the honesty gate).

Covers the overclaim semantics (a render asserting more support than the truth,
with under-claims and false-refutations correctly excluded), per-status FPR/FNR
confusion, coverage, the Hoeffding upper bounds, the empty/bad-confidence guards,
report serialisation, and every admissibility release-blocker rule (overclaim
above alpha, a missed falsification, and the abstention-crutch coverage floor).
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from director_ai.core.calibration.grader_validation import (
    GraderCase,
    GraderReport,
    GraderStatus,
    GraderValidationError,
    assert_grader_admissible,
    validate_grader,
)


def _grader(inputs: Mapping[str, object]) -> GraderStatus:
    """A controllable grader: it renders whatever the case put in ``predict``."""
    return inputs["predict"]  # type: ignore[return-value]


def _case(case_id: str, predict: GraderStatus, truth: GraderStatus) -> GraderCase:
    return GraderCase(case_id=case_id, inputs={"predict": predict}, ground_truth=truth)


# --- overclaim semantics ----------------------------------------------------


def test_perfect_grader_has_zero_overclaim_full_coverage() -> None:
    cases = [
        _case("a", "validated", "validated"),
        _case("b", "bounded", "bounded"),
        _case("c", "refuted", "refuted"),
        _case("d", "falsified", "falsified"),
    ]
    report = validate_grader(_grader, cases)
    assert report.headline_overclaim_rate == 0.0
    assert report.coverage == 1.0  # none rendered unknown
    assert report.n == 4


def test_validated_for_bounded_is_an_overclaim() -> None:
    cases = [_case("a", "validated", "bounded")]
    report = validate_grader(_grader, cases)
    assert report.headline_overclaim_rate == 1.0


def test_bounded_for_refuted_is_an_overclaim() -> None:
    report = validate_grader(_grader, [_case("a", "bounded", "refuted")])
    assert report.headline_overclaim_rate == 1.0


def test_unknown_for_validated_is_not_an_overclaim() -> None:
    """Abstaining on a true positive is conservative (under-claim), not overclaim."""
    report = validate_grader(_grader, [_case("a", "unknown", "validated")])
    assert report.headline_overclaim_rate == 0.0
    assert report.coverage == 0.0  # it abstained


def test_refuted_for_validated_is_not_an_overclaim() -> None:
    """A false refutation is its own error (refuted FPR), not a support overclaim."""
    report = validate_grader(_grader, [_case("a", "refuted", "validated")])
    assert report.headline_overclaim_rate == 0.0
    assert report.status("refuted").false_positive_rate == 1.0


# --- per-status confusion ---------------------------------------------------


def test_per_status_fpr_and_fnr() -> None:
    cases = [
        _case("a", "validated", "validated"),  # validated TP
        _case("b", "validated", "bounded"),  # validated FP (and bounded FN)
        _case("c", "bounded", "bounded"),  # bounded TP
        _case("d", "unknown", "validated"),  # validated FN
    ]
    report = validate_grader(_grader, cases)
    validated = report.status("validated")
    # truth==validated for a,d (support 2); predicted validated for a,b.
    assert validated.support == 2
    assert validated.predicted == 2
    # FP: predicted validated while truth != validated → case b. negatives = 2 (b,c).
    assert validated.false_positive_rate == pytest.approx(0.5)
    # FN: truth validated but predicted != validated → case d. support 2.
    assert validated.false_negative_rate == pytest.approx(0.5)
    assert 0.5 <= validated.false_positive_upper <= 1.0


def test_coverage_excludes_unknown_only() -> None:
    cases = [
        _case("a", "validated", "validated"),
        _case("b", "unknown", "bounded"),
        _case("c", "refuted", "refuted"),  # refuted counts as coverage (it answered)
    ]
    report = validate_grader(_grader, cases)
    assert report.coverage == pytest.approx(2 / 3)


# --- guards -----------------------------------------------------------------


def test_empty_cases_raises() -> None:
    with pytest.raises(ValueError, match="empty case set"):
        validate_grader(_grader, [])


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_bad_confidence_raises(bad: float) -> None:
    with pytest.raises(ValueError, match="confidence must be in"):
        validate_grader(_grader, [_case("a", "validated", "validated")], confidence=bad)


# --- serialisation ----------------------------------------------------------


def test_report_to_dict_shape() -> None:
    report = validate_grader(_grader, [_case("a", "validated", "validated")])
    payload = report.to_dict()
    assert payload["schema"] == "studio.grader-report.v1"
    assert payload["n"] == 1
    assert "headline_overclaim_rate" in payload
    assert "coverage" in payload
    assert isinstance(payload["per_status"], list)
    assert payload["per_status"][0]["status"] == "validated"


def test_status_lookup_unknown_key_raises() -> None:
    report = validate_grader(_grader, [_case("a", "validated", "validated")])
    with pytest.raises(KeyError):
        report.status("not-a-status")  # type: ignore[arg-type]


# --- admissibility (release-blocker) rules ----------------------------------


def _clean_cases() -> list[GraderCase]:
    # 20 correct cases across classes incl. a covered falsified; clean grader.
    cases = [_case(f"v{i}", "validated", "validated") for i in range(8)]
    cases += [_case(f"b{i}", "bounded", "bounded") for i in range(6)]
    cases += [_case(f"r{i}", "refuted", "refuted") for i in range(3)]
    cases += [_case(f"f{i}", "falsified", "falsified") for i in range(3)]
    return cases


def test_admissible_clean_grader_passes() -> None:
    report = validate_grader(_grader, _clean_cases())
    # With zero overclaims the Hoeffding upper bound on 0/20 may still exceed a
    # tight alpha; assert with a realistic alpha for a 20-case smoke set.
    assert_grader_admissible(report, overclaim_alpha=0.5, coverage_floor=0.7)


def test_overclaim_above_alpha_raises() -> None:
    cases = [_case(f"o{i}", "validated", "bounded") for i in range(10)]
    report = validate_grader(_grader, cases)
    with pytest.raises(GraderValidationError, match="above its evidence"):
        assert_grader_admissible(report, overclaim_alpha=0.05)


def test_missed_falsification_raises() -> None:
    cases = _clean_cases()
    # Corrupt one falsified case: the grader downgrades it to bounded (a miss).
    cases.append(_case("miss", "bounded", "falsified"))
    report = validate_grader(_grader, cases)
    with pytest.raises(GraderValidationError, match="missed a falsification"):
        assert_grader_admissible(report, overclaim_alpha=0.9, coverage_floor=0.0)


def test_abstention_crutch_coverage_floor_raises() -> None:
    # A grader that abstains on everything: zero overclaim, but useless.
    cases = [_case(f"u{i}", "unknown", "validated") for i in range(10)]
    report = validate_grader(_grader, cases)
    assert report.headline_overclaim_rate == 0.0
    with pytest.raises(GraderValidationError, match="abstention crutch"):
        assert_grader_admissible(report, overclaim_alpha=0.9, coverage_floor=0.7)


def test_public_surface_reexports() -> None:
    from director_ai.core import GraderReport as ExportedReport
    from director_ai.core import validate_grader as exported_fn

    assert ExportedReport is GraderReport
    assert exported_fn is validate_grader
