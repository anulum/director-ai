# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for numeric verification pipeline."""

from __future__ import annotations

from director_ai.core.verification.numeric_verifier import (
    NumericVerificationResult,
    verify_numeric,
)


class TestPercentageArithmetic:
    def test_correct_percentage(self):
        text = "Revenue increased 20% from $10,000 to $12,000."
        result = verify_numeric(text)
        assert result.error_count == 0

    def test_wrong_percentage(self):
        text = "Revenue grew 15% from $10,000 to $12,000."
        result = verify_numeric(text)
        assert result.error_count >= 1
        assert any(i.issue_type == "arithmetic" for i in result.issues)
        assert not result.valid

    def test_correct_decrease(self):
        text = "Sales decreased 25% from $100 to $75."
        result = verify_numeric(text)
        assert result.error_count == 0


class TestDateLogic:
    def test_death_before_birth(self):
        text = "He was born in 1950 and died in 1920."
        result = verify_numeric(text)
        assert any(i.issue_type == "date_logic" for i in result.issues)
        assert any("Death year" in i.description for i in result.issues)

    def test_valid_birth_death(self):
        text = "She was born in 1900 and died in 1980."
        result = verify_numeric(text)
        date_errors = [
            i
            for i in result.issues
            if i.issue_type == "date_logic" and i.severity == "error"
        ]
        assert len(date_errors) == 0

    def test_far_future_year(self):
        text = "The project will be completed by 2099."
        result = verify_numeric(text)
        assert any(
            i.issue_type == "date_logic" and "far future" in i.description
            for i in result.issues
        )

    def test_founded_in_future(self):
        text = "The company was founded in 2050."
        result = verify_numeric(text)
        assert any(
            i.issue_type == "date_logic" and "future" in i.description
            for i in result.issues
        )


class TestProbabilityBounds:
    def test_probability_over_100(self):
        text = "There is a 150% probability that it will rain."
        result = verify_numeric(text)
        assert any(i.issue_type == "probability" for i in result.issues)

    def test_valid_probability(self):
        text = "There is a 75% probability of success."
        result = verify_numeric(text)
        prob_issues = [i for i in result.issues if i.issue_type == "probability"]
        assert len(prob_issues) == 0

    def test_negative_probability(self):
        text = "There is a -5% probability of success."
        result = verify_numeric(text)
        assert any(i.issue_type == "probability" for i in result.issues)


class TestMagnitude:
    def test_earth_population_wrong(self):
        text = "Earth's population is 80 billion people."
        result = verify_numeric(text)
        assert any(i.issue_type == "magnitude" for i in result.issues)

    def test_earth_population_correct(self):
        text = "Earth's population is 8 billion people."
        result = verify_numeric(text)
        mag_issues = [i for i in result.issues if i.issue_type == "magnitude"]
        assert len(mag_issues) == 0


class TestInternalConsistency:
    def test_inconsistent_totals(self):
        text = "The total of 500 units were shipped. Later, a total of 600 units were counted."
        result = verify_numeric(text)
        assert any(i.issue_type == "internal" for i in result.issues)

    def test_consistent_totals(self):
        text = "A total of 500 units shipped. The total of 500 was confirmed."
        result = verify_numeric(text)
        internal_issues = [i for i in result.issues if i.issue_type == "internal"]
        assert len(internal_issues) == 0


class TestCleanText:
    def test_no_numbers(self):
        text = "The quick brown fox jumps over the lazy dog."
        result = verify_numeric(text)
        assert result.valid
        assert result.error_count == 0

    def test_simple_numbers_no_issues(self):
        text = "The meeting is at 3pm in room 42."
        result = verify_numeric(text)
        assert result.valid


class TestResultStructure:
    def test_result_properties(self):
        result = NumericVerificationResult(claims_found=5, issues=[], valid=True)
        assert result.error_count == 0
        assert result.warning_count == 0
        assert result.valid
