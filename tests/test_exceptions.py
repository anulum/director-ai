# ─────────────────────────────────────────────────────────────────────
# Tests — Exception hierarchy
# ─────────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass

import pytest

from director_ai.core.exceptions import (
    CoherenceError,
    DependencyError,
    DirectorAIError,
    GeneratorError,
    HallucinationError,
    KernelHaltError,
    ValidationError,
)

# ── Inheritance chain ────────────────────────────────────────────────


class TestHierarchy:
    @pytest.mark.parametrize(
        "exc_cls",
        [
            CoherenceError,
            KernelHaltError,
            GeneratorError,
            ValidationError,
            DependencyError,
            HallucinationError,
        ],
    )
    def test_all_inherit_from_base(self, exc_cls):
        assert issubclass(exc_cls, DirectorAIError)

    def test_base_inherits_exception(self):
        assert issubclass(DirectorAIError, Exception)

    def test_validation_is_also_value_error(self):
        assert issubclass(ValidationError, ValueError)

    def test_catch_all_with_base(self):
        """A single except DirectorAIError catches every library exception."""
        for cls in (
            CoherenceError,
            KernelHaltError,
            GeneratorError,
            DependencyError,
        ):
            with pytest.raises(DirectorAIError):
                raise cls("test")

    def test_validation_caught_by_value_error(self):
        with pytest.raises(ValueError):
            raise ValidationError("bad input")


# ── HallucinationError attributes ────────────────────────────────────


@dataclass
class _FakeScore:
    score: float = 0.3


class TestHallucinationError:
    def test_stores_query(self):
        err = HallucinationError("q", "r", _FakeScore(0.3))
        assert err.query == "q"

    def test_stores_response(self):
        err = HallucinationError("q", "r", _FakeScore(0.3))
        assert err.response == "r"

    def test_stores_score(self):
        err = HallucinationError("q", "r", _FakeScore(0.3))
        assert err.score.score == pytest.approx(0.3)

    def test_message_contains_coherence(self):
        err = HallucinationError("q", "r", _FakeScore(0.3))
        assert "0.300" in str(err)

    def test_message_truncates_response(self):
        long_response = "x" * 200
        err = HallucinationError("q", long_response, _FakeScore(0.1))
        assert len(str(err)) < 200

    def test_inherits_base(self):
        with pytest.raises(DirectorAIError):
            raise HallucinationError("q", "r", _FakeScore(0.5))
