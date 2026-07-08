# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ADR-0001 numeric_verifier fallback bit-exactness parity
"""Bit-exactness parity for the ``numeric_verifier`` pure-Python floor (ADR-0001).

``verify_numeric`` accelerates via ``backfire_kernel.rust_verify_numeric`` when the
compiled kernel is installed, and rebuilds the identical
``NumericVerificationResult`` in pure Python when it is not (the opt-in ``[rust]``
extra is absent). ADR-0001 requires that floor to be *bit-exact* with the kernel —
same ``claims_found``, same issue ``(type, description, severity, context)``
tuples (including the whole-vs-fractional number rendering), same ``valid``.

These tests force the Python branch (``_RUST_NUMERIC = False``) with a frozen
"current year" and compare it, with no mocks, to the real kernel over a fixed
corpus and a randomised generator. Skipped only when the compiled kernel is
unavailable (a base install), where there is nothing to compare against.
"""

from __future__ import annotations

import datetime as _dt
import random

import pytest

import director_ai.core.verification.numeric_verifier as nv

try:
    from backfire_kernel import rust_verify_numeric as _rust_verify_numeric

    _HAS_RUST = True
except ImportError:  # pragma: no cover - exercised only in a no-kernel install
    _HAS_RUST = False

_needs_rust = pytest.mark.skipif(
    not _HAS_RUST, reason="backfire_kernel (compiled) not installed"
)

_YEAR = 2026


class _FrozenDatetime(_dt.datetime):
    """A ``datetime`` whose ``now()`` is pinned, so both paths see the same year."""

    @classmethod
    def now(cls, tz: _dt.tzinfo | None = None) -> _FrozenDatetime:
        return cls(_YEAR, 6, 15, 12, 0, 0)


@pytest.fixture
def python_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the pure-Python branch and pin the year for the whole test."""
    monkeypatch.setattr(nv, "_RUST_NUMERIC", False)
    monkeypatch.setattr(nv, "datetime", _FrozenDatetime)


def _python_result(text: str) -> tuple[int, list[tuple[str, str, str, str]], bool]:
    """Run the pure-Python fallback (requires the ``python_floor`` fixture)."""
    result = nv.verify_numeric(text)
    return (
        result.claims_found,
        [(i.issue_type, i.description, i.severity, i.context) for i in result.issues],
        result.valid,
    )


def _rust_result(text: str) -> tuple[int, list[tuple[str, str, str, str]], bool]:
    claims_found, raw_issues, valid = _rust_verify_numeric(text, _YEAR)
    return (claims_found, [tuple(issue) for issue in raw_issues], valid)


_CORPUS = [
    "The company grew 15% from $10 million to $12 million.",
    "Revenue increased by 25% from 100 to 130 last year.",
    "Sales dropped 40% from 500 to 200 units.",
    "It grew 80% from 10.5 to 20.7 units.",
    "Rose 3% from 100.25 to 100.5 points.",
    "He was born in 1990 and died in 1985.",
    "The firm was founded in 2050.",
    "The project will be completed by 2099.",
    "There is a 150% probability of rain.",
    "The probability is -20% confidence.",
    "Earth population is 80 billion people.",
    "Earth's population is 800 million people.",
    "Speed of light is 500000 km/s in vacuum.",
    "The total of 1000 items, later a total of 1050 items.",
    "The total of 10.5, later a total of 20.25 recorded.",
    "Nothing numeric here at all, just words.",
    "In 2099 the population reached 8 billion, up 12% from 7 to 8 billion.",
    "Profit rose 33% from 3 to 4 and fell 50% from 4 to 2.",
    "A 99.99% chance and a 100.01% probability were reported.",
    "grew 50% from 0 to 100 units.",
    "",
    "100% probability certain, 0% probability impossible.",
    "The metric increased 12.345% from 987.654 to 1109.4 last cycle.",
    "born 2000 died 2000, founded 2026, completes by 2031.",
]


@_needs_rust
@pytest.mark.parametrize("text", _CORPUS)
@pytest.mark.usefixtures("python_floor")
def test_parity_fixed_corpus(text: str) -> None:
    assert _python_result(text) == _rust_result(text)


def _num(rng: random.Random) -> str:
    """A regex-friendly decimal — whole or up to three fractional digits."""
    if rng.randint(0, 9) == 0:
        return str(rng.randint(0, 999999))
    return f"{round(rng.uniform(0.1, 500000), rng.randint(1, 3))}"


def _gen(rng: random.Random) -> str:
    verb = rng.choice(
        ["grew", "increased", "decreased", "dropped", "rose", "fell", "gained", "lost"]
    )
    templates = [
        lambda: f"Revenue {verb} {_num(rng)}% from ${_num(rng)} to ${_num(rng)}.",
        lambda: f"The change was {_num(rng)}% from {_num(rng)} to {_num(rng)} million.",
        lambda: f"There is a {_num(rng)}% probability of success.",
        lambda: f"A -{_num(rng)}% chance was reported.",
        lambda: f"Earth's population is {_num(rng)} billion people.",
        lambda: f"Earth population is {_num(rng)} million residents.",
        lambda: f"The speed of light is {_num(rng)} km/s in vacuum.",
        lambda: f"The total of {_num(rng)} items, later a total of {_num(rng)} items.",
        lambda: (
            f"He was born in {rng.randint(1900, 2100)} and "
            f"died in {rng.randint(1900, 2100)}."
        ),
        lambda: f"The company was founded in {rng.randint(1900, 2100)}.",
        lambda: f"The project completes by {rng.randint(2000, 2200)}.",
        lambda: f"Nothing numeric here, {rng.choice(['alpha', 'beta'])} words only.",
    ]
    return " ".join(rng.choice(templates)() for _ in range(rng.randint(1, 3)))


@_needs_rust
@pytest.mark.usefixtures("python_floor")
def test_parity_randomised() -> None:
    rng = random.Random(20260708)
    for _ in range(4000):
        text = _gen(rng)
        assert _python_result(text) == _rust_result(text), text
