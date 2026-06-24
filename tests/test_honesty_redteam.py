# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the WS-1 adversarial strip-resistance battery.

Runs the real published ``scpn_studio_platform.seal`` (the ``honesty`` extra) and
asserts the battery catches every attack — strip-overclaim, tamper-unit, the
near-threshold margin-forge (decision-value separation), foreign-unit replay,
unknown-key, and missing-envelope — with zero survivors, that the survivor gate
raises, that an equal-grade strip is rejected as vacuous, and that the gate is
loud when the seal SDK is absent (never a silent pass).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

pytest.importorskip(
    "scpn_studio_platform.seal",
    reason="the honesty extra (scpn-studio-platform>=0.9, py>=3.12) is not installed",
)

from director_ai.core.calibration import honesty_redteam as rt  # noqa: E402
from director_ai.core.calibration.honesty_redteam import (  # noqa: E402
    AttackOutcome,
    BatteryReport,
    HonestyRedteamError,
    assert_zero_survivors,
    run_strip_resistance_battery,
)


def _regrade(unit: Mapping[str, Any]) -> str:
    """Margin-aware regrade: validated only when the signed margin clears 1.0."""
    return "validated" if float(unit.get("margin", 0.0)) >= 1.0 else "bounded"


def _run() -> BatteryReport:
    return run_strip_resistance_battery(
        unit={"claim": "a grounded claim", "margin": 0.5},
        true_grade="bounded",
        regrade=_regrade,
        inflated_grade="validated",
    )


def test_battery_catches_every_attack_zero_survivors() -> None:
    report = _run()
    assert report.all_blocked is True
    assert report.survivors == ()
    names = {o.attack for o in report.outcomes}
    assert names == {
        "strip-overclaim",
        "tamper-unit",
        "near-threshold-margin-forge",
        "replay-foreign-unit",
        "unknown-key",
        "missing-envelope",
    }
    # Every attack verdict is a non-VERIFIED loud state.
    assert all(not o.survived for o in report.outcomes)
    assert all(o.verdict in {"forged", "stripped", "ungraded"} for o in report.outcomes)


def test_assert_zero_survivors_passes_on_clean_report() -> None:
    assert_zero_survivors(_run())  # does not raise


def test_assert_zero_survivors_raises_on_a_survivor() -> None:
    report = BatteryReport(
        outcomes=(AttackOutcome("forged-pass", "verified", survived=True),),
        survivors=("forged-pass",),
        all_blocked=False,
    )
    with pytest.raises(HonestyRedteamError, match="release blocker"):
        assert_zero_survivors(report)


def test_vacuous_strip_grade_rejected() -> None:
    with pytest.raises(ValueError, match="over-claim"):
        run_strip_resistance_battery(
            unit={"claim": "x", "margin": 2.0},
            true_grade="validated",
            regrade=_regrade,
            inflated_grade="validated",
        )


def test_report_and_outcome_serialise() -> None:
    report = _run()
    payload = report.to_dict()
    assert payload["all_blocked"] is True
    assert payload["survivors"] == []
    assert isinstance(payload["outcomes"], list)
    first = payload["outcomes"][0]
    assert {"attack", "verdict", "survived"} <= set(first)


def test_margin_forge_is_caught_specifically() -> None:
    """The decision-value-separation attack must be forged, not verified."""
    report = _run()
    margin = next(
        o for o in report.outcomes if o.attack == "near-threshold-margin-forge"
    )
    assert margin.verdict == "forged"
    assert margin.survived is False


def test_unavailable_seal_is_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    """A security gate must raise — never silently pass — when the SDK is absent."""
    monkeypatch.setattr(rt, "_SEAL_AVAILABLE", False)
    with pytest.raises(HonestyRedteamError, match="honesty"):
        run_strip_resistance_battery(
            unit={"claim": "x"}, true_grade="bounded", regrade=_regrade
        )
