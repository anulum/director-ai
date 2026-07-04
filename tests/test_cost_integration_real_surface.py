# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real public-surface coverage for cost tracking integration."""

from __future__ import annotations

import json
import os
from typing import Any, cast

import pytest

from director_ai.cli import main
from director_ai.compliance.cost_analyser import CostAnalyser
from director_ai.core.config import DirectorConfig
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


def _clear_director_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove ambient Director configuration so CLI tests stay deterministic."""
    for key in tuple(os.environ):
        if key.startswith("DIRECTOR_"):
            monkeypatch.delenv(key, raising=False)


def _json_report(text: str) -> dict[str, Any]:
    """Parse a CLI JSON report as a typed mapping."""
    decoded = json.loads(text)
    assert isinstance(decoded, dict)
    return cast(dict[str, Any], decoded)


def test_cost_integration_unit_guard_declares_this_real_surface_companion() -> None:
    """The cost integration unit guard should declare this companion."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_cost_integration.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_cost_integration_real_surface.py" in category


def test_public_cost_analyser_reports_agent_scoped_usage() -> None:
    """CostAnalyser should aggregate model and agent-scoped public reports."""
    analyser = CostAnalyser(currency="CHF")
    analyser.add_pricing("local-judge", input_per_1k=0.01, output_per_1k=0.02)

    analyser.record("local-judge", input_tokens=1000, output_tokens=250)
    analyser.record(
        "local-judge",
        input_tokens=500,
        output_tokens=100,
        agent_id="reviewer",
    )

    report = analyser.report()

    assert report["currency"] == "CHF"
    assert report["total_tokens"] == 1850
    assert report["total_cost"] == pytest.approx(0.022)
    assert report["models"]["local-judge"]["call_count"] == 1
    assert report["models"]["local-judge"]["estimated_cost"] == pytest.approx(0.015)
    assert report["models"]["local-judge::reviewer"]["call_count"] == 1
    assert report["models"]["local-judge::reviewer"]["estimated_cost"] == pytest.approx(
        0.007
    )


def test_configured_scorer_records_costs_through_public_analyser() -> None:
    """DirectorConfig should wire judge usage into the attached CostAnalyser."""
    scorer = DirectorConfig(
        cost_tracking_enabled=True,
        use_nli=False,
    ).build_scorer()

    analyser = getattr(scorer, "_cost_analyser", None)
    judge = getattr(scorer, "_judge", None)
    callback = getattr(judge, "_cost_callback", None)
    assert isinstance(analyser, CostAnalyser)
    assert callable(callback)

    callback("gpt-4o-mini", 1200, 300)
    report = analyser.report()

    assert report["total_tokens"] == 1500
    assert report["models"]["gpt-4o-mini"]["call_count"] == 1
    assert report["models"]["gpt-4o-mini"]["input_tokens"] == 1200
    assert report["models"]["gpt-4o-mini"]["output_tokens"] == 300
    assert report["total_cost"] > 0.0


def test_cli_cost_report_uses_environment_enabled_scorer(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The public CLI should render a cost report when tracking is enabled."""
    _clear_director_env(monkeypatch)
    monkeypatch.setenv("DIRECTOR_COST_TRACKING_ENABLED", "true")
    monkeypatch.setenv("DIRECTOR_USE_NLI", "false")
    monkeypatch.setenv("DIRECTOR_MODE", "general")

    main(["cost-report", "--format", "json"])

    report = _json_report(capsys.readouterr().out)
    assert report == {
        "currency": "CHF",
        "total_cost": 0.0,
        "total_tokens": 0,
        "models": {},
    }


def test_cli_cost_report_fails_closed_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The public CLI should require explicit cost tracking enablement."""
    _clear_director_env(monkeypatch)
    monkeypatch.setenv("DIRECTOR_USE_NLI", "false")
    monkeypatch.setenv("DIRECTOR_MODE", "general")

    with pytest.raises(SystemExit) as exc_info:
        main(["cost-report"])

    assert exc_info.value.code == 1
    assert "Cost tracking is disabled" in capsys.readouterr().out
