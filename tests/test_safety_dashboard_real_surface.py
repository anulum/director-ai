# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - safety dashboard real-surface tests
"""Public production-surface coverage for safety dashboard report builders."""

from __future__ import annotations

import json
import math

import pytest

from director_ai.ui.safety_dashboard import (
    build_observability_operations_markdown,
    build_observability_operations_report,
    build_safety_dashboard,
    build_trust_console_report,
)
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


def _line(payload: dict[str, object]) -> str:
    """Return one JSONL row for public dashboard parser calls."""
    return json.dumps(payload, sort_keys=True) + "\n"


def test_safety_dashboard_unit_guard_has_real_surface_companion() -> None:
    """The safety dashboard unit guard should declare this companion."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_safety_dashboard.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_safety_dashboard_real_surface.py" in category


@pytest.mark.parametrize(
    "threshold",
    (math.nan, math.inf, -math.inf, -0.01, 1.01),
)
def test_public_dashboard_rejects_invalid_halt_threshold(threshold: float) -> None:
    """Operator alert thresholds must fail closed before summary generation."""
    with pytest.raises(
        ValueError,
        match="halt_alert_threshold must be finite and in \\[0, 1\\]",
    ):
        build_safety_dashboard(
            _line({"tenant_id": "tenant-a", "policy_decision": "halt"}),
            halt_alert_threshold=threshold,
        )


@pytest.mark.parametrize(
    "threshold",
    (math.nan, math.inf, -math.inf, -0.01, 1.01),
)
def test_public_trust_console_rejects_invalid_false_positive_threshold(
    threshold: float,
) -> None:
    """Trust Console alert thresholds must reject non-rate values."""
    with pytest.raises(
        ValueError,
        match="false_positive_alert_threshold must be finite and in \\[0, 1\\]",
    ):
        build_trust_console_report(
            _line({"tenant_id": "tenant-a", "policy_decision": "halt"}),
            false_positive_alert_threshold=threshold,
        )


@pytest.mark.parametrize(
    "threshold",
    (math.nan, math.inf, -math.inf, -0.01, 1.01),
)
def test_public_operations_report_rejects_invalid_drift_threshold(
    threshold: float,
) -> None:
    """Operations drift thresholds must be finite rates."""
    with pytest.raises(
        ValueError,
        match="drift_alert_threshold must be finite and in \\[0, 1\\]",
    ):
        build_observability_operations_report(
            _line({"tenant_id": "tenant-a", "policy_decision": "allow"}),
            drift_alert_threshold=threshold,
        )


@pytest.mark.parametrize("window_size", (0, -1))
def test_public_operations_report_rejects_non_positive_drift_window(
    window_size: int,
) -> None:
    """Drift windows must contain at least one event per side."""
    with pytest.raises(ValueError, match="min_drift_window_events must be >= 1"):
        build_observability_operations_report(
            _line({"tenant_id": "tenant-a", "policy_decision": "halt"}),
            min_drift_window_events=window_size,
        )


def test_public_operations_markdown_remains_tenant_safe() -> None:
    """Rendered operations markdown should expose only tenant-safe references."""
    markdown = build_observability_operations_markdown(
        _line(
            {
                "tenant_id": "tenant-a",
                "policy_decision": "halt",
                "trace_attribution": {"fact_source": "kb://policy-v7"},
                "prompt": "raw user prompt must not render",
                "response": "raw model output must not render",
                "customer_email": "jane@example.com",
            },
        ),
    )

    assert "kb://policy-v7" in markdown
    assert "raw user prompt" not in markdown
    assert "raw model output" not in markdown
    assert "jane@example.com" not in markdown
