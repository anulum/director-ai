# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - config wizard real-surface tests
"""Real-surface tests for config wizard CLI and trace detail edges."""

from __future__ import annotations

import json

import pytest
import yaml

import director_ai.ui.config_wizard as config_wizard_module
from director_ai.ui.config_wizard import launch_cli
from director_ai.ui.trace_explorer import build_trace_explorer


def test_trace_explorer_handles_sparse_attribution_and_counterfactual() -> None:
    """Trace detail rendering should tolerate sparse diagnostic payloads."""
    payload = {
        "events": [
            {
                "event_type": "halt",
                "trace_attribution": {"source": "manual-review"},
                "counterfactual_diagnostic": {"status": "not_applicable"},
            },
        ],
    }

    _summary, rows, detail = build_trace_explorer(json.dumps(payload))

    assert rows[0][7] == ""
    assert detail["trace_attribution"] == {"source": "manual-review"}
    assert detail["counterfactual"] == {"status": "not_applicable"}


def test_launch_cli_coerces_integer_fields_through_public_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI prompts should coerce integer defaults to integer YAML values."""
    monkeypatch.setattr(
        config_wizard_module,
        "CLI_KEY_FIELDS",
        (("history_window", "History window", 5),),
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "7")

    result = launch_cli()
    parsed = yaml.safe_load(
        "\n".join(
            line for line in result.splitlines() if line and not line.startswith("#")
        ),
    )

    assert parsed["history_window"] == 7
