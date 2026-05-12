# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Competitor AggreFact suite tests

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

from competitor_aggrefact_suite import (  # noqa: E402
    GalileoBackend,
    MockBackend,
    run_suite,
)

_DATASET = (
    {
        "doc": "Water boils at 100 C at sea level.",
        "claim": "Water boils at 100 C at sea level.",
        "label": 1,
        "dataset": "unit",
    },
    {
        "doc": "Earth has one moon.",
        "claim": "Earth has three moons.",
        "label": 0,
        "dataset": "unit",
    },
)


def test_mock_backend_results_are_marked_test_only(tmp_path):
    output = tmp_path / "mock.json"

    payload = run_suite(MockBackend(), dataset=_DATASET, output=output)
    written = json.loads(output.read_text(encoding="utf-8"))

    assert payload["model"] == "mock:test-only"
    assert payload["benchmark_evidence"] is False
    assert payload["provenance"]["test_only"] is True
    assert payload["provenance"]["credentialed_real_api"] is False
    assert "not valid benchmark evidence" in payload["provenance"]["warning"]
    assert written["provenance"] == payload["provenance"]


def test_unimplemented_real_backend_produces_no_metrics():
    with pytest.raises(NotImplementedError):
        run_suite(GalileoBackend(api_key="dummy"), dataset=_DATASET)
