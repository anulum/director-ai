# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — committed eval-gate dataset regression test

"""Guard the committed PR eval-gate dataset and the offline heuristic gate.

The ``Eval Gate`` workflow runs ``director-ai ci-gate --profile rules`` over
``benchmarks/eval_gate_cases.jsonl`` with the offline heuristic scorer. This test
pins that contract: the dataset loads, is balanced, and the rules-profile gate
clears the same thresholds the workflow enforces — so a regression in either the
dataset or the heuristic scorer is caught in the main test suite too.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from director_ai.ci_gate import GateThresholds, load_cases, run_eval_gate
from director_ai.core.config import DirectorConfig

_DATASET = Path(__file__).resolve().parents[1] / "benchmarks" / "eval_gate_cases.jsonl"


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def test_dataset_loads_and_is_balanced():
    cases = load_cases(_DATASET)
    assert len(cases) >= 8
    approve = [c for c in cases if c.expected == "approve"]
    reject = [c for c in cases if c.expected == "reject"]
    assert approve and reject  # both classes present
    assert abs(len(approve) - len(reject)) <= 1  # roughly balanced


def test_rules_profile_gate_passes_workflow_thresholds():
    cases = load_cases(_DATASET)
    scorer = DirectorConfig.from_profile("rules").build_scorer()
    report = run_eval_gate(
        cases,
        scorer,
        GateThresholds(min_accuracy=0.9, min_catch_rate=0.9, max_false_halt_rate=0.1),
    )
    assert report.passed, report.failures
    assert os.environ["HF_HUB_OFFLINE"] == "1"
