# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — actual complete CI aggregate result contracts
"""Exercise required results for every declared native category and event."""

import pytest

from tools.check_ci_gate import check_ci_results
from tools.ci_workflow_inventory import load_ci_workflow_policy


@pytest.mark.parametrize("event", ["push", "pull_request"])
def test_every_successful_category_is_required(event: str) -> None:
    """All declared native owners together satisfy the aggregate contract."""
    results = {
        category.identity: {"result": "success"}
        for category in load_ci_workflow_policy().categories
    }
    assert check_ci_results(results, event) == []
    for category in tuple(results):
        incomplete = dict(results)
        del incomplete[category]
        assert check_ci_results(incomplete, event)


@pytest.mark.parametrize("outcome", ["failure", "cancelled", "skipped", "pending"])
def test_any_unexpected_terminal_or_missing_result_fails(outcome: str) -> None:
    """Failure and skipped model/test owners cannot turn the required gate green."""
    for category in load_ci_workflow_policy().categories:
        results = {
            owner.identity: {"result": "success"}
            for owner in load_ci_workflow_policy().categories
        }
        results[category.identity]["result"] = outcome
        assert check_ci_results(results, "push")


def test_only_declared_pr_runtime_and_notification_skips_are_allowed() -> None:
    """PRs preserve the original push-only runtime and main-only notification."""
    results = {
        owner.identity: {"result": "success"}
        for owner in load_ci_workflow_policy().categories
    }
    for category in ("runtime", "notify"):
        results[category]["result"] = "skipped"
    assert check_ci_results(results, "pull_request") == []
    assert check_ci_results(results, "push")


def test_public_gate_cli_uses_actual_declared_categories() -> None:
    """The shipping CLI accepts complete results and refuses malformed native input."""
    import json
    import os
    import subprocess
    import sys

    from tools.ci_workflow_inventory import ROOT

    env = dict(os.environ)
    env["CI_RESULTS"] = json.dumps(
        {
            owner.identity: {"result": "success"}
            for owner in load_ci_workflow_policy().categories
        }
    )
    for value, expected in ((env["CI_RESULTS"], 0), ("{}", 1), ("[", 1)):
        env["CI_RESULTS"] = value
        result = subprocess.run(
            [sys.executable, "-m", "tools.check_ci_gate", "--event", "push"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == expected, result.stdout + result.stderr
