# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — native workflow ownership and graph refusals
"""Audit actual repository workflows and deliberate drift in their real bytes."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.audit_ci_workflow_modularity import (
    audit_ci_workflow_modularity,
    read_workflow,
)
from tools.ci_workflow_inventory import POLICY_PATH, ROOT, ci_workflow_paths


def test_actual_distributed_workflows_have_complete_ownership() -> None:
    """Audit the shipping coordinator, every actual owner and all native inputs."""
    assert audit_ci_workflow_modularity() == []


@pytest.mark.parametrize(
    "path,before,after",
    [
        (".github/workflows/ci.yml", "needs: [static, lite", "needs: [lite"),
        (
            ".github/workflows/ci-python.yml",
            "coverage report --fail-under=97",
            "coverage report --fail-under=96",
        ),
        (".github/workflows/ci-extras.yml", "test-extras:", "unowned-extra:"),
        (".github/workflows/ci-runtime.yml", "contents: read", "contents: write"),
        (
            ".github/workflows/ci-static.yml",
            "value: ${{ jobs.static-results.outputs.lint }}",
            "value: success",
        ),
        (
            ".github/workflows/ci-security.yml",
            "value: ${{ jobs.security-results.outputs.sast }}",
            "value: success",
        ),
        (".github/workflows/ci.yml", "name: CI\n", "name: CI\nrun-name: other\n"),
        (".github/workflows/ci.yml", "branches: [main]", "branches: [other]"),
        (".github/workflows/ci.yml", "contents: read", "contents: write"),
        (
            ".github/workflows/ci.yml",
            "cancel-in-progress: true",
            "cancel-in-progress: false",
        ),
        (
            ".github/workflows/ci.yml",
            "uses: ./.github/workflows/ci-lite.yml",
            "uses: other/repo/.github/workflows/ci.yml@main",
        ),
        (".github/workflows/ci.yml", "  lite:\n", "  lite:\n    needs: [static]\n"),
        (".github/workflows/ci.yml", "  lite:\n", "  lite:\n    if: false\n"),
        (".github/workflows/ci.yml", "  lite:\n", "  lite:\n    secrets: inherit\n"),
        (
            ".github/workflows/ci.yml",
            "  lite:\n",
            "  lite:\n    with: {unexpected: value}\n",
        ),
        (
            ".github/workflows/ci-lite.yml",
            "name: CI lite\n",
            "name: CI lite\nrun-name: other\n",
        ),
        (
            ".github/workflows/ci-lite.yml",
            "  workflow_call:\n",
            "  workflow_call:\n  push:\n",
        ),
        (
            ".github/workflows/ci-lite.yml",
            "  workflow_call:\n",
            "  workflow_call:\n    secrets: {TOKEN: {required: true}}\n",
        ),
        (
            ".github/workflows/ci-lite.yml",
            "  workflow_call:\n",
            "  workflow_call:\n    inputs: {unexpected: {type: string}}\n",
        ),
        (
            ".github/workflows/ci-lite.yml",
            "  workflow_call:\n",
            "  workflow_call:\n    unexpected: value\n",
        ),
        (
            ".github/workflows/ci-lite.yml",
            'CARGO_NET_RETRY: "10"',
            'CARGO_NET_RETRY: "1"',
        ),
        (
            ".github/workflows/ci-lite.yml",
            "    steps:\n",
            "    needs: [lint]\n    steps:\n",
        ),
        (
            ".github/workflows/ci-lite.yml",
            "    steps:\n",
            "    uses: invalid\n    steps:\n",
        ),
        (
            ".github/workflows/ci-lite.yml",
            "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
            "actions/checkout@main",
        ),
        (
            ".github/workflows/ci.yml",
            'check: ["Test (Python 3.11)", "Lint & Format"]',
            'check: ["Other"]',
        ),
        (
            ".github/workflows/ci.yml",
            "CI_EVENT: ${{ github.event_name }}",
            "CI_EVENT: push",
        ),
        (
            ".github/workflows/ci-python.yml",
            "partition: [0, 1, 2, 3, 4, 5, 6, 7]",
            "partition: [0, 1, 2]",
        ),
        ("tools/ci_workflow_policy.toml", '"sbom" = ["test"]', '"sbom" = ["unknown"]'),
        (
            "tools/ci_workflow_policy.toml",
            'caller_needs = ["python"]',
            "caller_needs = []",
        ),
        ("tools/ci_workflow_policy.toml", "skip_events = []", 'skip_events = ["push"]'),
        ("tools/ci_workflow_policy.toml", "reusable_count = 20", "reusable_count = 50"),
        (
            "tools/ci_workflow_policy.toml",
            "reusable_lines = 1000",
            "reusable_lines = 1",
        ),
        (
            "tools/ci_workflow_policy.toml",
            "coordinator_bytes = 20000",
            "coordinator_bytes = 1",
        ),
    ],
)
def test_real_native_workflow_drift_is_refused(
    tmp_path: Path,
    path: str,
    before: str,
    after: str,
) -> None:
    """Refuse omitted categories, weakened coverage, unowned jobs and privileges.

    Parameters
    ----------
    tmp_path : Path
        Isolated copy of the actual native workflow surfaces.
    path, before, after : str
        Exact real YAML bytes and the incompatible alteration under test.
    """
    for source in (
        *ci_workflow_paths(),
        ROOT / POLICY_PATH,
        ROOT / "benchmarks/halueval_ci_inputs.tsv",
    ):
        target = tmp_path / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    target = tmp_path / path
    original = target.read_text()
    assert before in original
    target.write_text(original.replace(before, after, 1))
    try:
        violations = audit_ci_workflow_modularity(tmp_path)
    except (ValueError, KeyError):
        return
    assert violations


def test_duplicate_yaml_mapping_members_are_refused(tmp_path: Path) -> None:
    """A repeated gate or job cannot hide behind the YAML parser's last value."""
    path = tmp_path / "duplicate.yml"
    path.write_text("jobs:\n  gate: {}\n  gate: {}\n")
    with pytest.raises(ValueError, match="repeated"):
        read_workflow(path)


def test_an_unowned_actual_ci_workflow_is_refused(tmp_path: Path) -> None:
    """An extra executable CI workflow cannot escape the declared owner inventory."""
    for source in (
        *ci_workflow_paths(),
        ROOT / POLICY_PATH,
        ROOT / "benchmarks/halueval_ci_inputs.tsv",
    ):
        target = tmp_path / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(
        tmp_path / ".github/workflows/ci-lite.yml",
        tmp_path / ".github/workflows/ci-undeclared.yml",
    )
    assert "CI workflow files differ from the exclusive ownership inventory" in (
        audit_ci_workflow_modularity(tmp_path)
    )


@pytest.mark.parametrize("source", ["jobs: [", "[not, a, mapping]"])
def test_malformed_native_yaml_is_refused(tmp_path: Path, source: str) -> None:
    """The public reader refuses invalid syntax and nonmapping native documents."""
    path = tmp_path / "workflow.yml"
    path.write_text(source)
    with pytest.raises(ValueError):
        read_workflow(path)


@pytest.mark.parametrize("valid", [True, False])
def test_public_workflow_audit_cli_uses_the_selected_checkout(
    tmp_path: Path, valid: bool
) -> None:
    """Run the real CLI against the shipping checkout or a missing contract."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.audit_ci_workflow_modularity",
            "--root",
            str(ROOT if valid else tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == (0 if valid else 1), result.stdout + result.stderr
    assert ("contracts verified" if valid else "refused") in result.stdout
