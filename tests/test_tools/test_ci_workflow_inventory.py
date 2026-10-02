# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — real distributed workflow inventory contracts
"""Read all actual CI owners and immutable public benchmark identities."""

from pathlib import Path

import pytest

from tools.ci_workflow_inventory import (
    ROOT,
    ci_workflow_paths,
    expected_halueval_inputs,
    load_ci_workflow_policy,
    read_ci_workflow_source,
    workflow_path_for_job,
)


def test_every_declared_job_resolves_to_its_real_owner() -> None:
    """Resolve the real YAML owner of every executable CI job."""
    policy = load_ci_workflow_policy()
    paths = ci_workflow_paths()
    assert len(paths) == len(policy.categories) + 1
    for category in policy.categories:
        for job in category.jobs:
            assert workflow_path_for_job(job) == ROOT / category.workflow
    source = read_ci_workflow_source()
    assert "pytest tests/" in source
    assert "--cov-fail-under=0" in source  # Original optional-extra denominator.
    assert "coverage report --fail-under=97" in source
    with pytest.raises(KeyError):
        workflow_path_for_job("undeclared-job")


def test_complete_original_public_input_manifest() -> None:
    """Require the original 200 rows and both labels for all three tasks."""
    inputs = expected_halueval_inputs()
    assert len(inputs) == 1200
    assert all(len(digest) == 64 for digest in inputs.values())
    assert all(
        (task, i, label) in inputs
        for task in ("qa", "summarization", "dialogue")
        for i in range(200)
        for label in (False, True)
    )


def test_absent_checkout_policy_is_refused(tmp_path: Path) -> None:
    """An absent native policy cannot become an empty successful inventory."""
    with pytest.raises(OSError):
        load_ci_workflow_policy(tmp_path)


def _native_checkout(tmp_path: Path) -> Path:
    """Copy actual shipping contracts for deliberate native refusal tests."""
    import shutil

    from tools.ci_workflow_inventory import POLICY_PATH

    for source in (
        *ci_workflow_paths(),
        ROOT / POLICY_PATH,
        ROOT / "benchmarks/halueval_ci_inputs.tsv",
    ):
        target = tmp_path / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return tmp_path


@pytest.mark.parametrize(
    "before,after",
    [
        ("schema_version = 1", "schema_version = 2"),
        ("schema_version = 1", "schema_version = true"),
        ('id = "static"', 'id = "INVALID"'),
        (
            'workflow = ".github/workflows/ci-static.yml"',
            'workflow = "../ci-static.yml"',
        ),
        ('condition = ""', "condition = false"),
        ('job_order = ["lint"', 'job_order = ["lint", "lint"'),
        ('jobs = ["director-lite"]', 'jobs = ["lint"]'),
        ('"lint" = []', '"unowned" = []'),
        ("partitions = 8", "partitions = true"),
        ("partitions = 8", "partitions = 3"),
        (
            'qa = "1b7118e6dcc3881e5fc87b77b48d9334f0a24d51073ea9dc05baf5a74814580f"',
            'qa = "not-a-digest"',
        ),
        (
            'qa = "1b7118e6dcc3881e5fc87b77b48d9334f0a24d51073ea9dc05baf5a74814580f"',
            'unknown = "1b7118e6dcc3881e5fc87b77b48d9334f0a24d51073ea9dc05baf5a74814580f"',
        ),
        ('job_order = ["lint"', "job_order = [false"),
        (
            'workflow = ".github/workflows/ci-lite.yml"',
            'workflow = ".github/workflows/ci-static.yml"',
        ),
    ],
)
def test_real_policy_contract_mutations_are_refused(
    tmp_path: Path,
    before: str,
    after: str,
) -> None:
    """Refuse malformed ownership, dependencies, range counts and input identities."""
    from tools.ci_workflow_inventory import POLICY_PATH

    root = _native_checkout(tmp_path)
    path = root / POLICY_PATH
    original = path.read_text()
    assert before in original
    path.write_text(original.replace(before, after, 1))
    with pytest.raises((ValueError, KeyError)):
        load_ci_workflow_policy(root)


@pytest.mark.parametrize(
    "mutation", ["digest", "fields", "label", "duplicate", "incomplete", "hash"]
)
def test_actual_input_manifest_drift_is_refused(tmp_path: Path, mutation: str) -> None:
    """Even a freshly declared checksum cannot authorize incomplete original inputs."""
    import hashlib

    from tools.ci_workflow_inventory import POLICY_PATH

    root = _native_checkout(tmp_path)
    path = root / "benchmarks/halueval_ci_inputs.tsv"
    original = path.read_bytes()
    lines = original.decode().splitlines()
    if mutation == "fields":
        lines[7] = "\t".join(lines[7].split("\t")[:3])
    elif mutation == "label":
        lines[7] = lines[7].replace("\tfalse\t", "\tunknown\t")
    elif mutation == "duplicate":
        lines.append(lines[7])
    elif mutation == "incomplete":
        lines.pop()
    elif mutation == "hash":
        lines[7] = "\t".join([*lines[7].split("\t")[:3], "invalid-sha"])
    else:
        lines.append("# changed original bytes")
    altered = ("\n".join(lines) + "\n").encode()
    path.write_bytes(altered)
    if mutation != "digest":
        policy = root / POLICY_PATH
        policy.write_text(
            policy.read_text().replace(
                hashlib.sha256(original).hexdigest(),
                hashlib.sha256(altered).hexdigest(),
            )
        )
    with pytest.raises(ValueError):
        expected_halueval_inputs(root)


@pytest.mark.parametrize("mutation", ["absent", "duplicate", "unowned"])
def test_compatibility_reader_refuses_real_owner_drift(
    tmp_path: Path, mutation: str
) -> None:
    """Existing contract readers cannot silently fall back to obsolete YAML bodies."""
    root = _native_checkout(tmp_path)
    path = workflow_path_for_job("director-lite", root)
    text = path.read_text()
    if mutation == "absent":
        text = text.replace("jobs:\n", "removed:\n", 1)
    elif mutation == "duplicate":
        text += "\n  director-lite:\n    runs-on: ubuntu-latest\n"
    else:
        text = text.replace("  director-lite:\n", "  wrong-owner:\n", 1)
    path.write_text(text)
    with pytest.raises(ValueError):
        read_ci_workflow_source(root)
    if mutation == "unowned":
        with pytest.raises(ValueError):
            workflow_path_for_job("director-lite", root)


def test_an_empty_category_inventory_is_refused(tmp_path: Path) -> None:
    """A syntactically valid policy must still declare its actual executable owners."""
    from tools.ci_workflow_inventory import POLICY_PATH

    path = tmp_path / POLICY_PATH
    path.parent.mkdir(parents=True)
    path.write_text("schema_version = 1\ncategories = []\n")
    with pytest.raises(ValueError, match="nonempty list"):
        load_ci_workflow_policy(tmp_path)
