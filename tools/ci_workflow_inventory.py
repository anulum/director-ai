# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — executable CI ownership inventory
"""Resolve real CI jobs through a versioned, validated ownership policy."""

from __future__ import annotations

import csv
import hashlib
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = Path("tools/ci_workflow_policy.toml")


def object_mapping(value: object) -> dict[str, object]:
    """Validate a decoded mapping without discarding unknown value types.

    Parameters
    ----------
    value : object
        Decoded JSON, TOML or YAML mapping.

    Returns
    -------
    dict[str, object]
        Mapping with string keys and original values.

    Raises
    ------
    ValueError
        If the value is not a mapping with string keys.
    """
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError("Expected a mapping with string keys")
    return {k: v for k, v in value.items()}


def string_tuple(value: object) -> tuple[str, ...]:
    """Validate an ordered list of unique nonempty strings.

    Parameters
    ----------
    value : object
        Decoded policy list.

    Returns
    -------
    tuple[str, ...]
        Original list order retained as immutable strings.

    Raises
    ------
    ValueError
        If values are empty, duplicated or not strings.
    """
    if not isinstance(value, list) or any(
        not isinstance(v, str) or not v for v in value
    ):
        raise ValueError("Expected a list of nonempty strings")
    result = tuple(v for v in value if isinstance(v, str))
    if len(set(result)) != len(result):
        raise ValueError("Duplicate policy strings")
    return result


def positive_integer(value: object) -> int:
    """Validate a strictly positive integer, refusing Boolean values.

    Parameters
    ----------
    value : object
        Decoded count or size limit.

    Returns
    -------
    int
        Positive count.

    Raises
    ------
    ValueError
        If the value is not a positive integer.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("Expected a positive integer")
    return value


@dataclass(frozen=True)
class WorkflowCategory:
    """Own one cohesive executable workflow and its caller dependencies.

    Attributes
    ----------
    identity, workflow : str
        Coordinator caller ID and repository-relative reusable workflow path.
    jobs, caller_needs, secrets, skip_events : tuple[str, ...]
        Exclusive jobs, caller dependencies, intended secrets and events where
        the category's explicit condition permits skipping.
    condition : str
        Exact caller condition, or empty when unconditional.
    """

    identity: str
    workflow: str
    jobs: tuple[str, ...]
    caller_needs: tuple[str, ...]
    secrets: tuple[str, ...]
    skip_events: tuple[str, ...]
    condition: str


@dataclass(frozen=True)
class WorkflowPolicy:
    """Describe category ownership, required checks and complete test inputs.

    Attributes
    ----------
    categories : tuple[WorkflowCategory, ...]
        Reusable owners in aggregate order.
    jobs, checks : tuple[str, ...]
        Executable job order and externally required compatibility names.
    dependencies : dict[str, tuple[str, ...]]
        Original logical job edges, including the new partition aggregate.
    limits : dict[str, int]
        Coordinator and reusable byte, line and category limits.
    partitions, rows_per_task : int
        Number of disjoint test partitions and original HaluEval task rows.
    dataset_sha256 : dict[str, str]
        Immutable SHA-256 values of the three public parquet inputs.
    """

    categories: tuple[WorkflowCategory, ...]
    jobs: tuple[str, ...]
    checks: tuple[str, ...]
    dependencies: dict[str, tuple[str, ...]]
    limits: dict[str, int]
    partitions: int
    rows_per_task: int
    dataset_sha256: dict[str, str]


def load_ci_workflow_policy(root: Path = ROOT) -> WorkflowPolicy:
    """Read a selected checkout's ownership and test partition contract.

    Parameters
    ----------
    root : Path
        Repository root containing the versioned TOML policy.

    Returns
    -------
    WorkflowPolicy
        Validated ownership, complete dataset and size contracts.

    Raises
    ------
    ValueError, KeyError
        If required policy fields or ownership are malformed.
    OSError
        If the policy cannot be read.
    """
    data = object_mapping(
        tomllib.loads((root / POLICY_PATH).read_text(encoding="utf-8"))
    )
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("Unsupported CI ownership policy version")
    categories_raw = data["categories"]
    if not isinstance(categories_raw, list) or not categories_raw:
        raise ValueError("CI categories must be a nonempty list")
    categories = []
    for raw in categories_raw:
        item = object_mapping(raw)
        identity, workflow, condition = item["id"], item["workflow"], item["condition"]
        if not isinstance(identity, str) or not re.fullmatch(
            r"[a-z][a-z0-9-]*", identity
        ):
            raise ValueError("Invalid category identifier")
        if not isinstance(workflow, str) or not re.fullmatch(
            r"\.github/workflows/ci-[a-z-]+\.yml", workflow
        ):
            raise ValueError("Invalid repository-local workflow path")
        if not isinstance(condition, str):
            raise ValueError("Category condition must be a string")
        categories.append(
            WorkflowCategory(
                identity,
                workflow,
                string_tuple(item["jobs"]),
                string_tuple(item["caller_needs"]),
                string_tuple(item["secrets"]),
                string_tuple(item["skip_events"]),
                condition,
            )
        )
    jobs, checks = (
        string_tuple(data["job_order"]),
        string_tuple(data["required_checks"]),
    )
    identities = [category.identity for category in categories]
    workflows = [category.workflow for category in categories]
    owned = [job for category in categories for job in category.jobs]
    if len(set(identities)) != len(identities) or len(set(workflows)) != len(workflows):
        raise ValueError("Duplicate category identity or workflow")
    if len(set(owned)) != len(owned) or set(owned) != set(jobs):
        raise ValueError("Every CI executable job must have exactly one owner")
    dependencies = {
        job: string_tuple(value)
        for job, value in object_mapping(data["dependencies"]).items()
    }
    if set(dependencies) != set(jobs):
        raise ValueError("Every executable job needs a declared dependency contract")
    limits = {
        key: positive_integer(value)
        for key, value in object_mapping(data["limits"]).items()
    }
    test = object_mapping(data["tests"])
    partitions, rows = (
        positive_integer(test["partitions"]),
        positive_integer(test["rows_per_task"]),
    )
    if rows % partitions:
        raise ValueError("HaluEval ranges must divide the complete task row count")
    hashes = object_mapping(test["dataset_sha256"])
    if set(hashes) != {"qa", "summarization", "dialogue"}:
        raise ValueError("The complete three-task HaluEval input set is required")
    dataset_sha256 = {}
    for task, digest in hashes.items():
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Dataset SHA-256 must be an immutable hexadecimal digest")
        dataset_sha256[task] = digest
    return WorkflowPolicy(
        tuple(categories),
        jobs,
        checks,
        dependencies,
        limits,
        partitions,
        rows,
        dataset_sha256,
    )


def ci_workflow_paths(root: Path = ROOT) -> tuple[Path, ...]:
    """Resolve the coordinator followed by its exclusively owned workflows.

    Parameters
    ----------
    root : Path
        Selected repository checkout.

    Returns
    -------
    tuple[Path, ...]
        Same-revision executable sources in policy order.
    """
    policy = load_ci_workflow_policy(root)
    return (
        root / ".github/workflows/ci.yml",
        *(root / c.workflow for c in policy.categories),
    )


def workflow_path_for_job(job: str, root: Path = ROOT) -> Path:
    """Resolve an executable job to its declared, existing owner.

    Parameters
    ----------
    job : str
        Exact executable job ID.
    root : Path
        Selected repository checkout.

    Returns
    -------
    Path
        Existing reusable workflow containing the executable job.

    Raises
    ------
    ValueError
        If the declared job is absent from its workflow.
    KeyError
        If no owner declares the job.
    """
    for category in load_ci_workflow_policy(root).categories:
        if job in category.jobs:
            path = root / category.workflow
            if job not in read_job_blocks(path):
                raise ValueError("Declared executable job is absent from its owner")
            return path
    raise KeyError(job)


def read_job_blocks(path: Path) -> dict[str, str]:
    """Read real top-level job blocks for compatibility and source audits.

    Parameters
    ----------
    path : Path
        Executable workflow source.

    Returns
    -------
    dict[str, str]
        Raw blocks keyed by job ID, preserving comments and step bodies.

    Raises
    ------
    ValueError
        If job IDs are repeated or a workflow has no jobs section.
    """
    text = path.read_text(encoding="utf-8")
    _, separator, body = text.partition("jobs:\n")
    if not separator:
        raise ValueError("Workflow has no jobs section")
    matches = list(re.finditer(r"^  ([A-Za-z0-9_-]+):\s*$", body, re.MULTILINE))
    result = {}
    for i, match in enumerate(matches):
        key = match.group(1)
        if key in result:
            raise ValueError("Duplicate executable job ID")
        result[key] = body[
            match.start() : matches[i + 1].start()
            if i + 1 < len(matches)
            else len(body)
        ]
    return result


def read_ci_workflow_source(root: Path = ROOT) -> str:
    """Read distributed executable jobs in their declared logical order.

    Parameters
    ----------
    root : Path
        Selected repository checkout.

    Returns
    -------
    str
        Actual executable source assembled for existing contract readers.

    Raises
    ------
    ValueError
        If executable ownership differs from the versioned inventory.
    """
    policy = load_ci_workflow_policy(root)
    blocks = {}
    for category in policy.categories:
        owned = read_job_blocks(root / category.workflow)
        if tuple(owned) != category.jobs:
            raise ValueError("Executable workflow differs from the ownership inventory")
        blocks.update(owned)
    return "jobs:\n" + "\n".join(blocks[job] for job in policy.jobs)


def expected_halueval_inputs(root: Path = ROOT) -> dict[tuple[str, int, bool], str]:
    """Read the exact first-200-row, two-label public input digest manifest.

    Parameters
    ----------
    root : Path
        Checkout containing the immutable TSV and matching policy digest.

    Returns
    -------
    dict[tuple[str, int, bool], str]
        Original task, row and label mapped to its actual input SHA-256.

    Raises
    ------
    ValueError
        If bytes, rows, labels, hashes or complete input identities differ
        from the versioned contract.
    """
    policy = load_ci_workflow_policy(root)
    raw = (root / "benchmarks/halueval_ci_inputs.tsv").read_bytes()
    data = object_mapping(
        tomllib.loads((root / POLICY_PATH).read_text(encoding="utf-8"))
    )
    tests = object_mapping(data["tests"])
    if hashlib.sha256(raw).hexdigest() != tests["input_manifest_sha256"]:
        raise ValueError(
            "HaluEval input manifest bytes differ from the declared digest"
        )
    rows = csv.reader(
        (line for line in raw.decode("utf-8").splitlines() if not line.startswith("#")),
        delimiter="\t",
    )
    result = {}
    for row in rows:
        if len(row) != 4:
            raise ValueError("HaluEval input row must have four TSV fields")
        task, index_text, label, digest = row
        if not index_text.isdecimal() or label not in {"false", "true"}:
            raise ValueError("HaluEval row index or label is invalid")
        key = (task, int(index_text), label == "true")
        if key in result or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Duplicate HaluEval identity or invalid input digest")
        result[key] = digest
    required = {
        (task, i, label)
        for task in policy.dataset_sha256
        for i in range(policy.rows_per_task)
        for label in (False, True)
    }
    if result.keys() != required:
        raise ValueError(
            "HaluEval manifest does not cover every original required input"
        )
    return result
