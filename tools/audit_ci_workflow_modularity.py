# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — fail-closed executable workflow ownership audit
"""Validate real YAML, exclusive ownership, logical edges and aggregate gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tomllib
from collections.abc import Hashable
from pathlib import Path

import yaml
from yaml.nodes import MappingNode

from tools.ci_workflow_inventory import (
    POLICY_PATH,
    ROOT,
    WorkflowCategory,
    WorkflowPolicy,
    ci_workflow_paths,
    expected_halueval_inputs,
    load_ci_workflow_policy,
    object_mapping,
    string_tuple,
)


class UniqueLoader(yaml.BaseLoader):
    """Parse workflow scalars literally and refuse repeated mapping members."""

    def construct_mapping(
        self, node: MappingNode, deep: bool = False
    ) -> dict[Hashable, object]:
        """Validate native YAML mapping keys before constructing their values."""
        result: dict[Hashable, object] = {}
        for key_node, value_node in node.value:
            key: object = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise ValueError("Invalid or repeated workflow YAML mapping member")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def read_workflow(path: Path) -> dict[str, object]:
    """Read actual workflow YAML with duplicate-member refusal.

    Parameters
    ----------
    path : Path
        Existing native workflow source.

    Returns
    -------
    dict[str, object]
        Original YAML structure with literal scalar strings.

    Raises
    ------
    ValueError, OSError
        If YAML, mapping keys or the source path is invalid.
    """
    try:
        value: object = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueLoader)
    except yaml.YAMLError as exc:
        raise ValueError("Malformed native workflow YAML") from exc
    return object_mapping(value)


def _audit_coordinator(
    coordinator: dict[str, object], policy: WorkflowPolicy, violations: list[str]
) -> dict[str, object]:
    """Check the actual event, permission, concurrency and category call surface."""
    if set(coordinator) != {"name", "on", "permissions", "concurrency", "jobs"}:
        violations.append(
            "Coordinator contains undeclared executable or policy sections"
        )
    events = object_mapping(coordinator["on"])
    if events != {
        "push": {"branches": ["main"]},
        "pull_request": {"branches": ["main"]},
    }:
        violations.append("CI event or main-branch contract differs")
    if coordinator["permissions"] != {"contents": "read"}:
        violations.append("Coordinator permissions exceed the read-only contract")
    if coordinator["concurrency"] != {
        "group": "${{ github.workflow }}-${{ github.ref }}",
        "cancel-in-progress": "true",
    }:
        violations.append("Coordinator concurrency contract differs")
    callers = object_mapping(coordinator["jobs"])
    identities = tuple(category.identity for category in policy.categories)
    if tuple(callers) != (*identities, "ci-gate"):
        violations.append(
            "Coordinator does not own exactly one call per category and one gate"
        )
    return callers


def _audit_caller(
    caller: dict[str, object], category: WorkflowCategory, violations: list[str]
) -> None:
    """Check one same-revision call and its explicit condition, dependencies and secrets."""
    allowed = {"uses", "needs", "if", "secrets", "with"}
    if set(caller) - allowed or caller["uses"] != "./" + category.workflow:
        violations.append(f"Invalid same-revision reusable call: {category.identity}")
    if string_tuple(caller.get("needs", [])) != category.caller_needs:
        violations.append(f"Caller dependencies differ: {category.identity}")
    if caller.get("if", "") != category.condition:
        violations.append(f"Caller condition differs: {category.identity}")
    secret_bindings = object_mapping(caller.get("secrets", {}))
    expected_bindings = {
        secret: "${{ secrets." + secret + " }}" for secret in category.secrets
    }
    if secret_bindings != expected_bindings:
        violations.append(
            f"Undeclared or incorrect secret forwarding: {category.identity}"
        )
    if bool(caller.get("with")) != (category.identity == "notify"):
        violations.append(f"Undeclared reusable inputs: {category.identity}")


def _audit_reusable_surface(
    workflow: dict[str, object], category: WorkflowCategory, violations: list[str]
) -> dict[str, object]:
    """Check one actual reusable call surface and exclusive executable ownership."""
    if set(workflow) != {"name", "on", "permissions", "env", "jobs"}:
        violations.append(f"Reusable policy sections differ: {category.identity}")
    call = object_mapping(object_mapping(workflow["on"])["workflow_call"] or {})
    if set(object_mapping(workflow["on"])) != {"workflow_call"}:
        violations.append(f"Reusable has undeclared triggers: {category.identity}")
    declared_secrets = object_mapping(call.get("secrets", {}))
    if tuple(declared_secrets) != category.secrets or any(
        v != {"required": "false"} for v in declared_secrets.values()
    ):
        violations.append(f"Reusable secret surface differs: {category.identity}")
    inputs = object_mapping(call.get("inputs", {}))
    expected_inputs = (
        {"results": {"required": "true", "type": "string"}}
        if category.identity == "notify"
        else {}
    )
    if inputs != expected_inputs:
        violations.append(f"Reusable input surface differs: {category.identity}")
    result_names = {
        "static": ("lint", "typecheck", "reuse"),
        "security": ("security", "sast"),
    }.get(category.identity, ())
    expected_outputs = {
        name: {
            "value": "${{ jobs."
            + category.identity
            + "-results.outputs."
            + name
            + " }}"
        }
        for name in result_names
    }
    if object_mapping(call.get("outputs", {})) != expected_outputs:
        violations.append(f"Native result output bindings differ: {category.identity}")
    if set(call) - {"inputs", "outputs", "secrets"}:
        violations.append(f"Undeclared reusable call sections: {category.identity}")
    if workflow["permissions"] != {"contents": "read"}:
        violations.append(f"Reusable permissions exceed policy: {category.identity}")
    if workflow["env"] != {
        "FORCE_JAVASCRIPT_ACTIONS_TO_NODE24": "true",
        "CARGO_HTTP_MULTIPLEXING": "false",
        "CARGO_HTTP_TIMEOUT": "60",
        "CARGO_NET_RETRY": "10",
    }:
        violations.append(
            f"Reusable native runtime environment differs: {category.identity}"
        )
    jobs = object_mapping(workflow["jobs"])
    if tuple(jobs) != category.jobs:
        violations.append(f"Executable ownership differs: {category.identity}")
    return jobs


def _audit_job(
    job_id: str,
    raw: object,
    category: WorkflowCategory,
    policy: WorkflowPolicy,
    owners: dict[str, str],
    hashes: dict[str, object],
    violations: list[str],
) -> dict[str, object]:
    """Check a real executable body, immutable actions and original internal edges."""
    job = object_mapping(raw)
    digest = hashlib.sha256(
        json.dumps(job, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    if digest != hashes[job_id]:
        violations.append(
            f"Executable body differs from its versioned contract: {job_id}"
        )
    declared = policy.dependencies[job_id]
    if any(dependency not in owners for dependency in declared):
        raise ValueError("Unknown logical job dependency")
    internal = tuple(
        dependency for dependency in declared if owners[dependency] == category.identity
    )
    if string_tuple(job.get("needs", [])) != internal:
        violations.append(f"Undeclared or missing internal job edge: {job_id}")
    steps = job.get("steps")
    if not isinstance(steps, list) or not steps or "uses" in job:
        violations.append(f"Executable owner lacks native steps: {job_id}")
    else:
        for raw_step in steps:
            step = object_mapping(raw_step)
            uses = step.get("uses")
            if uses is not None and (
                not isinstance(uses, str)
                or not re.fullmatch(r"[\w-]+/[\w/-]+@[0-9a-f]{40}", uses)
            ):
                violations.append(
                    f"Third-party action lacks an immutable full SHA: {job_id}"
                )
    return job


def _audit_gate(
    gate: dict[str, object],
    policy: WorkflowPolicy,
    identities: tuple[str, ...],
    violations: list[str],
) -> None:
    """Require every category and both externally protected native check names."""
    if string_tuple(gate["needs"]) != identities or gate.get("if") != "always()":
        violations.append(
            "Required aggregate does not include every category exactly once"
        )
    strategy = object_mapping(gate["strategy"])
    if string_tuple(object_mapping(strategy["matrix"])["check"]) != policy.checks:
        violations.append("Externally required check names differ")
    steps = gate["steps"]
    if not isinstance(steps, list) or not steps:
        raise ValueError("Required gate must have native steps")
    final_step = object_mapping(steps[-1])
    if (
        final_step.get("env")
        != {
            "CI_RESULTS": "${{ toJSON(needs) }}",
            "CI_EVENT": "${{ github.event_name }}",
        }
        or final_step.get("run") != 'python -m tools.check_ci_gate --event "$CI_EVENT"'
    ):
        violations.append("Required aggregate is not wired to the fail-closed verifier")


def audit_ci_workflow_modularity(root: Path = ROOT) -> list[str]:
    """Audit ownership, preserved logical dependencies and complete-test wiring.

    Parameters
    ----------
    root : Path
        Actual checkout to audit, including its versioned ownership policy.

    Returns
    -------
    list[str]
        Authored violations; an empty list proves all declared contracts.

    Raises
    ------
    ValueError, KeyError, OSError
        If a required policy, input manifest or YAML structure is malformed.
    """
    policy = load_ci_workflow_policy(root)
    data = object_mapping(
        tomllib.loads((root / POLICY_PATH).read_text(encoding="utf-8"))
    )
    hashes = object_mapping(data["job_sha256"])
    if set(hashes) != set(policy.jobs):
        raise ValueError("Every executable job needs an immutable body contract")
    expected_halueval_inputs(root)
    paths = ci_workflow_paths(root)
    actual_paths = set((root / ".github/workflows").glob("ci*.yml"))
    coordinator = read_workflow(paths[0])
    violations: list[str] = []
    if actual_paths != set(paths):
        violations.append(
            "CI workflow files differ from the exclusive ownership inventory"
        )
    digest = hashlib.sha256(
        json.dumps(coordinator, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    if digest != data["coordinator_sha256"]:
        violations.append("Coordinator body differs from its versioned contract")
    callers = _audit_coordinator(coordinator, policy, violations)
    identities = tuple(category.identity for category in policy.categories)
    owners = {
        job: category.identity
        for category in policy.categories
        for job in category.jobs
    }
    native_jobs: dict[str, object] = {}
    if (
        len(policy.categories) > policy.limits["reusable_count"]
        or policy.limits["reusable_count"] >= 50
    ):
        violations.append("Reusable workflow count exceeds repository/provider policy")
    for category in policy.categories:
        _audit_caller(object_mapping(callers[category.identity]), category, violations)
        jobs = _audit_reusable_surface(
            read_workflow(root / category.workflow), category, violations
        )
        for job_id, raw in jobs.items():
            native_jobs[job_id] = _audit_job(
                job_id, raw, category, policy, owners, hashes, violations
            )
        cross = {
            owners[dependency]
            for job in category.jobs
            for dependency in policy.dependencies[job]
            if dependency not in category.jobs
        }
        if cross != set(category.caller_needs):
            violations.append(
                f"Cross-category logical edges differ: {category.identity}"
            )
        allowed_skips = {"runtime": ("pull_request",), "notify": ("pull_request",)}.get(
            category.identity, ()
        )
        if category.skip_events != allowed_skips:
            violations.append(f"Unexpected skip policy: {category.identity}")
    _audit_gate(object_mapping(callers["ci-gate"]), policy, identities, violations)
    worker = object_mapping(native_jobs["test-partitions"])
    matrix = object_mapping(object_mapping(worker["strategy"])["matrix"])
    if (
        matrix
        != {
            "python-version": ["3.11", "3.12", "3.13"],
            "partition": [str(i) for i in range(policy.partitions)],
        }
        or policy.rows_per_task != 200
    ):
        violations.append("Complete original Python/model test matrix differs")
    for i, path in enumerate(paths):
        raw = path.read_bytes()
        prefix = "coordinator" if i == 0 else "reusable"
        if (
            len(raw) > policy.limits[prefix + "_bytes"]
            or len(raw.splitlines()) > policy.limits[prefix + "_lines"]
        ):
            violations.append(f"Workflow exceeds repository size policy: {path.name}")
    return violations


def main() -> int:
    """Audit the selected real checkout through the normal local and hosted CLI.

    Returns
    -------
    int
        Zero for complete ownership and wiring, one for any refusal.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        violations = audit_ci_workflow_modularity(args.root)
    except (ValueError, KeyError, OSError):
        print("CI workflow ownership audit refused malformed native contracts")
        return 1
    for violation in violations:
        print(violation)
    if not violations:
        print("CI workflow ownership, graph and aggregate contracts verified")
    return int(bool(violations))


if __name__ == "__main__":
    raise SystemExit(main())
