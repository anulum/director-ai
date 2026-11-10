# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — real distributed workflow inventory contracts
"""Read all actual CI owners and immutable public benchmark identities."""

import json
from pathlib import Path

import pytest

from tools.audit_ci_workflow_modularity import (
    audit_ci_workflow_modularity,
    read_workflow,
)
from tools.ci_workflow_inventory import (
    ROOT,
    ci_workflow_paths,
    expected_halueval_inputs,
    load_ci_workflow_policy,
    object_mapping,
    read_ci_workflow_source,
    workflow_path_for_job,
)
from tools.quality_ratchet_typescript import NODE_VERSION, TYPESCRIPT_VERSION


def test_native_node_consumers_match_the_executing_locked_profile() -> None:
    """Bind every maintained Node consumer to the actual adapter and compiler lock."""
    for path, job_id in (
        (workflow_path_for_job("lint"), "lint"),
        (workflow_path_for_job("test-partitions"), "test-partitions"),
        (ROOT / ".github/workflows/docs.yml", "typedoc"),
    ):
        job = object_mapping(object_mapping(read_workflow(path)["jobs"])[job_id])
        raw = job["steps"]
        assert isinstance(raw, list)
        steps = [object_mapping(step) for step in raw]
        setups = [
            step
            for step in steps
            if str(step.get("uses", "")).startswith("actions/setup-node@")
        ]
        assert len(setups) == 1
        setup = setups[0]
        assert (
            setup["uses"]
            == "actions/setup-node@949feb2413d6458794dcd2491c4babbbce0c15c1"
        )
        assert "if" not in setup and not setup.get("continue-on-error", False)
        assert object_mapping(setup["with"]) == {
            "node-version": NODE_VERSION,
            "cache": "npm",
            "cache-dependency-path": (
                "packages/vercel-ai/package-lock.json\ngateway/go/bench/package-lock.json\n"
                if job_id == "lint"
                else "packages/vercel-ai/package-lock.json"
            ),
        }
    locked = object_mapping(
        json.loads((ROOT / "packages/vercel-ai/package-lock.json").read_text())
    )
    compiler = object_mapping(
        object_mapping(locked["packages"])["node_modules/typescript"]
    )
    assert compiler["version"] == TYPESCRIPT_VERSION


def test_python_partitions_install_the_native_compiler_before_all_tests() -> None:
    """Keep a mandatory locked compiler before the complete unfiltered pytest command."""
    jobs = object_mapping(
        read_workflow(workflow_path_for_job("test-partitions"))["jobs"]
    )
    job = object_mapping(jobs["test-partitions"])
    raw = job["steps"]
    assert isinstance(raw, list)
    steps = [object_mapping(step) for step in raw]
    setup = next(
        i
        for i, step in enumerate(steps)
        if "actions/setup-node@" in str(step.get("uses", ""))
    )
    install = next(
        i
        for i, step in enumerate(steps)
        if step.get("name") == "Install the locked native TypeScript compiler"
    )
    tests = next(
        i
        for i, step in enumerate(steps)
        if step.get("name") == "Run the complete assigned test partition"
    )
    assert setup < install < tests
    assert steps[install]["working-directory"] == "packages/vercel-ai"
    assert steps[install]["run"] == "npm ci --ignore-scripts --no-audit --no-fund"
    assert "if" not in steps[install] and not steps[install].get(
        "continue-on-error", False
    )
    command = str(steps[tests]["run"])
    assert "-m pytest tests/" in command
    assert '--partition-index "$PARTITION_INDEX"' in command
    assert "--partition-report partition.json" in command
    assert "--junitxml partition.xml" in command


def test_studio_manifest_contracts_run_after_the_locked_compiler_install() -> None:
    """Keep real manifest types and HTTP tests mandatory in the existing CI gate."""
    job = object_mapping(
        object_mapping(read_workflow(workflow_path_for_job("lint"))["jobs"])["lint"]
    )
    raw = job["steps"]
    assert isinstance(raw, list)
    steps = [object_mapping(step) for step in raw]
    sdk = next(
        i
        for i, step in enumerate(steps)
        if step.get("name")
        == "Vercel middleware types, HTTP contracts and API documentation"
    )
    studio = next(
        i
        for i, step in enumerate(steps)
        if step.get("name") == "Studio manifest types and real HTTP contracts"
    )
    assert sdk < studio
    assert "npm ci --ignore-scripts" in str(steps[sdk]["run"])
    assert steps[studio]["working-directory"] == "studio-web"
    assert steps[studio]["run"] == "npm run test:manifest\nnpm run docs:manifest\n"
    assert "if" not in steps[studio]
    assert not steps[studio].get("continue-on-error", False)


@pytest.mark.parametrize(
    "before,after",
    (
        ("npm run test:manifest", "node --check src/contract/manifest.ts"),
        ("working-directory: studio-web", "working-directory: packages/vercel-ai"),
        (
            "      - name: Studio manifest types and real HTTP contracts\n",
            "      - name: Studio manifest types and real HTTP contracts\n        continue-on-error: true\n",
        ),
    ),
)
def test_studio_manifest_gate_drift_is_refused(
    tmp_path: Path, before: str, after: str
) -> None:
    """Refuse syntax-only substitutes, wrong roots and optional manifest failures.

    Parameters
    ----------
    tmp_path : Path
        Isolated real workflow owners and their declared executable policy.
    before : str
        Original mandatory Studio declaration.
    after : str
        Invalid replacement which must fail the public workflow audit.
    """
    root = _native_checkout(tmp_path)
    path = workflow_path_for_job("lint", root)
    original = path.read_text()
    assert before in original
    path.write_text(original.replace(before, after, 1))
    assert (
        "Executable body differs from its versioned contract: lint"
        in audit_ci_workflow_modularity(root)
    )


@pytest.mark.parametrize(
    "before,after",
    (
        ('node-version: "26.13.0"', 'node-version: "0.0.0"'),
        ("working-directory: packages/vercel-ai", "working-directory: packages/absent"),
        (
            "npm ci --ignore-scripts --no-audit --no-fund",
            "npm install --ignore-scripts",
        ),
        (
            "      - name: Install the locked native TypeScript compiler\n",
            "      - name: Install the locked native TypeScript compiler\n        if: false\n",
        ),
        (
            "      - name: Install the locked native TypeScript compiler\n",
            "      - name: Install the locked native TypeScript compiler\n        continue-on-error: true\n",
        ),
    ),
)
def test_native_compiler_prerequisite_drift_is_refused(
    tmp_path: Path,
    before: str,
    after: str,
) -> None:
    """Reject real executable prerequisite mutations through the public workflow audit.

    Parameters
    ----------
    tmp_path : Path
        Isolated original source-bound workflow checkout.
    before : str
        Exact original native prerequisite declaration.
    after : str
        Deliberately invalid version, installation, condition or failure policy.
    """
    root = _native_checkout(tmp_path)
    path = workflow_path_for_job("test-partitions", root)
    original = path.read_text()
    assert before in original
    path.write_text(original.replace(before, after, 1))
    assert (
        "Executable body differs from its versioned contract: test-partitions"
        in audit_ci_workflow_modularity(root)
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


def test_sdk_generated_package_gate_uses_actual_prepack_and_complete_inputs() -> None:
    """Keep native packaging, source-bound verification and Python owner gates required."""
    jobs = object_mapping(read_workflow(workflow_path_for_job("lint"))["jobs"])
    lint = object_mapping(jobs["lint"])
    raw = lint["steps"]
    assert isinstance(raw, list)
    steps = [object_mapping(step) for step in raw]
    sdk = next(
        step
        for step in steps
        if step.get("name")
        == "Vercel middleware types, HTTP contracts and API documentation"
    )
    assert sdk["working-directory"] == "packages/vercel-ai"
    assert "if" not in sdk and not sdk.get("continue-on-error", False)
    run = str(sdk["run"])
    assert run.index("npm audit --audit-level=low") < run.index(
        "npm pack --offline --json"
    )
    assert 'npm pack --offline --json --pack-destination "$artefacts"' in run
    assert "npm pack --ignore-scripts" not in run
    assert (
        'PYTHONPATH="$GITHUB_WORKSPACE" python -m tools.verify_typescript_generated'
        in run
    )
    assert '--root "$GITHUB_WORKSPACE"' in run
    assert '--artefacts "$GITHUB_WORKSPACE/packages/vercel-ai"' in run
    assert '--package "$artefacts/"*.tgz' in run
    assert '--workspace "$artefacts/verification"' in run
    documentation = next(
        step
        for step in steps
        if step.get("name") == "Generated SDK Python API documentation"
    )
    assert "--isolated --select D" in str(documentation["run"])
    assert "if" not in documentation and not documentation.get(
        "continue-on-error", False
    )
    typecheck = object_mapping(jobs["typecheck"])
    checks = typecheck["steps"]
    assert isinstance(checks, list)
    owner = next(
        object_mapping(step)
        for step in checks
        if object_mapping(step).get("name")
        == "Workflow ownership and complete-test contracts"
    )
    strict = str(owner["run"])
    assert "mypy --strict" in strict
    for name in (
        "typescript_generated_recipe",
        "typescript_package_artifacts",
        "verify_typescript_generated",
    ):
        assert f"tools/{name}.py" in strict
        assert f"tests/test_{name}.py" in strict
        assert f"tools/{name}.py" in str(documentation["run"])
        assert f"tests/test_{name}.py" in str(documentation["run"])


@pytest.mark.parametrize(
    "before,after",
    [
        ("npm pack --offline --json", "npm pack --ignore-scripts --offline --json"),
        ("python -m tools.verify_typescript_generated", "python -m compileall"),
        ('--package "$artefacts/"*.tgz', '--package "$artefacts/missing.tgz"'),
        (
            '--workspace "$artefacts/verification"',
            '--workspace "$GITHUB_WORKSPACE/packages/vercel-ai"',
        ),
        (
            "      - name: Vercel middleware types, HTTP contracts and API documentation\n",
            "      - name: Vercel middleware types, HTTP contracts and API documentation\n        continue-on-error: true\n",
        ),
        (
            "      - name: Generated SDK Python API documentation\n",
            "      - name: Generated SDK Python API documentation\n        if: false\n",
        ),
    ],
)
def test_generated_package_gate_mutations_refuse(
    tmp_path: Path, before: str, after: str
) -> None:
    """Reject actual packaging bypasses, incomplete inputs and optional checks.

    Parameters
    ----------
    tmp_path : Path
        Real source-bound workflow and policy fixture.
    before : str
        Original mandatory executable gate declaration.
    after : str
        Deliberately invalid packaging, input, workspace or failure policy.
    """
    root = _native_checkout(tmp_path)
    path = workflow_path_for_job("lint", root)
    original = path.read_text()
    assert before in original
    path.write_text(original.replace(before, after, 1))
    assert (
        "Executable body differs from its versioned contract: lint"
        in audit_ci_workflow_modularity(root)
    )


def test_gateway_native_profile_and_complete_lock_audit_are_required() -> None:
    """Bind real compiler hosts, verified k6 and all dependencies to the required job."""
    job = object_mapping(
        object_mapping(read_workflow(workflow_path_for_job("lint"))["jobs"])["lint"]
    )
    raw = job["steps"]
    assert isinstance(raw, list)
    steps = [object_mapping(step) for step in raw]
    install = next(
        step
        for step in steps
        if step.get("name") == "Install the pinned native k6 runtime"
    )
    gateway = next(
        step
        for step in steps
        if step.get("name")
        == "Gateway native JavaScript types, runtime and documentation"
    )
    assert steps.index(install) < steps.index(gateway)
    assert object_mapping(install["env"])["K6_ARCHIVE_SHA256"] == (
        "39c3117b6af817592dcd0ce4242105c0a7af10948c2a425306f0be8f7a8a8ab1"
    )
    assert "sha256sum --check --strict" in str(install["run"])
    assert "--strip-components=1 k6-v2.3.0-linux-amd64/k6" in str(install["run"])
    assert gateway["working-directory"] == "gateway/go/bench"
    assert (
        gateway["run"]
        == "printf '{}\\n' > \"$K6_CONFIG\"\nnpm ci --ignore-scripts --no-audit --no-fund\nnpm test\nnpm run docs\nnpm audit --audit-level=low\n"
    )
    assert object_mapping(gateway["env"]) == {
        "K6_BINARY": "${{ runner.temp }}/director-k6/k6",
        "K6_CONFIG": "${{ runner.temp }}/director-k6/config.json",
        "K6_NO_USAGE_REPORT": "true",
        "K6_AUTO_EXTENSION_RESOLUTION": "false",
        "K6_WEB_DASHBOARD": "false",
    }
    for step in (install, gateway):
        assert "if" not in step and not step.get("continue-on-error", False)
    package = ROOT / "gateway/go/bench"
    compiler = object_mapping(json.loads((package / "tsconfig.json").read_text()))
    assert compiler["files"] == ["passthrough.js"]
    options = object_mapping(compiler["compilerOptions"])
    assert options["types"] == ["k6"]
    assert options["lib"] == ["ES2022"]
    for name in (
        "strict",
        "checkJs",
        "noEmit",
        "allowJs",
        "exactOptionalPropertyTypes",
        "noUncheckedIndexedAccess",
        "noUnusedLocals",
        "noUnusedParameters",
    ):
        assert options[name] is True
    assert options["skipLibCheck"] is False
    tools = object_mapping(json.loads((package / "tsconfig.tools.json").read_text()))
    assert object_mapping(tools["compilerOptions"])["types"] == ["node"]
    assert tools["include"] == [
        "tools/**/*.mjs",
        "test/**/*.mjs",
        "../../../benchmarks/wasm_overhead_node.mjs",
        "../../../benchmarks/wasm_overhead_node.test.mjs",
    ]
    metadata = object_mapping(json.loads((package / "package.json").read_text()))
    scripts = object_mapping(metadata["scripts"])
    assert scripts["check"] == "tsc -p tsconfig.json && tsc -p tsconfig.tools.json"
    assert "--test-coverage-lines=100" in str(scripts["test"])
    assert "--test-coverage-branches=100" in str(scripts["test"])
    assert "--test-coverage-functions=100" in str(scripts["test"])
    assert "--test-coverage-include=test/*.mjs" in str(scripts["test"])
    assert "--test-coverage-include=tools/*.mjs" in str(scripts["test"])
    assert "--test-coverage-exclude=node_modules/**" in str(scripts["test"])
    assert "--test-concurrency=1" in str(scripts["test"])
    for flag in (
        "--test-coverage-lines=100",
        "--test-coverage-branches=100",
        "--test-coverage-functions=100",
        "--test-coverage-exclude=node_modules/**",
        "--test-coverage-include=../../../benchmarks/wasm_overhead_node.mjs",
        "--test-coverage-include=../../../benchmarks/wasm_overhead_node.test.mjs",
    ):
        assert flag in str(scripts["test:wasm"])


@pytest.mark.parametrize(
    "before,after",
    [
        ("sha256sum --check --strict", "sha256sum --version"),
        (
            'K6_AUTO_EXTENSION_RESOLUTION: "false"',
            'K6_AUTO_EXTENSION_RESOLUTION: "true"',
        ),
        (
            "working-directory: gateway/go/bench",
            "working-directory: packages/vercel-ai",
        ),
        (
            "npm audit --audit-level=low\n\n      - name: Studio manifest",
            "npm audit --audit-level=critical\n\n      - name: Studio manifest",
        ),
        (
            "      - name: Gateway native JavaScript types, runtime and documentation\n",
            "      - name: Gateway native JavaScript types, runtime and documentation\n        continue-on-error: true\n",
        ),
    ],
)
def test_gateway_native_gate_drift_refuses(
    tmp_path: Path, before: str, after: str
) -> None:
    """Refuse weakened real checksum, runtime, location, audit or failure contracts.

    Parameters
    ----------
    tmp_path : Path
        Actual workflow/policy inputs retained for the deliberate invalid control.
    before : str
        Original mandatory executable declaration.
    after : str
        Invalid replacement passed through the real public workflow audit.
    """
    root = _native_checkout(tmp_path)
    path = workflow_path_for_job("lint", root)
    original = path.read_text()
    assert before in original
    path.write_text(original.replace(before, after, 1))
    assert (
        "Executable body differs from its versioned contract: lint"
        in audit_ci_workflow_modularity(root)
    )
