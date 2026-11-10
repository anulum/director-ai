# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — WASM execution and comparison workflow contracts
"""Bind actual required jobs to real wasm32 tests and all-runtime comparisons."""

import shutil
import tomllib
from pathlib import Path

import pytest

from tools.audit_ci_workflow_modularity import (
    audit_ci_workflow_modularity,
    read_workflow,
)
from tools.ci_workflow_inventory import (
    POLICY_PATH,
    ROOT,
    ci_workflow_paths,
    object_mapping,
)
from tools.quality_ratchet_typescript import NODE_VERSION


def _steps(job_id: str, workflow: str) -> list[dict[str, object]]:
    """Read actual native workflow steps without substituting an invented graph."""
    job = object_mapping(object_mapping(read_workflow(ROOT / workflow)["jobs"])[job_id])
    raw = job["steps"]
    assert isinstance(raw, list)
    return [object_mapping(item) for item in raw]


def test_required_benchmark_job_runs_the_real_three_runtime_comparison() -> None:
    """The existing required job builds WASM before strict tests and full measurement."""
    steps = _steps("benchmark", ".github/workflows/ci-benchmark.yml")
    setup = next(
        item for item in steps if "actions/setup-node@" in str(item.get("uses", ""))
    )
    assert (
        setup["uses"] == "actions/setup-node@949feb2413d6458794dcd2491c4babbbce0c15c1"
    )
    assert object_mapping(setup["with"])["node-version"] == NODE_VERSION
    names = [item.get("name") for item in steps]
    build = names.index("Build the Node WASM package")
    check = names.index("Check streaming comparison sources and real runtime contracts")
    compare = names.index("Compare all three streaming runtimes")
    legacy = names.index("Run regression benchmarks")
    assert build < check < compare < legacy
    for index in [build, check, compare, legacy]:
        assert "if" not in steps[index]
        assert not steps[index].get("continue-on-error", False)
    assert (
        steps[build]["run"]
        == "cargo install wasm-pack@0.15.1 --locked\nmake wasm-build-node\n"
    )
    checks = str(steps[check]["run"])
    assert "npm --prefix gateway/go/bench run test:wasm" in checks
    assert "mypy --strict benchmarks/wasm_overhead_bench.py" in checks
    assert "tests/test_wasm_ci_contracts.py" in checks
    assert "coverage combine --keep" in checks
    assert "--fail-under=100" in checks
    assert "--treatWarningsAsErrors" in checks
    assert "--validation.notDocumented" in checks
    assert "coverage json" in checks
    assert str(object_mapping(steps[check]["env"])["NODE_V8_COVERAGE"]).endswith(
        "/build/streaming-comparison/node-coverage"
    )
    upload = next(item for item in steps if item.get("name") == "Upload results")
    assert upload["if"] == "always()"
    assert object_mapping(upload["with"])["include-hidden-files"] == "true"
    assert object_mapping(upload["with"])["path"] == (
        "benchmarks/results/\nbuild/streaming-comparison/\n"
    )
    command = str(steps[compare]["run"])
    assert "python -m benchmarks.wasm_overhead_bench" in command
    assert "pkg-nodejs/backfire_wasm.js" in command
    assert "--tokens" not in command and "--iterations" not in command
    assert "|| true" not in checks + command
    assert steps[legacy]["run"] == "python -m benchmarks.regression_suite"


def test_required_rust_job_executes_wasm32_tests_in_the_pinned_node_host() -> None:
    """Host Rust tests cannot substitute for the wasm-bindgen Node execution lane."""
    steps = _steps("rust", ".github/workflows/ci-rust.yml")
    setup = next(
        item for item in steps if "actions/setup-node@" in str(item.get("uses", ""))
    )
    assert object_mapping(setup["with"])["node-version"] == NODE_VERSION
    toolchain = next(
        item for item in steps if "dtolnay/rust-toolchain@" in str(item.get("uses", ""))
    )
    assert object_mapping(toolchain["with"])["targets"] == "wasm32-unknown-unknown"
    native = next(item for item in steps if item.get("name") == "Test")
    assert "cargo test --workspace --exclude backfire-ffi --locked" in str(
        native["run"]
    )
    wasm = next(
        item for item in steps if item.get("name") == "Test the wasm32 runtime in Node"
    )
    assert "if" not in wasm and not wasm.get("continue-on-error", False)
    assert "wasm-pack test --node -- --locked" in str(wasm["run"])


def test_release_and_make_consumers_keep_web_and_node_packages_distinct() -> None:
    """The actual public commands preserve the web package while adding Node output."""
    plan = tomllib.loads((ROOT / "requirements/wasm_release_plan.toml").read_text())
    assert plan["policy"]["wasm_pack_version"] == "0.15.1"
    assert "wasm-build-node" in plan["policy"]["make_targets"]
    makefile = (ROOT / "Makefile").read_text()
    assert "--target nodejs --release --out-dir pkg-nodejs -- --locked" in makefile
    assert "/media/anulum/" not in makefile
    steps = _steps("build-wasm", ".github/workflows/wheels.yml")
    install = next(item for item in steps if item.get("name") == "Install wasm-pack")
    assert install["run"] == "cargo install wasm-pack@0.15.1 --locked"
    build = next(item for item in steps if item.get("name") == "Build WASM package")
    assert (
        build["run"]
        == "wasm-pack build --target web --release --out-dir pkg -- --locked"
    )


@pytest.mark.parametrize(
    ("path", "before", "after"),
    [
        (
            ".github/workflows/ci-benchmark.yml",
            "npm --prefix gateway/go/bench run test:wasm",
            "echo omitted",
        ),
        (
            ".github/workflows/ci-benchmark.yml",
            "python -m benchmarks.wasm_overhead_bench",
            "echo omitted",
        ),
        (
            ".github/workflows/ci-rust.yml",
            "wasm-pack test --node -- --locked",
            "echo omitted",
        ),
    ],
)
def test_required_wasm_workflow_omissions_are_refused(
    tmp_path: Path,
    path: str,
    before: str,
    after: str,
) -> None:
    """The real workflow reader detects omitted mandatory runtime commands.

    Parameters
    ----------
    tmp_path : Path
        Isolated actual-workflow copy used by the public audit API.
    path, before, after : str
        Exact native workflow command and its omission under test.
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
    violations = audit_ci_workflow_modularity(tmp_path)
    assert any("Executable body differs" in violation for violation in violations)
