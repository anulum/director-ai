# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — complete generated SDK public verification tests
"""Exercise genuine compiler, documentation, npm and Git contexts through API/CLI."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.test_typescript_generated_recipe import ROOT, SDK, SDKCase
from tests.test_typescript_generated_recipe import sdk_case as sdk_case
from tests.test_typescript_generated_recipe import sdk_copy as sdk_copy
from tools.ci_workflow_inventory import object_mapping
from tools.typescript_generated_recipe import run_native
from tools.verify_typescript_generated import main, verify_typescript_generated


def test_complete_public_api_binds_real_source_package_and_documentation(
    sdk_case: SDKCase, tmp_path: Path
) -> None:
    """Regenerate with actual native tools and bind the real complete package.

    Parameters
    ----------
    sdk_case : SDKCase
        Actual original source/Git checkout with its genuine producer outputs.
    tmp_path : Path
        New external candidate/receipt workspace owned by the runner.
    """
    result = verify_typescript_generated(
        sdk_case.root, sdk_case.sdk, tmp_path / "api", sdk_case.archive
    )
    assert (
        result.source_commit
        == subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=sdk_case.root, text=True
        ).strip()
    )
    assert result.git_remote == "https://github.com/anulum/director-ai.git"
    assert result.package_version == "0.1.0"
    assert (
        result.package_archive_sha256
        == hashlib.sha256(sdk_case.archive.read_bytes()).hexdigest()
    )
    assert (
        result.inputs["packages/vercel-ai/src/index.ts"]
        == hashlib.sha256((sdk_case.sdk / "src/index.ts").read_bytes()).hexdigest()
    )
    assert (
        result.outputs["dist/index.d.ts"]
        == hashlib.sha256((sdk_case.sdk / "dist/index.d.ts").read_bytes()).hexdigest()
    )
    assert len(result.outputs) == 20 and len(result.package_members) == 6
    assert "typescript/lib/typescript.js" in result.dependencies
    assert len(result.identity) == len(result.engine_identity) == 64
    assert set(result.input_modes) == set(result.inputs)
    assert set(result.output_modes) == set(result.outputs)
    assert result.node_mode & 0o111


@pytest.mark.parametrize(
    "origin",
    [
        None,
        "git@github.com:anulum/director-ai.git",
        "ssh://git@github.com/anulum/director-ai.git",
    ],
)
def test_actual_origin_and_absence_are_native_documentation_inputs(
    sdk_copy: SDKCase, tmp_path: Path, origin: str | None
) -> None:
    """Bind real origin context and keep absent-origin native warnings blocking.

    Parameters
    ----------
    sdk_copy : SDKCase
        Actual source/Git fixture with independent mutable metadata.
    tmp_path : Path
        Owned renderer receipts and verifier workspace.
    origin : str | None
        Actual origin URL or explicitly absent source-link metadata.
    """
    command = (
        ["git", "remote", "remove", "origin"]
        if origin is None
        else ["git", "remote", "set-url", "origin", origin]
    )
    subprocess.run(command, cwd=sdk_copy.root, check=True, capture_output=True)
    if origin is None:
        workspace = tmp_path / "absent-origin"
        with pytest.raises(ValueError, match="did not qualify"):
            verify_typescript_generated(
                sdk_copy.root, sdk_copy.sdk, workspace, sdk_copy.archive
            )
        receipt = json.loads((workspace / "generator-1.json").read_text())
        assert receipt["exit"] != 0
        assert 'git remote "origin" was not valid' in receipt["stderr"]
        return
    run_native(
        [
            "node",
            str(SDK / "node_modules/typedoc/bin/typedoc"),
            "--options",
            str(sdk_copy.sdk / "typedoc.json"),
        ],
        sdk_copy.sdk,
        time.monotonic() + 90,
        os.environ.copy(),
        tmp_path / "real-renderer.json",
    )
    result = verify_typescript_generated(
        sdk_copy.root, sdk_copy.sdk, tmp_path / "context", sdk_copy.archive
    )
    assert result.git_remote == origin
    assert (
        result.outputs["build/api-docs/index.html"]
        == hashlib.sha256(
            (sdk_copy.sdk / "build/api-docs/index.html").read_bytes()
        ).hexdigest()
    )


@pytest.mark.parametrize(
    "origin",
    [
        "https://git:CONTROL_ONLY@github.com/anulum/director-ai.git",
        "https://github.com/anulum/director-ai.git?CONTROL_ONLY",
        "https://github.com/anulum/director-ai.git#CONTROL_ONLY",
        "https://user@github.com/anulum/director-ai.git",
        "file:///CONTROL_ONLY",
        " https://github.com/anulum/director-ai.git",
        "https://github.com/anulum/director-ai.git ",
        "https://github.com/anulum/director-ai.git\n",
    ],
)
def test_unsafe_actual_origins_refuse_without_leaking_probe_output(
    sdk_copy: SDKCase, tmp_path: Path, origin: str
) -> None:
    """Refuse authenticated/control-bearing metadata before any URL is persisted.

    Parameters
    ----------
    sdk_copy : SDKCase
        Real source fixture; no real credential is used in these negative inputs.
    tmp_path : Path
        Retained command receipt directory to inspect for accidental disclosure.
    origin : str
        Actual invalid Git config value, including significant surrounding space.
    """
    subprocess.run(
        ["git", "remote", "set-url", "origin", origin],
        cwd=sdk_copy.root,
        check=True,
        capture_output=True,
    )
    workspace = tmp_path / "unsafe"
    with pytest.raises(ValueError, match="safe origin"):
        verify_typescript_generated(
            sdk_copy.root, sdk_copy.sdk, workspace, sdk_copy.archive
        )
    for receipt in workspace.glob("*.json"):
        assert "CONTROL_ONLY" not in receipt.read_text()
    probe = json.loads((workspace / "before-remote-command.json").read_text())
    assert probe["stdout"] is None and probe["stderr"] is None
    assert not (workspace / "before-remote-context.json").exists()


@pytest.mark.parametrize(
    "configuration,section,key,value",
    [
        ("tsconfig.json", "compilerOptions", "sourceMap", True),
        ("tsconfig.json", "compilerOptions", "declarationMap", True),
        ("tsconfig.json", "compilerOptions", "inlineSources", True),
        ("tsconfig.json", "compilerOptions", "strict", False),
        ("typedoc.json", "validation", "notDocumented", False),
        ("typedoc.json", None, "out", "other-docs"),
    ],
)
def test_changed_native_configuration_requires_new_recipe_qualification(
    sdk_copy: SDKCase,
    tmp_path: Path,
    configuration: str,
    section: str | None,
    key: str,
    value: object,
) -> None:
    """Reject actual source-map, strictness, documentation or output recipe drift.

    Parameters
    ----------
    sdk_copy : SDKCase
        Real editable SDK configuration fixture.
    tmp_path : Path
        Fresh owned verifier workspace.
    configuration : str
        Actual compiler or documentation config file to change.
    section : str | None
        Nested config mapping, or the original top-level mapping.
    key : str
        Native option whose unqualified change must refuse.
    value : object
        Actual changed JSON value passed to the native configuration boundary.
    """
    path = sdk_copy.sdk / configuration
    config = object_mapping(json.loads(path.read_text()))
    if section is None:
        config[key] = value
    else:
        options = object_mapping(config[section])
        options[key] = value
        config[section] = options
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="configuration differs"):
        verify_typescript_generated(
            sdk_copy.root, sdk_copy.sdk, tmp_path / "config", sdk_copy.archive
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("version", ""),
        ("version", False),
        ("files", ["dist"]),
        ("main", "other.js"),
        ("types", None),
        ("bin", "unqualified-command"),
    ],
)
def test_actual_public_package_contract_changes_refuse(
    sdk_copy: SDKCase, tmp_path: Path, key: str, value: object
) -> None:
    """Require the exact native package entry points and complete licensed payload.

    Parameters
    ----------
    sdk_copy : SDKCase
        Actual package metadata fixture.
    tmp_path : Path
        Owned fresh verifier directory.
    key : str
        Native package field to change or introduce.
    value : object
        Actual changed JSON contract value.
    """
    path = sdk_copy.sdk / "package.json"
    package = object_mapping(json.loads(path.read_text()))
    package[key] = value
    path.write_text(json.dumps(package))
    with pytest.raises(ValueError, match="public package contract"):
        verify_typescript_generated(
            sdk_copy.root, sdk_copy.sdk, tmp_path / "package", sdk_copy.archive
        )


@pytest.mark.parametrize("fault", ["lock", "required-input", "git-root"])
def test_actual_source_and_installed_input_boundaries_refuse(
    sdk_copy: SDKCase, tmp_path: Path, fault: str
) -> None:
    """Use real Git and source bytes to refuse incomplete or mismatched recipes.

    Parameters
    ----------
    sdk_copy : SDKCase
        Real source/Git/native-output fixture.
    tmp_path : Path
        Owned verifier workspace outside all source roots.
    fault : str
        Actual lock mismatch, missing Git-visible input or nonroot selection.
    """
    root = sdk_copy.root
    if fault == "lock":
        with (sdk_copy.sdk / "package-lock.json").open("a") as stream:
            stream.write("\n")
    elif fault == "required-input":
        subprocess.run(
            ["git", "rm", "--", "packages/vercel-ai/tsconfig.test.json"],
            cwd=root,
            check=True,
            capture_output=True,
        )
    else:
        root = sdk_copy.root / "nested"
        shutil.copytree(sdk_copy.sdk, root / "packages/vercel-ai")
    with pytest.raises(
        ValueError, match="lock differs|inventory is incomplete|actual Git root"
    ):
        verify_typescript_generated(
            root, sdk_copy.sdk, tmp_path / "source", sdk_copy.archive
        )


@pytest.mark.parametrize(
    "fault",
    [
        "missing-entry",
        "empty-file",
        "fifo",
        "symbolic-file",
        "symbolic-directory",
        "empty-directory",
        "missing-output-root",
        "symbolic-root",
        "bytes",
        "extra",
        "mode",
    ],
)
def test_complete_actual_generated_tree_refusals(
    sdk_copy: SDKCase, tmp_path: Path, fault: str
) -> None:
    """Refuse real generated-tree corruption before or after genuine regeneration.

    Parameters
    ----------
    sdk_copy : SDKCase
        Actual mutable generated oracle, never an editable passed quality report.
    tmp_path : Path
        Fresh private verifier and link boundary locations.
    fault : str
        Real absent, changed, extra, empty, nonregular or permission input.
    """
    if fault in {"bytes", "extra", "mode"}:
        run_native(
            [
                "node",
                str(SDK / "node_modules/typedoc/bin/typedoc"),
                "--options",
                str(sdk_copy.sdk / "typedoc.json"),
            ],
            sdk_copy.sdk,
            time.monotonic() + 90,
            os.environ.copy(),
            tmp_path / "source-context-renderer.json",
        )
    artefacts = sdk_copy.sdk
    path = artefacts / "dist/index.js"
    if fault == "missing-entry":
        path.unlink()
    elif fault == "empty-file":
        path.write_bytes(b"")
    elif fault == "fifo":
        path.unlink()
        os.mkfifo(path)
    elif fault == "symbolic-file":
        path.unlink()
        path.symlink_to(artefacts / "README.md")
    elif fault == "symbolic-directory":
        (artefacts / "build/api-docs/alias").symlink_to(
            tmp_path, target_is_directory=True
        )
    elif fault == "empty-directory":
        (artefacts / "build/api-docs/empty").mkdir()
    elif fault == "missing-output-root":
        shutil.rmtree(artefacts / "dist")
    elif fault == "symbolic-root":
        artefacts = tmp_path / "alias"
        artefacts.symlink_to(sdk_copy.sdk, target_is_directory=True)
    elif fault == "bytes":
        with path.open("ab") as stream:
            stream.write(b"changed generated JavaScript")
    elif fault == "extra":
        (artefacts / "build/api-docs/extra.html").write_text("unowned documentation")
    else:
        path.chmod(path.stat().st_mode ^ 0o111)
    messages = {
        "missing-entry": "lacks required public entry points",
        "empty-file": "empty or nonregular",
        "fifo": "empty or nonregular",
        "symbolic-file": "empty or nonregular",
        "symbolic-directory": "symbolic directory",
        "empty-directory": "empty directory",
        "missing-output-root": "No such file or directory",
        "symbolic-root": "exact real directory",
        "bytes": "differ from the actual native recipe",
        "extra": "differ from the actual native recipe",
        "mode": "differ from the actual native recipe",
    }
    with pytest.raises((ValueError, OSError), match=messages[fault]):
        verify_typescript_generated(
            sdk_copy.root, artefacts, tmp_path / "outputs", sdk_copy.archive
        )


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_whole_api_deadlines_refuse(
    sdk_case: SDKCase, tmp_path: Path, timeout: float
) -> None:
    """Reject invalid whole budgets without starting any native generator.

    Parameters
    ----------
    sdk_case : SDKCase
        Original real source and native outputs.
    tmp_path : Path
        Owned candidate path that must remain absent.
    timeout : float
        Actual invalid nonpositive or nonfinite API budget.
    """
    workspace = tmp_path / "deadline"
    with pytest.raises(ValueError, match="positive and finite"):
        verify_typescript_generated(
            sdk_case.root, sdk_case.sdk, workspace, sdk_case.archive, timeout=timeout
        )
    assert not workspace.exists()


@pytest.mark.parametrize("fault", ["source", "artefact", "existing"])
def test_workspace_input_overlap_and_existing_custody_refuse(
    sdk_case: SDKCase, tmp_path: Path, fault: str
) -> None:
    """Preserve real sources, supplied outputs and existing owned workspace bytes.

    Parameters
    ----------
    sdk_case : SDKCase
        Actual protected SDK source and generated output roots.
    tmp_path : Path
        Real external existing-workspace fixture.
    fault : str
        Actual overlap or preexisting allocation to refuse.
    """
    paths = {
        "source": sdk_case.root / "forbidden",
        "artefact": sdk_case.sdk / "forbidden",
        "existing": tmp_path,
    }
    with pytest.raises((ValueError, OSError)):
        verify_typescript_generated(
            sdk_case.root, sdk_case.sdk, paths[fault], sdk_case.archive
        )
    assert not (sdk_case.root / "forbidden").exists()
    assert not (sdk_case.sdk / "forbidden").exists()


def test_real_cli_success_and_authored_refusal(
    sdk_case: SDKCase, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run the public module CLI and verify safe refusal through its callable entry.

    Parameters
    ----------
    sdk_case : SDKCase
        Actual SDK/native/npm fixture, never mocked producer evidence.
    tmp_path : Path
        Owned CLI workspace.
    capsys : pytest.CaptureFixture[str]
        Capture the authored public refusal, without revealing internal exceptions.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.verify_typescript_generated",
            "--root",
            str(sdk_case.root),
            "--artefacts",
            str(sdk_case.sdk),
            "--package",
            str(sdk_case.archive),
            "--workspace",
            str(tmp_path / "cli"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = object_mapping(json.loads(result.stdout))
    assert payload["package_version"] == "0.1.0"
    assert payload["git_remote"] == "https://github.com/anulum/director-ai.git"
    assert (
        main(
            [
                "--artefacts",
                str(tmp_path / "absent"),
                "--package",
                str(sdk_case.archive),
                "--workspace",
                str(tmp_path / "refused"),
            ]
        )
        == 2
    )
    assert (
        capsys.readouterr().out.strip()
        == "Native TypeScript generated artefact verification refused"
    )


@pytest.mark.parametrize(
    "fault",
    [
        "compiler",
        "renderer",
        "input",
        "input-mode",
        "head",
        "remote",
        "tool",
        "artefact",
        "artefact-mode",
        "archive",
    ],
)
def test_genuine_native_failures_and_late_custody_refuse(
    sdk_copy: SDKCase, tmp_path: Path, fault: str
) -> None:
    """Execute real generators and reject real producer failures or late mutations.

    Parameters
    ----------
    sdk_copy : SDKCase
        Independent actual SDK/Git fixture with its own qualified documentation.
    tmp_path : Path
        Owned native launcher, mutable archive and retained original receipts.
    fault : str
        Actual compiler/renderer option error or post-render filesystem/Git drift.
    """
    if fault not in {"compiler", "renderer"}:
        run_native(
            [
                "node",
                str(SDK / "node_modules/typedoc/bin/typedoc"),
                "--options",
                str(sdk_copy.sdk / "typedoc.json"),
            ],
            sdk_copy.sdk,
            time.monotonic() + 90,
            os.environ.copy(),
            tmp_path / "source-context-renderer.json",
        )
    archive = tmp_path / "mutable-package.tgz"
    shutil.copyfile(sdk_copy.archive, archive)
    executable = shutil.which("node")
    assert executable is not None
    launcher = tmp_path / "real-native-node"
    program = f"""#!{sys.executable}
import gzip
import subprocess
import sys
from pathlib import Path
args = sys.argv[1:]
fault = {fault!r}
source = Path({str(sdk_copy.sdk)!r})
root = Path({str(sdk_copy.root)!r})
archive = Path({str(archive)!r})
launcher = Path({str(launcher)!r})
if args and args[0].endswith('/typescript/bin/tsc') and fault == 'compiler':
    args.append('--director-unqualified-option')
if args and args[0].endswith('/typedoc/bin/typedoc') and fault == 'renderer':
    args.append('--director-unqualified-option')
result = subprocess.run([{executable!r}, *args], check=False)
if result.returncode == 0 and args and args[0].endswith('/typedoc/bin/typedoc'):
    if fault == 'input':
        (source / 'test/actual-post-render-change.mjs').write_text('export const changed = true;')
    elif fault == 'input-mode':
        path = source / 'README.md'
        path.chmod(path.stat().st_mode ^ 0o100)
    elif fault == 'head':
        subprocess.run(['git', 'commit', '--allow-empty', '-m', 'Actual source HEAD changed'], cwd=root, check=True)
    elif fault == 'remote':
        subprocess.run(['git', 'remote', 'set-url', 'origin', 'https://github.com/anulum/different.git'], cwd=root, check=True)
    elif fault == 'tool':
        with launcher.open('a') as stream:
            stream.write('\\n# Actual selected executable bytes changed\\n')
    elif fault == 'artefact':
        with (source / 'dist/index.js').open('ab') as stream:
            stream.write(b'actual late output change')
    elif fault == 'artefact-mode':
        path = source / 'dist/index.js'
        path.chmod(path.stat().st_mode ^ 0o100)
    elif fault == 'archive':
        archive.write_bytes(gzip.compress(gzip.decompress(archive.read_bytes()), mtime=1))
raise SystemExit(result.returncode)
"""
    launcher.write_text(program)
    launcher.chmod(0o700)
    workspace = tmp_path / "late"
    if fault in {"compiler", "renderer"}:
        message = "did not qualify"
    elif fault == "archive":
        message = "npm package changed during native generation"
    else:
        message = "custody changed"
    with pytest.raises(ValueError, match=message):
        verify_typescript_generated(
            sdk_copy.root, sdk_copy.sdk, workspace, archive, str(launcher)
        )
    compiler = json.loads((workspace / "generator-0.json").read_text())
    if fault == "compiler":
        assert compiler["exit"] != 0
        assert "director-unqualified-option" in compiler["stdout"]
    else:
        assert compiler["exit"] == 0
        renderer = json.loads((workspace / "generator-1.json").read_text())
        if fault == "renderer":
            assert renderer["exit"] != 0
            assert "director-unqualified-option" in renderer["stderr"]
        else:
            assert renderer["exit"] == 0


def test_actual_native_versions_cannot_satisfy_a_corrupted_profile(
    sdk_copy: SDKCase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep genuine native metadata and reject an actual invalid policy reference.

    Parameters
    ----------
    sdk_copy : SDKCase
        Real source fixture and original native artifacts.
    tmp_path : Path
        Actual raw native metadata receipt location.
    monkeypatch : pytest.MonkeyPatch
        Negative-only profile corruption; no native response or success is mocked.
    """
    from tools.verify_typescript_generated import NATIVE_VERSIONS

    monkeypatch.setitem(NATIVE_VERSIONS, "node", "0.0.0")
    workspace = tmp_path / "versions"
    with pytest.raises(ValueError, match="recipe versions differ"):
        verify_typescript_generated(
            sdk_copy.root, sdk_copy.sdk, workspace, sdk_copy.archive
        )
    receipt = json.loads((workspace / "native-versions.json").read_text())
    assert receipt["exit"] == 0
    actual = object_mapping(json.loads(receipt["stdout"]))
    assert actual["node"] == "26.13.0"
    assert actual["typescript"] == "6.0.3"
    assert actual["typedoc"] == "0.28.20"


def test_cross_checkout_documentation_is_not_normalized_into_acceptance(
    sdk_copy: SDKCase, tmp_path: Path
) -> None:
    """Reject original rendered source paths reused under a different physical checkout.

    Parameters
    ----------
    sdk_copy : SDKCase
        Real source/Git copy carrying the original checkout's supplied HTML.
    tmp_path : Path
        Fresh native regeneration and original comparison receipt directory.
    """
    workspace = tmp_path / "cross-checkout"
    with pytest.raises(ValueError, match="differ from the actual native recipe"):
        verify_typescript_generated(
            sdk_copy.root, sdk_copy.sdk, workspace, sdk_copy.archive
        )
    comparison = object_mapping(
        json.loads((workspace / "output-comparison.json").read_text())
    )
    expected = object_mapping(comparison["expected"])
    actual = object_mapping(comparison["actual"])
    changed = {name for name in expected if expected[name] != actual[name]}
    assert changed and all(name.startswith("build/api-docs/") for name in changed)
    assert comparison["expected_modes"] == comparison["actual_modes"]
    for index in range(2):
        receipt = json.loads((workspace / f"generator-{index}.json").read_text())
        assert receipt["exit"] == 0


@pytest.mark.parametrize("authenticated", [False, True])
def test_effective_git_rewrites_match_the_actual_native_renderer(
    sdk_copy: SDKCase, tmp_path: Path, authenticated: bool
) -> None:
    """Validate the rewritten origin actually used by TypeDoc before any rendering.

    Parameters
    ----------
    sdk_copy : SDKCase
        Real independent Git configuration and SDK source fixture.
    tmp_path : Path
        Owned effective-origin receipts and regenerated documentation output.
    authenticated : bool
        Whether the actual Git rewrite introduces a credential-bearing URL.
    """
    workspace = tmp_path / "effective-origin"
    if authenticated:
        replacement = "https://git:CONTROL_ONLY@github.com/"
        prefix = "https://github.com/"
    else:
        replacement = "https://github.com/"
        prefix = "director-upstream:"
        subprocess.run(
            [
                "git",
                "remote",
                "set-url",
                "origin",
                "director-upstream:anulum/director-ai.git",
            ],
            cwd=sdk_copy.root,
            check=True,
            capture_output=True,
        )
    subprocess.run(
        ["git", "config", f"url.{replacement}.insteadOf", prefix],
        cwd=sdk_copy.root,
        check=True,
        capture_output=True,
    )
    if authenticated:
        with pytest.raises(ValueError, match="safe origin"):
            verify_typescript_generated(
                sdk_copy.root, sdk_copy.sdk, workspace, sdk_copy.archive
            )
        assert not (workspace / "generator-0.json").exists()
        for receipt in workspace.glob("*.json"):
            assert "CONTROL_ONLY" not in receipt.read_text()
    else:
        run_native(
            [
                "node",
                str(SDK / "node_modules/typedoc/bin/typedoc"),
                "--options",
                str(sdk_copy.sdk / "typedoc.json"),
            ],
            sdk_copy.sdk,
            time.monotonic() + 90,
            os.environ.copy(),
            tmp_path / "effective-renderer.json",
        )
        result = verify_typescript_generated(
            sdk_copy.root, sdk_copy.sdk, workspace, sdk_copy.archive
        )
        assert result.git_remote == "https://github.com/anulum/director-ai.git"
        context = json.loads((workspace / "before-remote-context.json").read_text())
        assert context["origin"] == result.git_remote
