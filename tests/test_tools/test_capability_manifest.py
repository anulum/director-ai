# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_tool() -> Any:
    tool_path = _repo_root() / "tools" / "capability_manifest.py"
    spec = importlib.util.spec_from_file_location("capability_manifest", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_scans_director_ai_capability_surfaces() -> None:
    tool = _load_tool()
    manifest = tool.build_capability_manifest(_repo_root())

    assert manifest["schema_version"] == tool.CAPABILITY_MANIFEST_SCHEMA_VERSION
    assert manifest["generated_from"]["config"] == "tools/capability_manifest.toml"
    assert manifest["project_label"] == "Director-AI"
    assert manifest["project"]["name"] == "director-ai"
    assert manifest["project"]["readme"] == "README.md"
    assert manifest["counts"]["public_api_exports"] == len(manifest["package_exports"])
    assert manifest["counts"]["python_model_source_modules"] == len(
        manifest["models"]["python_source_modules"]
    )
    assert manifest["counts"]["python_model_classes"] == len(
        manifest["models"]["python_classes"]
    )
    assert manifest["counts"]["rust_pyo3_model_wrappers"] == len(
        manifest["models"]["rust_pyo3_wrappers"]
    )
    assert "enterprise" in manifest["packaging"]["optional_extras"]
    assert ".github/workflows/ci.yml" in manifest["quality_gates"]["github_workflows"]
    assert "tests/test_actor.py" in manifest["quality_gates"]["test_files"]
    assert "customer-model-factory.md" in manifest["models"]["documentation_pages"]
    assert "CoherenceScorer" in manifest["package_exports"]
    assert "RustCoherenceScorer" in manifest["models"]["rust_pyo3_wrappers"]


def test_manifest_validation_rejects_count_drift() -> None:
    tool = _load_tool()
    manifest = tool.build_capability_manifest(_repo_root())
    manifest["counts"]["python_model_classes"] += 1

    report = tool.validate_manifest(manifest)

    assert not report["passed"]
    assert "counts.python_model_classes does not match list length" in report["errors"]


def test_generated_outputs_are_current() -> None:
    tool = _load_tool()

    tool.assert_outputs_current(_repo_root())


def test_readme_snapshot_matches_generated_markdown() -> None:
    tool = _load_tool()
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")
    start = "<!-- capability-snapshot:start -->"
    end = "<!-- capability-snapshot:end -->"

    block = readme.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0].strip()

    assert (
        block
        == tool.render_markdown_snapshot(
            tool.build_capability_manifest(_repo_root())
        ).strip()
    )


def test_markdown_snapshot_is_readme_safe() -> None:
    tool = _load_tool()
    manifest = tool.build_capability_manifest(_repo_root())
    snapshot = tool.render_markdown_snapshot(manifest)

    assert "do not edit counts by hand" in snapshot
    assert "| Package version | 3.18.0 |" in snapshot
    assert "Evidence boundary" in snapshot


def test_refresh_outputs_updates_configured_readme_block() -> None:
    tool = _load_tool()
    with _tempdir() as repo:
        _write_portable_fixture(repo)
        config = tool.load_config(repo)

        json_path, markdown_path, readme_path = tool.refresh_outputs(
            repo, config=config
        )
        manifest = json.loads(json_path.read_text(encoding="utf-8"))
        readme = (repo / "README.md").read_text(encoding="utf-8")

        assert readme_path == repo / "README.md"
        assert markdown_path == repo / "docs/_generated/capability_snapshot.md"
        assert manifest["project_label"] == "Portable Project"
        assert manifest["counts"]["public_api_exports"] == 1
        assert manifest["counts"]["python_model_classes"] == 2
        assert manifest["models"]["rust_pyo3_wrappers"] == ["PortableRustFn"]
        assert "### Portable Project Capability Inventory" in readme
        assert "| Portable API exports | 1 |" in readme

        tool.assert_outputs_current(repo, config=config)


def test_cli_writes_valid_manifest_and_markdown() -> None:
    tool_path = _repo_root() / "tools" / "capability_manifest.py"
    with _tempdir() as tmpdir:
        json_path = tmpdir / "capability_manifest.json"
        markdown_path = tmpdir / "capability_snapshot.md"
        result = subprocess.run(
            [
                sys.executable,
                str(tool_path),
                "--repo",
                str(_repo_root()),
                "--output",
                str(json_path),
                "--markdown-output",
                str(markdown_path),
                "--no-readme",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        manifest = json.loads(json_path.read_text(encoding="utf-8"))
        assert manifest["schema_version"] == "capability-manifest.v1"
        assert markdown_path.read_text(encoding="utf-8").startswith(
            "<!-- SPDX-License-Identifier"
        )

        validate = subprocess.run(
            [
                sys.executable,
                str(tool_path),
                "--validate",
                str(json_path),
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert validate.returncode == 0, validate.stderr


def test_cli_uses_portable_config_and_refreshes_readme() -> None:
    tool_path = _repo_root() / "tools" / "capability_manifest.py"
    with _tempdir() as repo:
        _write_portable_fixture(repo)

        result = subprocess.run(
            [
                sys.executable,
                str(tool_path),
                "--repo",
                str(repo),
                "--config",
                "tools/capability_manifest.toml",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert "Refreshed" in result.stdout
        assert (repo / "docs/_generated/capability_manifest.json").exists()
        assert "Portable Project Capability Inventory" in (
            repo / "README.md"
        ).read_text(encoding="utf-8")


def test_git_scanner_reflects_pending_adds_and_deletes() -> None:
    tool = _load_tool()
    with _tempdir() as repo:
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        source_root = repo / "src/demo"
        _write_file(source_root / "old_module.py", "class OldModule:\n    pass\n")
        subprocess.run(
            ["git", "add", "src/demo/old_module.py"],
            cwd=repo,
            check=True,
            capture_output=True,
        )

        (source_root / "old_module.py").unlink()
        _write_file(source_root / "new_module.py", "class NewModule:\n    pass\n")

        paths = tool._tracked_or_discovered_files(
            source_root,
            repo=repo,
            suffixes=(".py",),
        )

        assert paths == [source_root / "new_module.py"]


def test_portable_config_defaults_and_path_fallbacks() -> None:
    """Config loading should support missing files and external config paths."""
    tool = _load_tool()
    with _tempdir() as repo:
        _write_portable_fixture(repo)

        defaults = tool.load_config(repo, Path("missing.toml"))
        assert defaults.project_label == "Director-AI"
        assert defaults.source_path is None

        external = repo.parent / "external_capability_manifest.toml"
        external.write_text(
            'project_label = "External Project"\n',
            encoding="utf-8",
        )
        config = tool.load_config(repo, external)

        assert config.project_label == "External Project"
        assert config.source_path == external.resolve()


def test_readme_refresh_rejects_missing_markers() -> None:
    """README refresh should fail loudly when configured markers are absent."""
    tool = _load_tool()
    with _tempdir() as repo:
        _write_portable_fixture(repo)
        config = tool.load_config(repo)
        (repo / "README.md").write_text("# Missing markers\n", encoding="utf-8")

        with pytest.raises(RuntimeError, match="missing capability snapshot markers"):
            tool.refresh_readme_block(repo, "snapshot", config=config)


def test_refresh_outputs_can_skip_readme_update() -> None:
    """Output refresh should support JSON/Markdown-only generation."""
    tool = _load_tool()
    with _tempdir() as repo:
        _write_portable_fixture(repo)
        config = tool.load_config(repo)

        json_path, markdown_path, readme_path = tool.refresh_outputs(
            repo,
            config=config,
            update_readme=False,
        )

        assert json_path.exists()
        assert markdown_path.exists()
        assert readme_path is None


def test_manifest_validation_reports_schema_shape_and_count_errors() -> None:
    """Manifest validation should report malformed payloads deterministically."""
    tool = _load_tool()

    report = tool.validate_manifest(
        {
            "schema_version": "wrong",
            "counts": {"python_model_classes": -1, "broken": "not-int"},
            "models": {"python_classes": "not-a-list"},
        }
    )

    assert not report["passed"]
    assert "schema_version mismatch" in report["errors"]
    assert "missing top-level key: project" in report["errors"]
    assert (
        "counts.python_model_classes must be a non-negative integer" in report["errors"]
    )
    assert "counts.broken must be a non-negative integer" in report["errors"]
    assert "models list missing for count python_model_classes" in report["errors"]

    counts_report = tool.validate_manifest({"schema_version": "wrong", "counts": []})
    assert "counts must be an object" in counts_report["errors"]


def test_assert_outputs_current_reports_missing_stale_and_readme_drift() -> None:
    """Generated-output checker should combine all drift errors in one message."""
    tool = _load_tool()
    with _tempdir() as repo:
        _write_portable_fixture(repo)
        config = tool.load_config(repo)
        tool.refresh_outputs(repo, config=config)
        (repo / "docs/_generated/capability_manifest.json").write_text(
            "{}\n", encoding="utf-8"
        )
        (repo / "docs/_generated/capability_snapshot.md").unlink()
        (repo / "README.md").write_text(
            "\n".join(
                [
                    "# Portable Project",
                    "",
                    "<!-- capability-snapshot:start -->",
                    "stale",
                    "<!-- capability-snapshot:end -->",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        with pytest.raises(RuntimeError) as exc_info:
            tool.assert_outputs_current(repo, config=config)

        message = str(exc_info.value)
        assert "stale generated manifest" in message
        assert "missing generated snapshot" in message
        assert "stale README capability block" in message

        tool.refresh_outputs(repo, config=config)
        (repo / "docs/_generated/capability_manifest.json").unlink()
        with pytest.raises(RuntimeError, match="missing generated manifest"):
            tool.assert_outputs_current(repo, config=config)

        tool.refresh_outputs(repo, config=config)
        (repo / "docs/_generated/capability_snapshot.md").write_text(
            "stale\n", encoding="utf-8"
        )
        with pytest.raises(RuntimeError, match="stale generated snapshot"):
            tool.assert_outputs_current(repo, config=config)

        tool.refresh_outputs(repo, config=config)
        (repo / "README.md").write_text("# Missing markers\n", encoding="utf-8")
        tool.assert_outputs_current(repo, config=config, check_readme=False)


def test_static_scanners_cover_absent_roots_and_ast_variants() -> None:
    """Static scanners should handle absent roots and export AST variants."""
    tool = _load_tool()
    with _tempdir() as repo:
        assert tool._public_exports(repo / "missing.py") == []
        assert (
            tool._python_model_sources(repo / "missing", repo=repo, exclude_parts=())
            == []
        )
        assert (
            tool._python_model_classes(repo / "missing", repo=repo, exclude_parts=())
            == []
        )
        assert tool._project_extras({"project": {"optional-dependencies": []}}) == []
        assert tool._workflow_files(repo / "missing", repo=repo) == []
        assert tool._python_files(repo / "missing", repo=repo) == []
        assert tool._markdown_docs(repo / "missing", repo=repo, exclude_parts=()) == []
        assert not tool._readme_block_matches(
            repo / "missing.md",
            "snapshot",
            config=tool.load_config(repo, Path("missing.toml")),
        )
        marked_config = tool.load_config(repo, Path("missing.toml"))
        no_marker_readme = repo / "README.md"
        no_marker_readme.write_text("# Missing markers\n", encoding="utf-8")
        assert not tool._readme_block_matches(
            no_marker_readme,
            "snapshot",
            config=marked_config,
        )

        init_path = repo / "package/__init__.py"
        _write_file(
            init_path,
            "\n".join(
                [
                    "__all__ = ['Beta', 123, 'Alpha']",
                    "IGNORED = sorted('abc', 'extra')",
                    "",
                ]
            ),
        )
        assert tool._public_exports(init_path) == ["Alpha", "Beta"]

        _write_file(init_path, "__all__ = sorted(MISSING)\n")
        assert tool._public_exports(init_path) == []

        _write_file(init_path, "__all__ = NAMES\n")
        assert tool._public_exports(init_path) == []

        _write_file(init_path, "__all__ = sorted()\n")
        assert tool._public_exports(init_path) == []

        _write_file(init_path, "__all__ = sorted(['Alpha'])\n")
        assert tool._public_exports(init_path) == []

        _write_file(init_path, "__all__ = list(NAMES)\n")
        assert tool._public_exports(init_path) == []

        _write_file(init_path, "__all__ = sorted(NAMES, key=str)\n")
        assert tool._public_exports(init_path) == []

        _write_file(init_path, "NAMES = {'Gamma': object()}\n")
        assert tool._public_exports(init_path) == []

        _write_file(init_path, "NAMES = {}\n__all__ = sorted(NAMES)\n")
        assert tool._public_exports(init_path) == []

        _write_file(init_path, "NAMES = {123: object()}\n__all__ = sorted(NAMES)\n")
        assert tool._public_exports(init_path) == []


def test_rust_and_git_scanners_cover_file_and_failure_paths(monkeypatch: Any) -> None:
    """Rust and Git scanners should handle files, nonzero exits, and failures."""
    tool = _load_tool()
    with _tempdir() as repo:
        rust_file = repo / "bindings.rs"
        rust_file.write_text(
            "\n".join(
                [
                    'py_neuron_default!("NeuronA");',
                    '#[pyclass(name = "PyClassA")]',
                    "struct Ignored;",
                    "#[pyfunction]",
                    "pub fn exported_fn() {}",
                    "wrap_pyfunction!(wrapped_fn, module)?;",
                    "m.add_class::<AddedClass>()?;",
                ]
            ),
            encoding="utf-8",
        )
        assert tool._rust_pyo3_wrapper_names(rust_file) == [
            "AddedClass",
            "NeuronA",
            "PyClassA",
            "exported_fn",
            "wrapped_fn",
        ]

        class FailedRun:
            returncode = 1
            stdout = ""

        monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: FailedRun())
        assert tool._git_tracked_files(repo, repo=repo, suffixes=(".py",)) is None
        assert tool._git_untracked_files(repo, repo=repo, suffixes=(".py",)) is None

        def raise_os_error(*args: object, **kwargs: object) -> None:
            raise OSError("git unavailable")

        monkeypatch.setattr(subprocess, "run", raise_os_error)
        assert tool._git_tracked_files(repo, repo=repo, suffixes=(".py",)) is None
        assert tool._git_untracked_files(repo, repo=repo, suffixes=(".py",)) is None


def test_cli_main_validate_check_and_refresh_paths(capsys: Any) -> None:
    """Direct CLI entrypoint calls should cover validate, check, and refresh paths."""
    tool = _load_tool()
    with _tempdir() as repo:
        _write_portable_fixture(repo)
        config_path = Path("tools/capability_manifest.toml")
        config = tool.load_config(repo, config_path)
        json_path, _markdown_path, _readme_path = tool.refresh_outputs(
            repo,
            config=config,
        )

        assert (
            tool.main(["--repo", str(repo), "--config", str(config_path), "--check"])
            == 0
        )
        assert tool.main(["--repo", str(repo), "--config", str(config_path)]) == 0
        assert (
            tool.main(
                [
                    "--repo",
                    str(repo),
                    "--config",
                    str(config_path),
                    "--no-readme",
                ]
            )
            == 0
        )

        valid_status = tool.main(
            [
                "--repo",
                str(repo),
                "--config",
                str(config_path),
                "--validate",
                str(json_path),
            ]
        )
        assert valid_status == 0

        invalid_path = repo / "invalid_manifest.json"
        invalid_path.write_text('{"schema_version": "wrong"}\n', encoding="utf-8")
        assert tool.main(["--repo", str(repo), "--validate", str(invalid_path)]) == 1

        (repo / "docs/_generated/capability_manifest.json").write_text(
            "{}\n", encoding="utf-8"
        )
        assert (
            tool.main(["--repo", str(repo), "--config", str(config_path), "--check"])
            == 1
        )

        captured = capsys.readouterr()
        assert "Wrote" in captured.out
        assert "Refreshed" in captured.out
        assert "schema_version mismatch" in captured.out
        assert "stale generated manifest" in captured.err


@contextmanager
def _tempdir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory)


def _write_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_portable_fixture(repo: Path) -> None:
    _write_file(
        repo / "pyproject.toml",
        "\n".join(
            [
                "[project]",
                'name = "portable-project"',
                'version = "1.2.3"',
                'requires-python = ">=3.11"',
                'readme = "README.md"',
                'license = "Apache-2.0"',
                "",
                "[project.optional-dependencies]",
                'analysis = ["numpy"]',
                "",
            ]
        ),
    )
    _write_file(
        repo / "README.md",
        "\n".join(
            [
                "# Portable Project",
                "",
                "<!-- capability-snapshot:start -->",
                "stale",
                "<!-- capability-snapshot:end -->",
                "",
            ]
        ),
    )
    _write_file(
        repo / "src/portable_project/__init__.py",
        "\n".join(
            [
                '_LAZY_IMPORTS = {"PortableModel": (".models.portable", "PortableModel")}',
                "__all__ = sorted(_LAZY_IMPORTS)",
                "",
            ]
        ),
    )
    _write_file(
        repo / "src/portable_project/models/portable.py",
        "\n".join(
            [
                "class PortableModel:",
                "    pass",
                "",
                "class NestedModel:",
                "    pass",
                "",
            ]
        ),
    )
    _write_file(
        repo / "src/portable_project/generated_pb2.py",
        "class IgnoredGenerated:\n    pass\n",
    )
    _write_file(repo / "docs/api/models/portable.md", "# Portable model\n")
    _write_file(repo / "docs/internal/private.md", "# Private\n")
    _write_file(
        repo / "tests/test_portable.py",
        "def test_portable() -> None:\n    assert True\n",
    )
    _write_file(repo / ".github/workflows/ci.yml", "name: CI\non: [push]\njobs: {}\n")
    _write_file(
        repo / "engine/src/lib.rs",
        "#[pyfunction]\nfn PortableRustFn() {}\n",
    )
    _write_file(
        repo / "tools/capability_manifest.toml",
        "\n".join(
            [
                'project_label = "Portable Project"',
                'schema_version = "capability-manifest.v1"',
                'exclude_doc_parts = ["internal", "_generated"]',
                'exclude_source_parts = ["generated_pb2.py"]',
                "",
                "[paths]",
                'json_output = "docs/_generated/capability_manifest.json"',
                'markdown_output = "docs/_generated/capability_snapshot.md"',
                'package_root = "src/portable_project"',
                'model_sources = "src/portable_project"',
                'model_docs = "docs/api/models"',
                'tests_root = "tests"',
                'docs_root = "docs"',
                'workflows_root = ".github/workflows"',
                'rust_wrappers = "engine/src"',
                "",
                "[readme]",
                'path = "README.md"',
                'marker_start = "<!-- capability-snapshot:start -->"',
                'marker_end = "<!-- capability-snapshot:end -->"',
                "",
                "[labels]",
                'public_api_exports = "Portable API exports"',
            ]
        ),
    )
