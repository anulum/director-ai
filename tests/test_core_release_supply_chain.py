# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Core release supply-chain contract tests

"""Repository-level contract tests for the single-build Core release path."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, cast

import yaml

ROOT = Path(__file__).resolve().parents[1]
RELEASE_PATH = ROOT / ".github" / "workflows" / "release.yml"
PUBLISH_PATH = ROOT / ".github" / "workflows" / "publish.yml"


def _workflow(path: Path) -> dict[str, Any]:
    """Load one production workflow from the repository."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return cast("dict[str, Any]", loaded)


def _step(job: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a named workflow step from a production job."""
    return next(step for step in job["steps"] if step.get("name") == name)


def _download_names(job: dict[str, Any]) -> set[str]:
    """Return the explicitly named workflow artefacts downloaded by a job."""
    return {
        step["with"]["name"]
        for step in job["steps"]
        if str(step.get("uses", "")).startswith("actions/download-artifact@")
    }


def test_release_workflow_creates_metadata_without_building_core() -> None:
    """The tag workflow must not create a second Core distribution."""
    workflow = _workflow(RELEASE_PATH)
    release = workflow["jobs"]["release"]
    create = _step(release, "Create GitHub Release")
    text = RELEASE_PATH.read_text(encoding="utf-8")

    assert "python -m build" not in text
    assert "actions/setup-python@" not in text
    assert "files" not in create.get("with", {})
    assert workflow["jobs"]["dispatch-publish"]["needs"] == "release"


def test_publish_workflow_builds_core_once_from_the_release_tag() -> None:
    """Only the publish workflow may build the Core wheel and sdist."""
    workflow = _workflow(PUBLISH_PATH)
    build = workflow["jobs"]["build"]
    text = PUBLISH_PATH.read_text(encoding="utf-8")
    checkout = next(
        step
        for step in build["steps"]
        if str(step.get("uses", "")).startswith("actions/checkout@")
    )

    assert text.count("python -m build") == 1
    assert checkout["with"]["ref"] == "${{ env.RELEASE_TAG }}"
    assert 'sha256sum -- "${distributions[@]}" > dist-sha256.txt' in text
    assert _step(build, "Upload dist")["with"]["name"] == "dist"
    assert (
        _step(build, "Upload distribution checksums")["with"]["name"] == "dist-sha256"
    )


def test_checksum_step_executes_against_the_distribution_directory(
    tmp_path: Path,
) -> None:
    """The checked-in checksum command accepts exactly one wheel and sdist."""
    build = _workflow(PUBLISH_PATH)["jobs"]["build"]
    command = _step(build, "Record distribution checksums")["run"]
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "director_ai-3.21.0-py3-none-any.whl").write_bytes(b"wheel")
    (dist / "director_ai-3.21.0.tar.gz").write_bytes(b"sdist")

    accepted = subprocess.run(
        ["bash", "-c", command],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert accepted.returncode == 0, accepted.stderr
    manifest = (tmp_path / "dist-sha256.txt").read_text(encoding="utf-8")
    assert "dist/director_ai-3.21.0-py3-none-any.whl" in manifest
    assert "dist/director_ai-3.21.0.tar.gz" in manifest

    (dist / "director_ai_duplicate-3.21.0-py3-none-any.whl").write_bytes(b"extra")
    rejected = subprocess.run(
        ["bash", "-c", command],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert rejected.returncode != 0
    assert "Expected one Core sdist and one Core wheel, found 3" in rejected.stdout


def test_release_consumers_verify_the_single_build_checksums() -> None:
    """Signing, provenance, publication, and attachment consume one build."""
    jobs = _workflow(PUBLISH_PATH)["jobs"]

    for job_name in ("sign", "provenance", "publish"):
        job = jobs[job_name]
        assert {"dist", "dist-sha256"} <= _download_names(job)
        assert _step(job, "Verify distribution checksums").get("run") == (
            "sha256sum --check dist-sha256.txt"
        )

    attach = jobs["attach-artifacts"]
    assert {"dist", "dist-sha256"} <= _download_names(attach)
    assert (
        "sha256sum --check dist-sha256.txt" in _step(attach, "Attach to release")["run"]
    )

    assert set(jobs["publish"]["needs"]) == {"build", "sign", "provenance"}
    assert set(jobs["attach-artifacts"]["needs"]) == {
        "build",
        "sign",
        "provenance",
        "publish",
    }
    assert jobs["sign"]["permissions"] == {"id-token": "write"}


def test_release_attachment_is_exact_and_non_overwriting() -> None:
    """The GitHub Release receives the published files without replacement."""
    attach = _workflow(PUBLISH_PATH)["jobs"]["attach-artifacts"]
    command = _step(attach, "Attach to release")["run"]

    assert 'gh release upload "$RELEASE_TAG"' in command
    for path in (
        "dist/*",
        "dist-sha256.txt",
        "sbom.json",
        "sigstore/*.sigstore.json",
    ):
        assert path in command
    assert '--repo "$GITHUB_REPOSITORY"' in command
    assert "--clobber" not in command


def test_release_workflows_pin_every_external_action() -> None:
    """Every external action reference remains pinned to a full commit SHA."""
    for path in (RELEASE_PATH, PUBLISH_PATH):
        action_refs = re.findall(
            r"uses: [^@\n]+@([^\s#]+)", path.read_text(encoding="utf-8")
        )
        assert action_refs
        assert all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in action_refs)
