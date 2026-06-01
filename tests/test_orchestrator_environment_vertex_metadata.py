# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Vertex benchmark metadata tests

"""Regression tests for benchmark environment capture in gitless containers."""

from __future__ import annotations

import subprocess

import benchmarks.orchestrator.environment as environment_module
from benchmarks.orchestrator.environment import capture_environment


def test_vertex_environment_uses_explicit_git_metadata(monkeypatch):
    monkeypatch.setenv(
        "DIRECTOR_GIT_COMMIT",
        "87edd82a2286982e0724379135b8ac2d906a16b7",
    )
    monkeypatch.setenv("DIRECTOR_GIT_BRANCH", "main")
    monkeypatch.setenv("DIRECTOR_RUN_ENV", "vertex")

    env = capture_environment()

    assert env.git_commit == "87edd82a2286982e0724379135b8ac2d906a16b7"
    assert env.git_branch == "main"
    assert env.git_dirty is False
    assert env.runner == "vertex"


def test_vertex_environment_defaults_branch_for_detached_metadata(monkeypatch):
    monkeypatch.setenv(
        "DIRECTOR_GIT_COMMIT",
        "87edd82a2286982e0724379135b8ac2d906a16b7",
    )
    monkeypatch.delenv("DIRECTOR_GIT_BRANCH", raising=False)

    env = capture_environment(runner="vertex")

    assert env.git_commit == "87edd82a2286982e0724379135b8ac2d906a16b7"
    assert env.git_branch == "detached"


def test_environment_keeps_commit_when_git_status_fails(monkeypatch):
    calls: list[tuple[str, ...]] = []

    def fake_check_output(cmd, **kwargs):
        del kwargs
        calls.append(tuple(cmd))
        if cmd == ["git", "rev-parse", "HEAD"]:
            return "87edd82a2286982e0724379135b8ac2d906a16b7\n"
        if cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return "main\n"
        if cmd == ["git", "status", "--porcelain"]:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=5)
        if cmd == [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]:
            raise FileNotFoundError("nvidia-smi")
        raise AssertionError(f"unexpected command: {cmd!r}")

    monkeypatch.setattr(
        environment_module.subprocess, "check_output", fake_check_output
    )

    env = capture_environment(runner="ci")

    assert env.git_commit == "87edd82a2286982e0724379135b8ac2d906a16b7"
    assert env.git_branch == "main"
    assert env.git_dirty is True
    assert ("git", "status", "--porcelain") in calls
