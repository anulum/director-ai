# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Vertex benchmark metadata tests

"""Regression tests for benchmark environment capture in gitless containers."""

from __future__ import annotations

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
