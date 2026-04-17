# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Proxy facts_path security tests

"""VULN-DAI-002 regression: ``_load_facts`` must refuse ``facts_path``
that escapes ``facts_root``, including via symlinks and relative
traversal (``../``). ``facts_root=None`` preserves operator-trusted CLI
behaviour.
"""

from __future__ import annotations

import pathlib

import pytest

from director_ai.core import GroundTruthStore
from director_ai.proxy import _load_facts, create_proxy_app


def _write_facts(path: pathlib.Path, body: str = "sky: blue\n") -> None:
    path.write_text(body, encoding="utf-8")


class TestFactsRootEnforcement:
    def test_loads_when_inside_root(self, tmp_path):
        root = tmp_path / "kb"
        root.mkdir()
        facts = root / "facts.txt"
        _write_facts(facts)
        store = GroundTruthStore()
        _load_facts(store, str(facts), facts_root=str(root))
        assert store.retrieve_context("sky")

    def test_rejects_path_outside_root(self, tmp_path):
        root = tmp_path / "kb"
        root.mkdir()
        elsewhere = tmp_path / "secret.txt"
        _write_facts(elsewhere, "admin: password\n")
        store = GroundTruthStore()
        with pytest.raises(ValueError, match="outside facts_root"):
            _load_facts(store, str(elsewhere), facts_root=str(root))

    def test_rejects_dotdot_traversal(self, tmp_path):
        root = tmp_path / "kb"
        root.mkdir()
        elsewhere = tmp_path / "etc_passwd.txt"
        _write_facts(elsewhere, "root:x:0:0\n")
        traversal = root / ".." / "etc_passwd.txt"
        store = GroundTruthStore()
        with pytest.raises(ValueError, match="outside facts_root"):
            _load_facts(store, str(traversal), facts_root=str(root))

    def test_rejects_symlink_escape(self, tmp_path):
        root = tmp_path / "kb"
        root.mkdir()
        outside = tmp_path / "outside.txt"
        _write_facts(outside, "leak: yes\n")
        symlink = root / "alias.txt"
        symlink.symlink_to(outside)
        store = GroundTruthStore()
        with pytest.raises(ValueError, match="outside facts_root"):
            _load_facts(store, str(symlink), facts_root=str(root))

    def test_accepts_symlink_inside_root(self, tmp_path):
        root = tmp_path / "kb"
        root.mkdir()
        real = root / "real.txt"
        _write_facts(real)
        alias = root / "alias.txt"
        alias.symlink_to(real)
        store = GroundTruthStore()
        _load_facts(store, str(alias), facts_root=str(root))
        assert store.retrieve_context("sky")

    def test_missing_facts_raises_file_not_found(self, tmp_path):
        root = tmp_path / "kb"
        root.mkdir()
        store = GroundTruthStore()
        with pytest.raises(FileNotFoundError, match="Facts file not found"):
            _load_facts(store, str(root / "missing.txt"), facts_root=str(root))

    def test_missing_root_raises_file_not_found(self, tmp_path):
        facts = tmp_path / "facts.txt"
        _write_facts(facts)
        store = GroundTruthStore()
        with pytest.raises(FileNotFoundError, match="facts_root not found"):
            _load_facts(store, str(facts), facts_root=str(tmp_path / "no_such_dir"))

    def test_root_must_be_directory(self, tmp_path):
        not_a_dir = tmp_path / "file.txt"
        _write_facts(not_a_dir)
        facts = tmp_path / "facts.txt"
        _write_facts(facts)
        store = GroundTruthStore()
        with pytest.raises(ValueError, match="must be a directory"):
            _load_facts(store, str(facts), facts_root=str(not_a_dir))


class TestFactsRootBackwardCompat:
    def test_no_root_accepts_any_readable_path(self, tmp_path):
        facts = tmp_path / "facts.txt"
        _write_facts(facts)
        store = GroundTruthStore()
        _load_facts(store, str(facts))
        assert store.retrieve_context("sky")


class TestCreateProxyAppWiring:
    def test_proxy_constructs_with_facts_root(self, tmp_path):
        root = tmp_path / "kb"
        root.mkdir()
        facts = root / "facts.txt"
        _write_facts(facts)
        app = create_proxy_app(
            facts_path=str(facts),
            facts_root=str(root),
            upstream_url="https://test.example.com",
        )
        assert app is not None

    def test_proxy_rejects_facts_outside_root(self, tmp_path):
        root = tmp_path / "kb"
        root.mkdir()
        elsewhere = tmp_path / "secret.txt"
        _write_facts(elsewhere)
        with pytest.raises(ValueError, match="outside facts_root"):
            create_proxy_app(
                facts_path=str(elsewhere),
                facts_root=str(root),
                upstream_url="https://test.example.com",
            )
