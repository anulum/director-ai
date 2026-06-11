# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Behavioral tests for module-specific test-surface policy enforcement."""

from __future__ import annotations

from pathlib import Path

from tools.check_test_surface_policy import find_forbidden_test_surfaces


def _write_test(root: Path, relative: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("def test_contract():\n    assert True\n", encoding="utf-8")


def test_policy_rejects_bucket_style_test_file_names(tmp_path: Path) -> None:
    _write_test(tmp_path, "tests/test_hardware_runner.py")
    _write_test(tmp_path, "tests/test_final_push.py")
    _write_test(tmp_path, "tests/new_modules/test_crypto.py")

    offenders = find_forbidden_test_surfaces(tmp_path)

    assert offenders == [
        (Path("tests/new_modules/test_crypto.py"), "new_modules"),
        (Path("tests/test_final_push.py"), "final"),
        (Path("tests/test_final_push.py"), "push"),
    ]


def test_policy_uses_tokens_not_substrings(tmp_path: Path) -> None:
    _write_test(tmp_path, "tests/test_physical_grounding_evaluator.py")
    _write_test(tmp_path, "tests/test_pushdown_automaton.py")

    assert find_forbidden_test_surfaces(tmp_path) == []
