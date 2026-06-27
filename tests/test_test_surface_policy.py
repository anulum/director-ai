# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Behavioral tests for module-specific test-surface policy enforcement."""

from __future__ import annotations

from pathlib import Path

import pytest

import tools.check_test_surface_policy as policy
from tools.check_test_surface_policy import (
    SurfaceClassification,
    find_forbidden_test_surfaces,
    find_unclassified_mock_surfaces,
    validate_classifications,
)
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


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


def test_policy_rejects_unclassified_mock_surface(tmp_path: Path) -> None:
    path = tmp_path / "tests/test_adapter_contract.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "from unittest.mock import MagicMock",
                "",
                "def test_contract():",
                "    assert MagicMock().called is False",
            ]
        ),
        encoding="utf-8",
    )

    offenders = find_unclassified_mock_surfaces(tmp_path, classifications={})

    assert offenders == [(Path("tests/test_adapter_contract.py"), "unittest.mock")]


def test_policy_rejects_unclassified_private_helper_surface(tmp_path: Path) -> None:
    path = tmp_path / "tests/test_private_contract.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "from director_ai.core.scoring.nli import _normalise_text",
                "",
                "def test_contract():",
                "    assert _normalise_text('A') == 'a'",
            ]
        ),
        encoding="utf-8",
    )

    offenders = find_unclassified_mock_surfaces(tmp_path, classifications={})

    assert offenders == [
        (Path("tests/test_private_contract.py"), "private-director-ai-import")
    ]


def test_policy_accepts_classified_known_violation(tmp_path: Path) -> None:
    path = tmp_path / "tests/test_known_contract.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "from unittest.mock import patch\n\ndef test_contract():\n    assert patch\n",
        encoding="utf-8",
    )
    classifications = {
        "tests/test_known_contract.py": SurfaceClassification(
            classification="violation",
            category="external SDK adapter fake",
        )
    }

    assert (
        find_unclassified_mock_surfaces(
            tmp_path,
            classifications=classifications,
        )
        == []
    )


def test_policy_accepts_marked_protocol_fake_with_companion(tmp_path: Path) -> None:
    _write_test(tmp_path, "tests/test_adapter_real_surface.py")
    path = tmp_path / "tests/test_adapter_contract.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# test-surface: approved-protocol-fake",
                "# real-surface-companion: tests/test_adapter_real_surface.py",
                "import sys",
                "",
                "def test_contract():",
                '    assert "missing_sdk" not in sys.modules',
            ]
        ),
        encoding="utf-8",
    )

    assert find_unclassified_mock_surfaces(tmp_path, classifications={}) == []


def test_policy_reports_missing_inline_companion_marker(tmp_path: Path) -> None:
    path = tmp_path / "tests/test_adapter_contract.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# test-surface: approved-protocol-fake",
                "import sys",
                "",
                "def test_contract():",
                '    assert "missing_sdk" not in sys.modules',
            ]
        ),
        encoding="utf-8",
    )

    assert find_unclassified_mock_surfaces(tmp_path, classifications={}) == [
        (
            Path("tests/test_adapter_contract.py"),
            "missing real-surface-companion marker",
        )
    ]


def test_policy_reports_missing_inline_companion_file(tmp_path: Path) -> None:
    path = tmp_path / "tests/test_adapter_contract.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# test-surface: unit-guard-with-companion",
                "# real-surface-companion: tests/test_adapter_real_surface.py",
                "import sys",
                "",
                "def test_contract():",
                '    assert "missing_sdk" not in sys.modules',
            ]
        ),
        encoding="utf-8",
    )

    assert find_unclassified_mock_surfaces(tmp_path, classifications={}) == [
        (
            Path("tests/test_adapter_contract.py"),
            "missing companion tests/test_adapter_real_surface.py",
        )
    ]


def test_policy_handles_missing_tests_directory(tmp_path: Path) -> None:
    assert find_forbidden_test_surfaces(tmp_path) == []
    assert find_unclassified_mock_surfaces(tmp_path, classifications={}) == []


def test_policy_falls_back_when_tokenization_fails() -> None:
    assert policy._code_without_literals("'") == "'"


def test_policy_validates_classification_manifest() -> None:
    classifications = {
        "tests/test_bad_kind.py": SurfaceClassification(
            classification="approved",
            category="module/workflow fake requiring review",
        ),
        "tests/test_blank_category.py": SurfaceClassification(
            classification="violation",
            category=" ",
        ),
    }

    assert validate_classifications(classifications) == [
        "tests/test_bad_kind.py: invalid classification 'approved'",
        "tests/test_blank_category.py: category must not be blank",
    ]


def test_knowledge_api_unit_guard_has_real_surface_companion() -> None:
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_knowledge_api.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_knowledge_api_real_surface.py" in category


def test_middleware_unit_guard_has_real_surface_companion() -> None:
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_middleware.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_middleware_real_surface.py" in category


def test_policy_main_returns_success_for_clean_tree(tmp_path: Path) -> None:
    _write_test(tmp_path, "tests/test_real_surface.py")

    assert policy.main(["--root", str(tmp_path)]) == 0


def test_policy_main_reports_forbidden_and_unclassified_surfaces(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_test(tmp_path, "tests/test_final_push.py")
    path = tmp_path / "tests/test_adapter_contract.py"
    path.write_text(
        "from unittest.mock import MagicMock\n\ndef test_contract():\n    assert MagicMock\n",
        encoding="utf-8",
    )

    assert policy.main(["--root", str(tmp_path)]) == 1

    captured = capsys.readouterr()
    assert "Forbidden bucket-style test file names detected" in captured.err
    assert "Unclassified mock/sys.modules test surfaces detected" in captured.err
    assert "tests/test_final_push.py: token 'final'" in captured.err
    assert "tests/test_adapter_contract.py: unittest.mock" in captured.err


def test_policy_main_reports_invalid_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        policy,
        "_default_classifications",
        lambda: {
            "tests/test_bad.py": SurfaceClassification(
                classification="bad",
                category="module/workflow fake requiring review",
            )
        },
    )

    assert policy.main(["--root", str(tmp_path)]) == 1

    captured = capsys.readouterr()
    assert "Invalid test-surface classification manifest" in captured.err
    assert "tests/test_bad.py: invalid classification 'bad'" in captured.err
