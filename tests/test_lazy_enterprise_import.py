# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Enterprise Import Tests
"""Multi-angle tests for enterprise module lazy loading.

Covers: core import isolation (tenant/policy/audit not loaded eagerly),
enterprise package importability, old import path deprecation errors,
parametrised module checking, and pipeline documentation.

Uses subprocess isolation to ensure clean sys.modules state.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def _run_python(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )


# ── Lazy import isolation ─────────────────────────────────────────


class TestLazyImportIsolation:
    """Core import must not eagerly load enterprise modules."""

    @pytest.mark.parametrize(
        "module",
        [
            "director_ai.core.tenant",
            "director_ai.enterprise.tenant",
            "director_ai.enterprise.policy",
            "director_ai.enterprise.audit",
        ],
    )
    def test_core_import_does_not_load_enterprise(self, module):
        result = _run_python(
            f"from director_ai.core import CoherenceScorer; "
            f"import sys; "
            f"assert '{module}' not in sys.modules, "
            f"'{module} was eagerly loaded'"
        )
        assert result.returncode == 0, result.stderr


# ── Enterprise importability ─────────────────────────────────────


class TestEnterpriseImportable:
    """Enterprise package must be importable when deps available."""

    def test_enterprise_package_importable(self):
        result = _run_python(
            "from director_ai.enterprise import TenantRouter, Policy, AuditLogger; "
            "assert TenantRouter is not None; "
            "assert Policy is not None; "
            "assert AuditLogger is not None"
        )
        assert result.returncode == 0, result.stderr

    @pytest.mark.parametrize(
        "cls",
        [
            "TenantRouter",
            "Policy",
            "AuditLogger",
        ],
    )
    def test_individual_class_importable(self, cls):
        result = _run_python(
            f"from director_ai.enterprise import {cls}; assert {cls} is not None"
        )
        assert result.returncode == 0, result.stderr


# ── Old import path deprecation ──────────────────────────────────


class TestOldImportPathDeprecation:
    """Old import paths must raise ImportError with migration hint."""

    def test_old_root_path_raises(self):
        result = _run_python(
            "try:\n"
            "    from director_ai import TenantRouter\n"
            "    raise AssertionError('should have raised ImportError')\n"
            "except ImportError as e:\n"
            "    assert 'director_ai.enterprise' in str(e)\n"
        )
        assert result.returncode == 0, result.stderr

    def test_old_core_path_raises(self):
        result = _run_python(
            "try:\n"
            "    from director_ai.core import TenantRouter\n"
            "    raise AssertionError('should have raised ImportError')\n"
            "except ImportError as e:\n"
            "    assert 'director_ai.enterprise' in str(e)\n"
        )
        assert result.returncode == 0, result.stderr

    @pytest.mark.parametrize(
        "old_path,cls",
        [
            ("director_ai", "TenantRouter"),
            ("director_ai", "Policy"),
            ("director_ai.core", "TenantRouter"),
        ],
    )
    def test_parametrised_old_paths(self, old_path, cls):
        result = _run_python(
            f"try:\n"
            f"    from {old_path} import {cls}\n"
            f"    raise AssertionError('should have raised ImportError')\n"
            f"except ImportError:\n"
            f"    pass\n"
        )
        assert result.returncode == 0, result.stderr


# ── Pipeline documentation ───────────────────────────────────────


class TestEnterprisePipelineDoc:
    """Enterprise module must integrate with core scorer."""

    def test_core_scorer_available_without_enterprise(self):
        result = _run_python(
            "from director_ai.core import CoherenceScorer; "
            "s = CoherenceScorer(use_nli=False); "
            "ok, sc = s.review('test', 'test'); "
            "assert isinstance(ok, bool)"
        )
        assert result.returncode == 0, result.stderr
