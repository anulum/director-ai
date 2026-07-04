# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real public-surface coverage for lazy enterprise import routing."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


def _enterprise_module() -> ModuleType:
    """Return the enterprise package through normal import resolution."""
    return importlib.import_module("director_ai.enterprise")


def _enterprise_resolver(module: ModuleType) -> Callable[[str], object]:
    """Return the package-level lazy symbol resolver."""
    return cast("Callable[[str], object]", module.__dict__["__getattr__"])


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    """Run a Python snippet in a fresh interpreter and capture its result."""
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=90,
    )


def test_lazy_enterprise_import_unit_guard_declares_this_companion() -> None:
    """The lazy enterprise import unit guard should declare this companion."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_lazy_enterprise_import.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_lazy_enterprise_import_real_surface.py" in category


def test_enterprise_package_resolves_known_symbols_and_rejects_unknown() -> None:
    """The enterprise package facade should expose known symbols only."""
    enterprise = _enterprise_module()
    resolve = _enterprise_resolver(enterprise)

    tenant_router = cast("type[object]", resolve("TenantRouter"))
    policy = cast("type[object]", resolve("Policy"))
    audit_logger = cast("type[object]", resolve("AuditLogger"))

    assert tenant_router.__module__ == "director_ai.core.tenant"
    assert policy.__module__ == "director_ai.core.safety.policy"
    assert audit_logger.__module__ == "director_ai.core.safety.audit"
    with pytest.raises(AttributeError, match="NoSuchEnterpriseExport"):
        resolve("NoSuchEnterpriseExport")


def test_enterprise_public_imports_drive_tenant_policy_and_audit_surfaces(
    tmp_path: Path,
) -> None:
    """Enterprise lazy imports should run tenant, policy, and audit workflows."""
    audit_path = tmp_path / "audit.jsonl"
    result = _run_python(
        textwrap.dedent(
            f"""
            import json
            from pathlib import Path

            from director_ai.enterprise import AuditLogger, Policy, TenantRouter

            router = TenantRouter()
            router.add_fact(
                "tenant-alpha",
                "retention",
                "Tenant alpha requires signed retention approval.",
            )
            policy = Policy(forbidden=["ignore previous instructions"])
            violations = policy.check("Please ignore previous instructions.")
            audit = AuditLogger(
                path=Path({str(audit_path)!r}),
                hmac_secret="stable-test-secret",
            )
            entry = audit.log_review(
                query="retention?",
                response="Signed retention approval is required.",
                approved=not violations,
                score=0.91,
                policy_violations=[violation.rule for violation in violations],
                tenant_id="tenant-alpha",
            )
            chain_ok, first_bad_index = audit.verify_chain()
            print(
                json.dumps(
                    {{
                        "audit_chain_ok": chain_ok,
                        "audit_entry_tenant": entry.tenant_id,
                        "audit_first_bad_index": first_bad_index,
                        "audit_policy_violations": entry.policy_violations,
                        "audit_response_length": entry.response_length,
                        "fact": router.get_store("tenant-alpha").facts["retention"],
                        "modules": {{
                            "AuditLogger": AuditLogger.__module__,
                            "Policy": Policy.__module__,
                            "TenantRouter": TenantRouter.__module__,
                        }},
                        "tenant_ids": router.tenant_ids,
                    }},
                    sort_keys=True,
                )
            )
            """
        )
    )

    assert result.returncode == 0, result.stderr
    payload = cast("dict[str, object]", json.loads(result.stdout))

    assert payload["audit_chain_ok"] is True
    assert payload["audit_entry_tenant"] == "tenant-alpha"
    assert payload["audit_first_bad_index"] is None
    assert payload["audit_policy_violations"] == ["forbidden"]
    assert payload["audit_response_length"] == len(
        "Signed retention approval is required."
    )
    assert payload["fact"] == "Tenant alpha requires signed retention approval."
    assert payload["modules"] == {
        "AuditLogger": "director_ai.core.safety.audit",
        "Policy": "director_ai.core.safety.policy",
        "TenantRouter": "director_ai.core.tenant",
    }
    assert payload["tenant_ids"] == ["tenant-alpha"]


def test_root_keeps_current_enterprise_exports_and_moved_symbol_guidance() -> None:
    """Root exports should expose current helpers and reject moved symbols."""
    result = _run_python(
        textwrap.dedent(
            """
            import json

            import director_ai
            from director_ai import ContentModerator, CustomRule, RulesDslError
            from director_ai.enterprise import (
                ContentModerator as EnterpriseContentModerator,
                CustomRule as EnterpriseCustomRule,
                RulesDslError as EnterpriseRulesDslError,
            )

            try:
                from director_ai import TenantRouter
            except ImportError as exc:
                root_hint = str(exc)
            else:
                raise AssertionError(f"unexpected root TenantRouter: {TenantRouter!r}")

            try:
                from director_ai.core import TenantRouter as CoreTenantRouter
            except ImportError as exc:
                core_hint = str(exc)
            else:
                raise AssertionError(
                    f"unexpected core TenantRouter: {CoreTenantRouter!r}"
                )

            print(
                json.dumps(
                    {
                        "current_exports": {
                            "ContentModerator": (
                                ContentModerator is EnterpriseContentModerator
                            ),
                            "CustomRule": CustomRule is EnterpriseCustomRule,
                            "RulesDslError": RulesDslError is EnterpriseRulesDslError,
                        },
                        "listed_exports": {
                            name: name in director_ai.__all__
                            for name in (
                                "ContentModerator",
                                "CustomRule",
                                "RulesDslError",
                            )
                        },
                        "core_hint": core_hint,
                        "root_hint": root_hint,
                    },
                    sort_keys=True,
                )
            )
            """
        )
    )

    assert result.returncode == 0, result.stderr
    payload = cast("dict[str, object]", json.loads(result.stdout))

    assert payload["current_exports"] == {
        "ContentModerator": True,
        "CustomRule": True,
        "RulesDslError": True,
    }
    assert payload["listed_exports"] == {
        "ContentModerator": True,
        "CustomRule": True,
        "RulesDslError": True,
    }
    assert "from director_ai.enterprise import TenantRouter" in cast(
        str,
        payload["root_hint"],
    )
    assert "from director_ai.enterprise import TenantRouter" in cast(
        str,
        payload["core_hint"],
    )
