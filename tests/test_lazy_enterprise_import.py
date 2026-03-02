# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Lazy Enterprise Import Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import subprocess
import sys


class TestLazyEnterprise:
    def test_core_import_does_not_load_tenant(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from director_ai.core import CoherenceScorer; "
                    "import sys; "
                    "assert 'director_ai.core.tenant' not in sys.modules, "
                    "'tenant was eagerly loaded'"
                ),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_enterprise_symbols_still_importable(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from director_ai.core import TenantRouter; "
                    "assert TenantRouter is not None"
                ),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_top_level_enterprise_lazy(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from director_ai import TenantRouter, Policy, AuditLogger; "
                    "assert TenantRouter is not None; "
                    "assert Policy is not None; "
                    "assert AuditLogger is not None"
                ),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
