# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — CLI verification deep coverage tests
"""Behavioural coverage for verification and diagnostics CLI branches."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import director_ai._cli_verify as verify_cli


class TestOptionalDependencyDiagnostics:
    def test_optional_module_reports_missing_import_failed_and_versionless(
        self,
        monkeypatch,
    ):
        import importlib
        import importlib.util

        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        assert verify_cli._check_optional_module("missing_pkg") == (
            False,
            "not installed",
        )

        monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())

        def fail_import(name: str):
            raise RuntimeError(f"{name} exploded")

        monkeypatch.setattr(importlib, "import_module", fail_import)
        assert verify_cli._check_optional_module("broken_pkg") == (
            False,
            "import failed: broken_pkg exploded",
        )

        monkeypatch.setattr(
            importlib,
            "import_module",
            lambda name: SimpleNamespace(),
        )
        assert verify_cli._check_optional_module("plain_pkg") == (
            True,
            "installed",
        )

    def test_stack_warnings_surface_invalid_config_and_missing_optional_deps(
        self,
        monkeypatch,
    ):
        from director_ai.core.config import DirectorConfig

        monkeypatch.setattr(
            DirectorConfig,
            "from_env",
            staticmethod(lambda: (_ for _ in ()).throw(ValueError("bad threshold"))),
        )
        assert verify_cli._stack_warnings([]) == [
            "Invalid DIRECTOR_* configuration: bad threshold"
        ]

        cfg = SimpleNamespace(
            use_nli=True,
            scorer_backend="onnx",
            onnx_path="",
            vector_backend="chroma",
        )
        monkeypatch.setattr(DirectorConfig, "from_env", staticmethod(lambda: cfg))
        monkeypatch.setattr(
            verify_cli, "_check_optional_module", lambda name: (False, "missing")
        )

        warnings = verify_cli._stack_warnings(
            [
                ("Rust kernel", False, "missing"),
                ("Docker Compose", True, "installed"),
            ]
        )

        assert warnings == [
            "DIRECTOR_USE_NLI=true but torch/transformers are missing.",
            "DIRECTOR_SCORER_BACKEND=onnx but onnxruntime is missing.",
            "DIRECTOR_SCORER_BACKEND=onnx but DIRECTOR_ONNX_PATH is empty.",
            "DIRECTOR_VECTOR_BACKEND=chroma but chromadb is missing.",
        ]


class TestLicenseCliBranches:
    def test_license_generate_requires_admin_key(self, monkeypatch, capsys):
        monkeypatch.delenv("DIRECTOR_ADMIN_KEY", raising=False)

        with pytest.raises(SystemExit) as exc_info:
            verify_cli._cmd_license(["generate"])

        assert exc_info.value.code == 1
        assert "DIRECTOR_ADMIN_KEY" in capsys.readouterr().out

    def test_license_generate_writes_requested_file(
        self,
        monkeypatch,
        tmp_path,
        capsys,
    ):
        import director_ai.core.license as license_mod

        output = tmp_path / "license.json"
        monkeypatch.setenv("DIRECTOR_ADMIN_KEY", "admin-present")
        monkeypatch.setattr(
            license_mod,
            "generate_license",
            lambda **kwargs: {
                "key": "DIR-AI-TEST-KEY",
                "tier": kwargs["tier"],
                "licensee": kwargs["licensee"],
                "email": kwargs["email"],
                "expires": "2027-01-01T00:00:00Z",
                "deployments": kwargs["deployments"],
            },
        )

        verify_cli._cmd_license(
            [
                "generate",
                "--tier",
                "pro",
                "--licensee",
                "Pilot Operator",
                "--email",
                "pilot@example.test",
                "--days",
                "45",
                "--deployments",
                "3",
                "--output",
                str(output),
            ]
        )

        data = json.loads(output.read_text(encoding="utf-8"))
        assert data["tier"] == "pro"
        assert data["licensee"] == "Pilot Operator"
        assert data["deployments"] == 3
        assert "License generated" in capsys.readouterr().out

    def test_license_validate_usage_and_exit_code(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc_info:
            verify_cli._cmd_license(["validate"])
        assert exc_info.value.code == 1
        assert "license validate <path>" in capsys.readouterr().out

        import director_ai.core.license as license_mod

        monkeypatch.setattr(
            license_mod,
            "validate_file",
            lambda path: SimpleNamespace(
                valid=False,
                tier="community",
                licensee="",
                message=f"invalid: {path}",
            ),
        )
        with pytest.raises(SystemExit) as exc_info:
            verify_cli._cmd_license(["validate", "bad-license.json"])

        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "Valid:    False" in out
        assert "invalid: bad-license.json" in out

    def test_license_unknown_subcommand_exits(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            verify_cli._cmd_license(["rotate"])

        assert exc_info.value.code == 1
        assert "Unknown license subcommand" in capsys.readouterr().out


class TestComplianceCliBranches:
    def test_compliance_missing_database_exits(self, tmp_path, capsys):
        missing = tmp_path / "missing.db"

        with pytest.raises(SystemExit) as exc_info:
            verify_cli._cmd_compliance(["status", "--db", str(missing)])

        assert exc_info.value.code == 1
        assert "Audit database not found" in capsys.readouterr().out

    def test_compliance_json_report_passes_since_and_until(
        self,
        monkeypatch,
        tmp_path,
        capsys,
    ):
        import director_ai.compliance.audit_log as audit_mod
        import director_ai.compliance.reporter as reporter_mod

        db_file = tmp_path / "audit.db"
        db_file.touch()
        calls: dict[str, tuple[float | None, float | None]] = {}

        class FakeLog:
            def __init__(self, path: str) -> None:
                self.path = path

            def close(self) -> None:
                calls["closed"] = (None, None)

        class FakeReporter:
            def __init__(self, log: FakeLog) -> None:
                self.log = log

            def generate_report(self, *, since=None, until=None):
                calls["window"] = (since, until)
                return SimpleNamespace(
                    total_interactions=12,
                    overall_hallucination_rate=0.125,
                    overall_hallucination_rate_ci=(0.1, 0.2),
                    avg_score=0.91,
                    drift_detected=True,
                    incident_count=2,
                )

        monkeypatch.setattr(audit_mod, "AuditLog", FakeLog)
        monkeypatch.setattr(reporter_mod, "ComplianceReporter", FakeReporter)

        verify_cli._cmd_compliance(
            [
                "report",
                "--db",
                str(db_file),
                "--since",
                "10.5",
                "--until",
                "20.25",
                "--format",
                "json",
            ]
        )

        payload = json.loads(capsys.readouterr().out)
        assert calls["window"] == (10.5, 20.25)
        assert calls["closed"] == (None, None)
        assert payload["total_interactions"] == 12
        assert payload["drift_detected"] is True


class TestCostReportCliBranches:
    def test_cost_report_disabled_and_missing_analyser_exit(self, monkeypatch, capsys):
        from director_ai.core.config import DirectorConfig

        monkeypatch.setattr(
            DirectorConfig,
            "from_env",
            staticmethod(lambda: SimpleNamespace(cost_tracking_enabled=False)),
        )
        with pytest.raises(SystemExit) as exc_info:
            verify_cli._cmd_cost_report([])
        assert exc_info.value.code == 1
        assert "Cost tracking is disabled" in capsys.readouterr().out

        monkeypatch.setattr(
            DirectorConfig,
            "from_env",
            staticmethod(
                lambda: SimpleNamespace(
                    cost_tracking_enabled=True,
                    build_scorer=lambda: SimpleNamespace(),
                )
            ),
        )
        with pytest.raises(SystemExit) as exc_info:
            verify_cli._cmd_cost_report([])
        assert exc_info.value.code == 1
        assert "No CostAnalyser" in capsys.readouterr().out

    def test_cost_report_json_html_and_text_modes(self, monkeypatch, capsys):
        from director_ai.compliance import report_templates
        from director_ai.core.config import DirectorConfig

        report = {
            "currency": "CHF",
            "total_cost": 1.25,
            "total_tokens": 1234,
            "models": {
                "local": {
                    "call_count": 2,
                    "total_tokens": 1234,
                    "estimated_cost": 1.25,
                }
            },
        }
        analyser = SimpleNamespace(report=lambda: report)
        scorer = SimpleNamespace(_cost_analyser=analyser)
        monkeypatch.setattr(
            DirectorConfig,
            "from_env",
            staticmethod(
                lambda: SimpleNamespace(
                    cost_tracking_enabled=True,
                    build_scorer=lambda: scorer,
                )
            ),
        )
        monkeypatch.setattr(
            report_templates,
            "render_cost_html",
            lambda payload: f"<html>{payload['currency']}</html>",
        )

        verify_cli._cmd_cost_report(["--format", "json"])
        assert json.loads(capsys.readouterr().out)["total_tokens"] == 1234

        verify_cli._cmd_cost_report(["--format", "html"])
        assert "<html>CHF</html>" in capsys.readouterr().out

        verify_cli._cmd_cost_report([])
        out = capsys.readouterr().out
        assert "Total cost: CHF 1.250000" in out
        assert "local: 2 calls" in out


class TestSafetyDashboardFallback:
    def test_safety_dashboard_gradio_import_error_prints_text_hint(
        self,
        monkeypatch,
        capsys,
    ):
        import director_ai.ui.safety_dashboard as dashboard_mod

        def raise_import_error(*, port: int, share: bool) -> None:
            raise ImportError("gradio absent")

        monkeypatch.setattr(
            dashboard_mod,
            "launch_safety_dashboard",
            raise_import_error,
        )

        verify_cli._cmd_safety_dashboard(["--port", "9000", "--share"])

        assert "Gradio not installed" in capsys.readouterr().out
