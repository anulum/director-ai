# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — CLI verification deep coverage tests
"""Behavioural coverage for verification and diagnostics CLI branches."""

from __future__ import annotations

import json
import sys
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

    def test_stack_warnings_surface_model_revision_health(self, monkeypatch):
        from director_ai.core.config import DirectorConfig

        cfg = SimpleNamespace(
            use_nli=False,
            scorer_backend="lite",
            onnx_path="",
            vector_backend="memory",
            model_revision_health=lambda: {
                "ok": False,
                "checks": {
                    "nli": {
                        "status": "error",
                        "detail": "remote model requires an immutable revision",
                    }
                },
            },
        )
        monkeypatch.setattr(DirectorConfig, "from_env", staticmethod(lambda: cfg))
        monkeypatch.setattr(
            verify_cli, "_check_optional_module", lambda name: (True, "installed")
        )

        warnings = verify_cli._stack_warnings([("Rust kernel", True, "installed")])

        assert warnings == [
            "Model revision health failed for nli: remote model requires an immutable revision"
        ]

        cfg.model_revision_health = lambda: {
            "ok": False,
            "checks": {"nli": {"status": "warning", "detail": "advisory only"}},
        }
        assert verify_cli._stack_warnings([("Rust kernel", True, "installed")]) == []

        cfg.model_revision_health = lambda: {"ok": True, "checks": {}}
        assert verify_cli._stack_warnings([("Rust kernel", True, "installed")]) == []

    def test_stack_status_reports_optional_runtime_tools(self, monkeypatch):
        import importlib.util
        import shutil

        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name: object() if name == "backfire_kernel" else None,
        )
        monkeypatch.setattr(
            shutil,
            "which",
            lambda name: "/usr/bin/" + name if name in {"docker", "lake"} else None,
        )

        statuses = {name: ok for name, ok, _detail in verify_cli._stack_status()}

        assert statuses == {
            "Python-only core": True,
            "Rust kernel": True,
            "Docker Compose": True,
            "Go gateway": False,
            "Julia tuner": False,
            "Lean verifier": True,
        }

    def test_stack_warnings_surface_missing_rust_kernel(self, monkeypatch):
        from director_ai.core.config import DirectorConfig

        cfg = SimpleNamespace(
            use_nli=False,
            scorer_backend="rust",
            onnx_path="",
            vector_backend="memory",
        )
        monkeypatch.setattr(DirectorConfig, "from_env", staticmethod(lambda: cfg))
        monkeypatch.setattr(
            verify_cli, "_check_optional_module", lambda name: (True, "installed")
        )

        warnings = verify_cli._stack_warnings([("Rust kernel", False, "missing")])

        assert warnings == [
            "DIRECTOR_SCORER_BACKEND=rust but backfire_kernel is missing."
        ]


class TestDoctorCliBranches:
    def test_doctor_prints_dependency_stack_and_warnings(self, monkeypatch, capsys):
        torch_mod = SimpleNamespace(
            __version__="2.6.0",
            cuda=SimpleNamespace(is_available=lambda: False),
        )
        ort_mod = SimpleNamespace(
            __version__="1.20.0",
            get_available_providers=lambda: ["CPUExecutionProvider"],
        )
        monkeypatch.setitem(sys.modules, "torch", torch_mod)
        monkeypatch.setitem(sys.modules, "onnxruntime", ort_mod)
        monkeypatch.setattr(
            verify_cli,
            "_check_optional_module",
            lambda name: (
                name != "slowapi",
                "installed" if name != "slowapi" else "not installed",
            ),
        )
        monkeypatch.setattr(
            "director_ai.core.scoring.nli.nli_available",
            lambda: True,
        )
        monkeypatch.setattr(
            verify_cli,
            "_stack_status",
            lambda: [
                ("Python-only core", True, "supported default"),
                ("Rust kernel", False, "optional backfire_kernel"),
            ],
        )
        monkeypatch.setattr(
            verify_cli,
            "_stack_warnings",
            lambda stack: [
                "DIRECTOR_SCORER_BACKEND=rust but backfire_kernel is missing."
            ],
        )

        verify_cli._cmd_doctor([])

        out = capsys.readouterr().out
        assert "director-ai" in out
        assert "[+] torch: 2.6.0 (CUDA: False)" in out
        assert "[+] onnxruntime: 1.20.0 (CPUExecutionProvider)" in out
        assert "[-] slowapi: not installed" in out
        assert "Runtime stack:" in out
        assert "DIRECTOR_SCORER_BACKEND=rust" in out

    def test_doctor_reports_nli_import_failure(self, monkeypatch, capsys):
        monkeypatch.setattr(
            verify_cli,
            "_check_optional_module",
            lambda name: (False, "not installed"),
        )

        def fail_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "director_ai.core.scoring.nli":
                raise RuntimeError("nli backend exploded")
            return original_import(name, globals, locals, fromlist, level)

        original_import = __import__
        monkeypatch.setattr("builtins.__import__", fail_import)
        monkeypatch.setattr(verify_cli, "_stack_status", lambda: [])
        monkeypatch.setattr(verify_cli, "_stack_warnings", lambda stack: [])

        verify_cli._cmd_doctor([])

        out = capsys.readouterr().out
        assert "NLI model ready: nli backend exploded" in out


class TestLicenseCliBranches:
    def test_license_status_prints_loaded_license_metadata(self, monkeypatch, capsys):
        import director_ai.core.license as license_mod

        monkeypatch.setattr(
            license_mod,
            "load_license",
            lambda: SimpleNamespace(
                tier="enterprise",
                valid=True,
                licensee="Pilot Tenant",
                expires="2027-06-01",
                key="DIR-AI-ENTERPRISE-TEST-KEY",
                message="active",
            ),
        )

        verify_cli._cmd_license([])

        out = capsys.readouterr().out
        assert "Tier:     enterprise" in out
        assert "Valid:    True" in out
        assert "Licensee: Pilot Tenant" in out
        assert "Expires:  2027-06-01" in out
        assert "Key:      DIR-AI-ENTERPRISE-TE..." in out
        assert "Message:  active" in out

        monkeypatch.setattr(
            license_mod,
            "load_license",
            lambda: SimpleNamespace(
                tier="community",
                valid=False,
                licensee="",
                expires=None,
                key="",
                message="missing",
            ),
        )

        verify_cli._cmd_license(["status"])

        out = capsys.readouterr().out
        assert "Licensee: (community)" in out
        assert "Message:  missing" in out
        assert "Expires:" not in out
        assert "Key:" not in out

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

    def test_license_polar_env_reports_readiness(self, monkeypatch, capsys):
        report = SimpleNamespace(
            ready=False,
            errors=["DIRECTOR_LICENSE_KEY is not configured"],
            warnings=["DIRECTOR_AI_POLAR_WEBHOOK_SECRET is not configured"],
        )
        monkeypatch.setattr(
            "director_ai.core.polar_license.validate_polar_deployment_env",
            lambda: report,
        )

        with pytest.raises(SystemExit) as exc_info:
            verify_cli._cmd_license(["polar-env"])

        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "Ready:    False" in out
        assert "DIRECTOR_LICENSE_KEY is not configured" in out
        assert "DIRECTOR_AI_POLAR_WEBHOOK_SECRET is not configured" in out

        ready_report = SimpleNamespace(ready=True, errors=[], warnings=[])
        monkeypatch.setattr(
            "director_ai.core.polar_license.validate_polar_deployment_env",
            lambda: ready_report,
        )
        with pytest.raises(SystemExit) as exc_info:
            verify_cli._cmd_license(["polar-env"])
        assert exc_info.value.code == 0
        assert capsys.readouterr().out == "Ready:    True\n"

    def test_license_polar_env_json_is_machine_readable_without_secrets(
        self,
        monkeypatch,
        capsys,
    ):
        report = SimpleNamespace(
            ready=True,
            errors=[],
            warnings=["DIRECTOR_AI_POLAR_ACTIVATION_ID is not configured"],
        )
        monkeypatch.setenv("DIRECTOR_LICENSE_KEY", "polar-secret-key")
        monkeypatch.setenv("DIRECTOR_AI_POLAR_ACCESS_TOKEN", "polar-secret-token")
        monkeypatch.setattr(
            "director_ai.core.polar_license.validate_polar_deployment_env",
            lambda: report,
        )

        with pytest.raises(SystemExit) as exc_info:
            verify_cli._cmd_license(["polar-env", "--json"])

        assert exc_info.value.code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload == {
            "ready": True,
            "errors": [],
            "warnings": ["DIRECTOR_AI_POLAR_ACTIVATION_ID is not configured"],
        }
        assert "polar-secret-key" not in json.dumps(payload)
        assert "polar-secret-token" not in json.dumps(payload)


class TestComplianceCliBranches:
    def test_compliance_help_returns_without_database_lookup(self, capsys):
        verify_cli._cmd_compliance(["--help"])

        out = capsys.readouterr().out
        assert "Usage: director-ai compliance" in out
        assert "report  [--db PATH]" in out

    def test_compliance_missing_database_exits(self, tmp_path, capsys):
        missing = tmp_path / "missing.db"

        with pytest.raises(SystemExit) as exc_info:
            verify_cli._cmd_compliance(["status", "--db", str(missing)])

        assert exc_info.value.code == 1
        assert "Audit database not found" in capsys.readouterr().out

    def test_article15_context_loader_rejects_invalid_inputs(self, tmp_path, capsys):
        missing = tmp_path / "missing.json"
        invalid_json = tmp_path / "invalid.json"
        non_object = tmp_path / "array.json"
        incomplete = tmp_path / "incomplete.json"
        invalid_json.write_text("{", encoding="utf-8")
        non_object.write_text("[]", encoding="utf-8")
        incomplete.write_text('{"system_name": ""}', encoding="utf-8")

        with pytest.raises(SystemExit) as missing_exc:
            verify_cli._load_article15_context(str(missing))
        assert missing_exc.value.code == 1
        assert "Article 15 context file not found" in capsys.readouterr().out

        with pytest.raises(SystemExit) as invalid_exc:
            verify_cli._load_article15_context(str(invalid_json))
        assert invalid_exc.value.code == 1
        assert "Invalid Article 15 context JSON" in capsys.readouterr().out

        with pytest.raises(SystemExit) as shape_exc:
            verify_cli._load_article15_context(str(non_object))
        assert shape_exc.value.code == 1
        assert "Article 15 context must be a JSON object" in capsys.readouterr().out

        with pytest.raises(SystemExit) as incomplete_exc:
            verify_cli._load_article15_context(str(incomplete))
        assert incomplete_exc.value.code == 1
        assert "Incomplete Article 15 context" in capsys.readouterr().out

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
                "--ignored",
            ]
        )

        payload = json.loads(capsys.readouterr().out)
        assert calls["window"] == (10.5, 20.25)
        assert calls["closed"] == (None, None)
        assert payload["total_interactions"] == 12
        assert payload["drift_detected"] is True

    def test_compliance_article15_context_and_pdf_outputs(
        self,
        monkeypatch,
        tmp_path,
        capsys,
    ):
        import director_ai.compliance.audit_log as audit_mod
        import director_ai.compliance.report_templates as templates_mod
        import director_ai.compliance.reporter as reporter_mod

        db_file = tmp_path / "audit.db"
        db_file.touch()
        context_file = tmp_path / "article15.json"
        output_file = tmp_path / "report.pdf"
        context_file.write_text(
            json.dumps(
                {
                    "system_name": "Director-AI",
                    "intended_purpose": "Guard generated answers.",
                    "deployment_context": "EU operator gateway.",
                    "risk_management_summary": "Reject unsafe responses.",
                    "data_governance_summary": "Tenant-safe audit records.",
                    "robustness_summary": "Drift and adversarial checks.",
                    "cybersecurity_summary": "Signed write paths.",
                    "human_oversight_summary": "Reviewers can override.",
                    "post_market_monitoring_summary": "Weekly KPI review.",
                    "known_limitations": ["Requires curated evidence."],
                    "residual_risks": ["Sparse context can reduce confidence."],
                    "evidence_refs": ["docs/internal/evidence.md"],
                }
            ),
            encoding="utf-8",
        )

        class FakeLog:
            def __init__(self, path: str) -> None:
                self.path = path

            def close(self) -> None:
                pass

        class FakeReport:
            total_interactions = 7
            overall_hallucination_rate = 0.25
            overall_hallucination_rate_ci = 0.05
            avg_score = 0.7
            avg_latency_ms = 12.0
            drift_detected = False
            incident_count = 1

            def to_article15_template(self, context):
                return {"system": {"name": context.system_name}, "total": 7}

            def to_article15_markdown(self, context):
                return f"# {context.system_name}\nArticle 15"

        class FakeReporter:
            def __init__(self, log: FakeLog) -> None:
                self.log = log

            def generate_report(self, *, since=None, until=None):
                return FakeReport()

        monkeypatch.setattr(audit_mod, "AuditLog", FakeLog)
        monkeypatch.setattr(reporter_mod, "ComplianceReporter", FakeReporter)
        monkeypatch.setattr(templates_mod, "render_compliance_pdf", lambda data: b"PDF")

        verify_cli._cmd_compliance(
            [
                "report",
                "--db",
                str(db_file),
                "--context",
                str(context_file),
                "--format",
                "json",
            ]
        )
        assert json.loads(capsys.readouterr().out)["system"]["name"] == "Director-AI"

        verify_cli._cmd_compliance(
            [
                "report",
                "--db",
                str(db_file),
                "--context",
                str(context_file),
            ]
        )
        assert "# Director-AI" in capsys.readouterr().out

        verify_cli._cmd_compliance(
            [
                "report",
                "--db",
                str(db_file),
                "--format",
                "pdf",
                "--output",
                str(output_file),
            ]
        )
        assert output_file.read_bytes() == b"PDF"
        assert f"Wrote PDF compliance report to {output_file}" in capsys.readouterr().out

    def test_compliance_markdown_status_drift_and_unknown_subcommand(
        self,
        monkeypatch,
        tmp_path,
        capsys,
    ):
        import director_ai.compliance.audit_log as audit_mod
        import director_ai.compliance.drift_detector as drift_mod
        import director_ai.compliance.reporter as reporter_mod

        db_file = tmp_path / "audit.db"
        db_file.touch()
        calls: list[str] = []

        class FakeLog:
            def __init__(self, path: str) -> None:
                calls.append(f"log:{path}")

            def close(self) -> None:
                calls.append("closed")

        class FakeReporter:
            def __init__(self, log: FakeLog) -> None:
                self.log = log

            def generate_report(self, *, since=None, until=None):
                calls.append(f"report:{since}:{until}")
                return SimpleNamespace(
                    total_interactions=1234,
                    overall_hallucination_rate=0.031,
                    overall_hallucination_rate_ci=(0.02, 0.04),
                    avg_score=0.88,
                    drift_detected=False,
                    incident_count=7,
                    to_markdown=lambda: "## Compliance\nNo drift.",
                )

        class FakeDriftDetector:
            def __init__(self, log: FakeLog) -> None:
                self.log = log

            def analyze(self, *, since=None, until=None):
                calls.append(f"drift:{since}:{until}")
                return SimpleNamespace(
                    detected=True,
                    severity="high",
                    z_score=2.5,
                    p_value=0.0123,
                    rate_change=0.042,
                    windows=[1, 2],
                )

        monkeypatch.setattr(audit_mod, "AuditLog", FakeLog)
        monkeypatch.setattr(reporter_mod, "ComplianceReporter", FakeReporter)
        monkeypatch.setattr(drift_mod, "DriftDetector", FakeDriftDetector)

        verify_cli._cmd_compliance(["report", "--db", str(db_file)])
        assert "## Compliance" in capsys.readouterr().out

        verify_cli._cmd_compliance(["status", "--db", str(db_file)])
        status_out = capsys.readouterr().out
        assert "Interactions: 1,234" in status_out
        assert "Hallucination rate: 3.10%" in status_out
        assert "Drift: no" in status_out

        verify_cli._cmd_compliance(
            ["drift", "--db", str(db_file), "--since", "1", "--until", "2"]
        )
        drift_out = capsys.readouterr().out
        assert "Drift: DETECTED (high)" in drift_out
        assert "z=2.50 p=0.0123" in drift_out
        assert "Windows: 2" in drift_out

        with pytest.raises(SystemExit) as exc_info:
            verify_cli._cmd_compliance(["rotate", "--db", str(db_file)])
        assert exc_info.value.code == 1
        assert "Unknown compliance subcommand: rotate" in capsys.readouterr().out
        assert calls.count("closed") == 3


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

        verify_cli._cmd_cost_report(["--format", "json", "--ignored"])
        assert json.loads(capsys.readouterr().out)["total_tokens"] == 1234

        verify_cli._cmd_cost_report(["--format", "html"])
        assert "<html>CHF</html>" in capsys.readouterr().out

        verify_cli._cmd_cost_report([])
        out = capsys.readouterr().out
        assert "Total cost: CHF 1.250000" in out
        assert "local: 2 calls" in out


class TestKnowledgeBaseHealthCliBranches:
    def test_kb_health_ignores_unknown_args_and_prints_warnings(
        self,
        monkeypatch,
        capsys,
    ):
        import director_ai.core.retrieval.kb_health as kb_mod
        from director_ai.core.config import DirectorConfig

        captured: dict[str, object] = {}

        class FakeKBHealthCheck:
            def __init__(
                self, store, *, min_documents: int, max_query_latency_ms: float
            ):
                captured["store"] = store
                captured["min_documents"] = min_documents
                captured["max_query_latency_ms"] = max_query_latency_ms

            def run(self):
                return SimpleNamespace(
                    summary="KB HEALTHY",
                    issues=[],
                    warnings=["latency above warning floor"],
                    healthy=True,
                )

        store = object()
        monkeypatch.setattr(
            DirectorConfig,
            "from_env",
            staticmethod(lambda: SimpleNamespace(build_store=lambda: store)),
        )
        monkeypatch.setattr(kb_mod, "KBHealthCheck", FakeKBHealthCheck)

        with pytest.raises(SystemExit) as exc_info:
            verify_cli._cmd_kb_health(
                ["--unknown", "--min-docs", "4", "--max-latency", "250"]
            )

        assert exc_info.value.code == 0
        assert captured == {
            "store": store,
            "min_documents": 4,
            "max_query_latency_ms": 250.0,
        }
        out = capsys.readouterr().out
        assert "KB HEALTHY" in out
        assert "WARNING: latency above warning floor" in out


class TestVerificationCommandBranches:
    def test_numeric_usage_and_issue_output(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc_info:
            verify_cli._cmd_verify_numeric([])
        assert exc_info.value.code == 1
        assert "verify-numeric <text>" in capsys.readouterr().out

        monkeypatch.setattr(
            "director_ai.core.verification.numeric_verifier.verify_numeric",
            lambda text: SimpleNamespace(
                valid=False,
                claims_found=2,
                error_count=1,
                warning_count=1,
                issues=[
                    SimpleNamespace(
                        severity="error",
                        issue_type="mismatch",
                        description=f"bad claim in {text}",
                    )
                ],
            ),
        )

        verify_cli._cmd_verify_numeric(["Revenue", "grew", "200%"])

        out = capsys.readouterr().out
        assert "Valid:    False" in out
        assert "Claims:  2" in out
        assert "[error] mismatch: bad claim in Revenue grew 200%" in out

    def test_reasoning_usage_and_verdict_output(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc_info:
            verify_cli._cmd_verify_reasoning([])
        assert exc_info.value.code == 1
        assert "verify-reasoning <text>" in capsys.readouterr().out

        monkeypatch.setattr(
            "director_ai.core.verification.reasoning_verifier.verify_reasoning_chain",
            lambda text: SimpleNamespace(
                chain_valid=False,
                steps_found=2,
                issues_found=1,
                verdicts=[
                    SimpleNamespace(
                        step_index=1,
                        verdict="invalid",
                        confidence=0.83,
                        reason=f"unsupported step in {text}",
                    )
                ],
            ),
        )

        verify_cli._cmd_verify_reasoning(["Step", "1", "therefore", "Step", "2"])

        out = capsys.readouterr().out
        assert "Chain valid: False" in out
        assert "Steps:       2" in out
        assert "Step 1: invalid (0.83) unsupported step" in out

    def test_temporal_freshness_usage_and_claim_output(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc_info:
            verify_cli._cmd_temporal_freshness([])
        assert exc_info.value.code == 1
        assert "temporal-freshness <text>" in capsys.readouterr().out

        monkeypatch.setattr(
            "director_ai.core.scoring.temporal_freshness.score_temporal_freshness",
            lambda text: SimpleNamespace(
                has_temporal_claims=True,
                overall_staleness_risk=0.72,
                stale_claims=["old benchmark"],
                claims=[
                    SimpleNamespace(
                        claim_type="benchmark",
                        text=text,
                        staleness_risk=0.72,
                    )
                ],
            ),
        )

        verify_cli._cmd_temporal_freshness(["Latest", "score", "is", "stable"])

        out = capsys.readouterr().out
        assert "Has temporal claims: True" in out
        assert "Staleness risk:      0.72" in out
        assert "[benchmark] Latest score is stable (risk: 0.72)" in out

    def test_check_step_usage_and_monitor_reasons(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc_info:
            verify_cli._cmd_check_step(["goal-only"])
        assert exc_info.value.code == 1
        assert "check-step <goal> <action>" in capsys.readouterr().out

        class FakeLoopMonitor:
            def __init__(self, *, goal: str) -> None:
                self.goal = goal

            def check_step(self, *, action: str, args: str):
                return SimpleNamespace(
                    step_number=4,
                    should_halt=True,
                    should_warn=True,
                    goal_drift_score=0.65,
                    budget_remaining_pct=0.25,
                    reasons=[f"{self.goal}:{action}:{args}"],
                )

        monkeypatch.setattr(
            "director_ai.agentic.loop_monitor.LoopMonitor",
            FakeLoopMonitor,
        )

        verify_cli._cmd_check_step(["publish audit", "send", "draft"])

        out = capsys.readouterr().out
        assert "Step:    4" in out
        assert "Halt:    True" in out
        assert "Budget:  25%" in out
        assert "publish audit:send:draft" in out

        class QuietLoopMonitor:
            def __init__(self, *, goal: str) -> None:
                self.goal = goal

            def check_step(self, *, action: str, args: str):
                return SimpleNamespace(
                    step_number=1,
                    should_halt=False,
                    should_warn=False,
                    goal_drift_score=0.0,
                    budget_remaining_pct=1.0,
                    reasons=[],
                )

        monkeypatch.setattr(
            "director_ai.agentic.loop_monitor.LoopMonitor",
            QuietLoopMonitor,
        )
        verify_cli._cmd_check_step(["goal", "action"])
        quiet_out = capsys.readouterr().out
        assert "Step:    1" in quiet_out
        assert "->" not in quiet_out

    def test_consensus_usage_invalid_argument_and_pair_output(
        self,
        monkeypatch,
        capsys,
    ):
        with pytest.raises(SystemExit) as exc_info:
            verify_cli._cmd_consensus(["only-one"])
        assert exc_info.value.code == 1
        assert "director-ai consensus" in capsys.readouterr().out

        with pytest.raises(SystemExit) as exc_info:
            verify_cli._cmd_consensus(["judge-a:Paris", "missing-colon"])
        assert exc_info.value.code == 1
        assert "Invalid format: 'missing-colon'" in capsys.readouterr().out

        class FakeConsensusScorer:
            def __init__(self, *, models: list[str]) -> None:
                self.models = models

            def score_responses(self, responses):
                assert self.models == ["judge-a", "judge-b"]
                assert [item.response for item in responses] == ["Paris", "Paris"]
                return SimpleNamespace(
                    num_models=2,
                    agreement_score=0.91,
                    has_consensus=True,
                    lowest_pair_agreement=0.82,
                    pairs=[
                        SimpleNamespace(
                            model_a="judge-a",
                            model_b="judge-b",
                            agreed=True,
                            divergence=0.18,
                        )
                    ],
                )

        monkeypatch.setattr(
            "director_ai.core.scoring.consensus.ConsensusScorer",
            FakeConsensusScorer,
        )

        verify_cli._cmd_consensus(["judge-a:Paris", "judge-b:Paris"])

        out = capsys.readouterr().out
        assert "Models:    2" in out
        assert "Consensus: True" in out
        assert "judge-a vs judge-b: agree (divergence=0.18)" in out


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

    def test_safety_dashboard_text_mode_uses_files_and_thresholds(
        self,
        monkeypatch,
        tmp_path,
        capsys,
    ):
        import director_ai.ui.safety_dashboard as dashboard_mod

        events = tmp_path / "events.jsonl"
        feedback = tmp_path / "feedback.jsonl"
        events.write_text('{"tenant_id":"alpha"}\n', encoding="utf-8")
        feedback.write_text('{"label":"false_positive"}\n', encoding="utf-8")
        calls: dict[str, object] = {}

        def fake_dashboard(
            events_jsonl: str,
            feedback_jsonl: str,
            halt_threshold: float,
            false_positive_threshold: float,
        ):
            calls["payload"] = (
                events_jsonl,
                feedback_jsonl,
                halt_threshold,
                false_positive_threshold,
            )
            return (
                "Safety Operations: OK",
                [("tenant-a", 0.2)],
                [("source-a", 3)],
                [("halt-a", "evidence-a")],
                "director-ai tune --threshold 0.2",
            )

        monkeypatch.setattr(dashboard_mod, "build_safety_dashboard", fake_dashboard)

        verify_cli._cmd_safety_dashboard(
            [
                "--text",
                "--events",
                str(events),
                "--feedback",
                str(feedback),
                "--halt-alert-threshold",
                "0.2",
                "--false-positive-alert-threshold",
                "0.1",
            ]
        )

        assert calls["payload"] == (
            '{"tenant_id":"alpha"}\n',
            '{"label":"false_positive"}\n',
            0.2,
            0.1,
        )
        out = capsys.readouterr().out
        assert "Safety Operations: OK" in out
        assert "tenant-a | 0.2" in out
        assert "source-a | 3" in out
        assert "halt-a | evidence-a" in out
        assert "Retune: director-ai tune --threshold 0.2" in out


class TestAdversarialCliBranches:
    def test_adversarial_test_prints_guardrail_report(self, monkeypatch, capsys):
        from director_ai.core.config import DirectorConfig

        class FakeScore:
            score = 0.87

        class FakeScorer:
            def review(self, prompt: str, response: str):
                assert prompt == "Probe prompt"
                assert response == "synthetic response"
                return True, FakeScore()

        class FakeTester:
            def __init__(self, *, review_fn, prompt: str) -> None:
                assert prompt == "Probe prompt"
                approved, score = review_fn(prompt, "synthetic response")
                assert approved is True
                assert score == 0.87

            def run(self):
                return SimpleNamespace(
                    total_patterns=5,
                    detected=4,
                    bypassed=1,
                    detection_rate=0.8,
                    is_robust=False,
                    vulnerable_categories=["prompt injection"],
                )

        monkeypatch.setattr(
            DirectorConfig,
            "from_env",
            staticmethod(lambda: SimpleNamespace(build_scorer=lambda: FakeScorer())),
        )
        monkeypatch.setattr(
            "director_ai.testing.adversarial_suite.AdversarialTester",
            FakeTester,
        )

        verify_cli._cmd_adversarial_test(["Probe prompt"])

        out = capsys.readouterr().out
        assert "Patterns:   5" in out
        assert "Detected:   4" in out
        assert "Rate:       80%" in out
        assert "Robust:     False" in out
        assert "Vulnerable: prompt injection" in out

    def test_adversarial_test_omits_vulnerable_line_when_report_is_clean(
        self,
        monkeypatch,
        capsys,
    ):
        from director_ai.core.config import DirectorConfig

        class FakeScorer:
            def review(self, prompt: str, response: str):
                return True, SimpleNamespace(score=0.99)

        class CleanTester:
            def __init__(self, *, review_fn, prompt: str) -> None:
                self.review_fn = review_fn
                self.prompt = prompt

            def run(self):
                return SimpleNamespace(
                    total_patterns=1,
                    detected=1,
                    bypassed=0,
                    detection_rate=1.0,
                    is_robust=True,
                    vulnerable_categories=[],
                )

        monkeypatch.setattr(
            DirectorConfig,
            "from_env",
            staticmethod(lambda: SimpleNamespace(build_scorer=lambda: FakeScorer())),
        )
        monkeypatch.setattr(
            "director_ai.testing.adversarial_suite.AdversarialTester",
            CleanTester,
        )

        verify_cli._cmd_adversarial_test([])

        out = capsys.readouterr().out
        assert "Robust:     True" in out
        assert "Vulnerable:" not in out
