# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Compliance Evidence Kit Tests

"""Multi-angle tests for the one-command compliance evidence kit.

The kit is assembled against a real SQLite ``AuditLog`` (no mocks), and
the CLI path is exercised through the public ``director_ai.cli.main``
entry point so flag parsing, honest degradation, and file layout are all
covered end to end.
"""

import json
import time

import pytest

from director_ai.cli import main as cli_main
from director_ai.cli_verify.evidence_kit import (
    EvidenceKitResult,
    EvidenceKitSection,
    build_evidence_kit,
)
from director_ai.compliance import AuditEntry, AuditLog
from director_ai.compliance.reporter import Article15TemplateContext
from director_ai.core.config import DirectorConfig

_KIT_FILES = (
    "governance_controls.md",
    "governance_controls.json",
    "article15_report.md",
    "article15_report.json",
    "soc2_iso_readiness.md",
    "soc2_iso_readiness.json",
    "hipaa_documentation.md",
    "hipaa_documentation.json",
    "INDEX.md",
)


def _entry(**overrides):
    base = {
        "prompt": "What colour is the sky?",
        "response": "Blue.",
        "model": "mock-model",
        "provider": "test",
        "score": 0.9,
        "approved": True,
        "verdict_confidence": 0.8,
        "task_type": "chat",
        "domain": "",
        "latency_ms": 12.0,
        "timestamp": time.time(),
    }
    base.update(overrides)
    return AuditEntry(**base)


@pytest.fixture
def audit_db(tmp_path):
    db_path = tmp_path / "audit.sqlite"
    log = AuditLog(str(db_path))
    log.log(_entry())
    log.log(_entry(response="Azure.", score=0.7, approved=False))
    log.close()
    return db_path


def _context() -> Article15TemplateContext:
    return Article15TemplateContext(
        system_name="Director-AI test deployment",
        intended_purpose="Hallucination guardrail for a support assistant",
        deployment_context="On-host inference behind the customer firewall",
        risk_management_summary="Thresholds reviewed quarterly",
        data_governance_summary="Grounding corpus curated and versioned",
        robustness_summary="Adversarial suite run per release",
        cybersecurity_summary="TLS everywhere; keys rotated",
        human_oversight_summary="Rejections routed to human review",
        post_market_monitoring_summary="Drift analysis on recorded traffic",
        known_limitations=("English-language corpora only",),
        residual_risks=("Long-context summarisation recall",),
        evidence_refs=("benchmarks/PUBLIC_BENCHMARKS.md",),
    )


def _section(result: EvidenceKitResult, name: str) -> EvidenceKitSection:
    matches = [s for s in result.sections if s.name == name]
    assert len(matches) == 1, f"section {name} missing"
    return matches[0]


class TestBuildEvidenceKit:
    def test_full_kit_with_db_and_context(self, audit_db, tmp_path):
        out = tmp_path / "kit"
        result = build_evidence_kit(
            db_path=str(audit_db),
            context=_context(),
            evidence_root=tmp_path,
            out_dir=out,
        )
        assert all(s.included for s in result.sections)
        for name in _KIT_FILES:
            assert (out / name).is_file(), f"{name} not written"
        art15 = _section(result, "article15_report")
        assert "full technical documentation" in art15.note
        index = (out / "INDEX.md").read_text(encoding="utf-8")
        for section in result.sections:
            assert section.name in index

    def test_missing_db_skips_article15_honestly(self, tmp_path):
        out = tmp_path / "kit"
        result = build_evidence_kit(
            db_path=str(tmp_path / "absent.sqlite"),
            evidence_root=tmp_path,
            out_dir=out,
        )
        art15 = _section(result, "article15_report")
        assert art15.included is False
        assert "audit database not found" in art15.note
        assert not (out / "article15_report.md").exists()
        for name in (
            "governance_controls",
            "soc2_iso_readiness",
            "hipaa_documentation",
        ):
            assert _section(result, name).included is True
        index = (out / "INDEX.md").read_text(encoding="utf-8")
        assert "**SKIPPED**" in index

    def test_without_context_emits_metrics_summary(self, audit_db, tmp_path):
        out = tmp_path / "kit"
        result = build_evidence_kit(
            db_path=str(audit_db), evidence_root=tmp_path, out_dir=out
        )
        art15 = _section(result, "article15_report")
        assert art15.included is True
        assert "metrics summary only" in art15.note
        payload = json.loads(
            (out / "article15_report.json").read_text(encoding="utf-8")
        )
        assert payload["total_interactions"] == 2
        assert 0.0 <= payload["hallucination_rate"] <= 1.0

    def test_every_json_artifact_parses(self, audit_db, tmp_path):
        out = tmp_path / "kit"
        build_evidence_kit(
            db_path=str(audit_db),
            context=_context(),
            evidence_root=tmp_path,
            out_dir=out,
        )
        for name in _KIT_FILES:
            if name.endswith(".json"):
                json.loads((out / name).read_text(encoding="utf-8"))

    def test_nested_out_dir_created(self, tmp_path):
        out = tmp_path / "a" / "b" / "kit"
        result = build_evidence_kit(
            db_path=str(tmp_path / "absent.sqlite"),
            evidence_root=tmp_path,
            out_dir=out,
        )
        assert result.out_dir == out
        assert (out / "INDEX.md").is_file()

    def test_generated_at_override_propagates(self, audit_db, tmp_path):
        out = tmp_path / "kit"
        stamp = "2026-07-14T00:00:00Z"
        result = build_evidence_kit(
            db_path=str(audit_db),
            evidence_root=tmp_path,
            out_dir=out,
            generated_at=stamp,
        )
        assert result.generated_at == stamp
        index = (out / "INDEX.md").read_text(encoding="utf-8")
        assert stamp in index
        governance = json.loads(
            (out / "governance_controls.json").read_text(encoding="utf-8")
        )
        assert governance["generated_at"] == stamp

    def test_written_files_and_summary_line(self, audit_db, tmp_path):
        full = build_evidence_kit(
            db_path=str(audit_db), evidence_root=tmp_path, out_dir=tmp_path / "k1"
        )
        assert len(full.written_files) == 8
        assert "4/4 sections" in full.summary_line()
        degraded = build_evidence_kit(
            db_path=str(tmp_path / "absent.sqlite"),
            evidence_root=tmp_path,
            out_dir=tmp_path / "k2",
        )
        assert len(degraded.written_files) == 6
        assert "3/4 sections" in degraded.summary_line()
        assert "skipped: article15_report" in degraded.summary_line()

    def test_index_carries_disclaimer_and_version(self, tmp_path):
        import director_ai

        out = tmp_path / "kit"
        build_evidence_kit(
            db_path=str(tmp_path / "absent.sqlite"),
            evidence_root=tmp_path,
            out_dir=out,
        )
        index = (out / "INDEX.md").read_text(encoding="utf-8")
        assert "Nothing in this kit is an EU AI Act conformity assessment" in index
        assert director_ai.__version__ in index

    def test_config_signals_flow_into_governance(self, audit_db, tmp_path):
        out = tmp_path / "kit"
        config = DirectorConfig(
            use_nli=False,
            scorer_backend="lite",
            redact_pii=True,
            tenant_routing=True,
        )
        build_evidence_kit(
            db_path=str(audit_db),
            config=config,
            evidence_root=tmp_path,
            out_dir=out,
        )
        governance = json.loads(
            (out / "governance_controls.json").read_text(encoding="utf-8")
        )
        assert governance["inputs"]["config_attached"] is True
        data_control = next(
            c for c in governance["controls"] if c["control_id"] == "GOV-DATA-01"
        )
        pii = next(
            s for s in data_control["signals"] if s["name"] == "pii_redaction_enabled"
        )
        assert pii["satisfied"] is True

    def test_covers_article_crosswalk_9_to_72(self, audit_db, tmp_path):
        out = tmp_path / "kit"
        build_evidence_kit(db_path=str(audit_db), evidence_root=tmp_path, out_dir=out)
        governance = (out / "governance_controls.md").read_text(encoding="utf-8")
        for article in (
            "Article 9(",
            "Article 10(",
            "Article 11(",
            "Article 12(",
            "Article 13(",
            "Article 14(",
            "Article 15(",
            "Article 72(",
        ):
            assert article in governance, f"{article} missing from crosswalk"


class TestEvidenceKitCli:
    def test_cli_writes_bundle(self, audit_db, tmp_path, capsys):
        out = tmp_path / "cli_kit"
        cli_main(
            [
                "compliance",
                "evidence-kit",
                "--db",
                str(audit_db),
                "--evidence-root",
                str(tmp_path),
                "--output",
                str(out),
            ]
        )
        captured = capsys.readouterr().out
        assert "Evidence kit: 4/4 sections" in captured
        assert (out / "INDEX.md").is_file()

    def test_cli_missing_db_still_produces_kit(self, tmp_path, capsys):
        out = tmp_path / "cli_kit"
        cli_main(
            [
                "compliance",
                "evidence-kit",
                "--db",
                str(tmp_path / "absent.sqlite"),
                "--evidence-root",
                str(tmp_path),
                "--output",
                str(out),
            ]
        )
        captured = capsys.readouterr().out
        assert "3/4 sections" in captured
        assert "- article15_report" in captured
        assert (out / "governance_controls.md").is_file()

    def test_cli_with_context_file(self, audit_db, tmp_path, capsys):
        ctx_file = tmp_path / "context.json"
        ctx = _context()
        ctx_file.write_text(
            json.dumps(
                {
                    "system_name": ctx.system_name,
                    "intended_purpose": ctx.intended_purpose,
                    "deployment_context": ctx.deployment_context,
                    "risk_management_summary": ctx.risk_management_summary,
                    "data_governance_summary": ctx.data_governance_summary,
                    "robustness_summary": ctx.robustness_summary,
                    "cybersecurity_summary": ctx.cybersecurity_summary,
                    "human_oversight_summary": ctx.human_oversight_summary,
                    "post_market_monitoring_summary": (
                        ctx.post_market_monitoring_summary
                    ),
                }
            ),
            encoding="utf-8",
        )
        out = tmp_path / "cli_kit"
        cli_main(
            [
                "compliance",
                "evidence-kit",
                "--db",
                str(audit_db),
                "--context",
                str(ctx_file),
                "--evidence-root",
                str(tmp_path),
                "--output",
                str(out),
            ]
        )
        captured = capsys.readouterr().out
        assert "full technical documentation" in captured
        report = (out / "article15_report.md").read_text(encoding="utf-8")
        assert "Director-AI test deployment" in report

    def test_help_mentions_evidence_kit(self, capsys):
        cli_main(["compliance", "--help"])
        assert "evidence-kit" in capsys.readouterr().out
