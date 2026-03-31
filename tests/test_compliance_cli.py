# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for the compliance CLI subcommand.

Covers: drift detection, missing DB error, report md/json formats,
status output, unknown subcommand guard, parametrised formats,
pipeline integration, and performance documentation.
"""

from __future__ import annotations

import time

import pytest

from director_ai.cli import main as cli_main
from director_ai.compliance.audit_log import AuditEntry, AuditLog


def _populate_db(db_path, n_approved=10, n_rejected=3):
    log = AuditLog(db_path)
    for _ in range(n_approved):
        log.log(
            AuditEntry(
                prompt="q",
                response="a",
                model="gpt-4o",
                provider="proxy",
                score=0.85,
                approved=True,
                verdict_confidence=0.9,
                task_type="chat",
                domain="",
                latency_ms=15.0,
                timestamp=time.time(),
            )
        )
    for _ in range(n_rejected):
        log.log(
            AuditEntry(
                prompt="q",
                response="hallucinated",
                model="gpt-4o",
                provider="proxy",
                score=0.2,
                approved=False,
                verdict_confidence=0.8,
                task_type="chat",
                domain="",
                latency_ms=20.0,
                timestamp=time.time(),
            )
        )
    log.close()


class TestComplianceCli:
    def test_help(self, capsys):
        cli_main(["compliance", "--help"])
        out = capsys.readouterr().out
        assert "report" in out
        assert "status" in out
        assert "drift" in out

    def test_no_db(self, tmp_path, capsys):
        import pytest

        with pytest.raises(SystemExit, match="1"):
            cli_main(["compliance", "status", "--db", str(tmp_path / "nope.db")])

    def test_report_md(self, tmp_path, capsys):
        db = str(tmp_path / "test.db")
        _populate_db(db)
        cli_main(["compliance", "report", "--db", db])
        out = capsys.readouterr().out
        assert "Article 15" in out
        assert "Accuracy Metrics" in out

    def test_report_json(self, tmp_path, capsys):
        db = str(tmp_path / "test.db")
        _populate_db(db)
        cli_main(["compliance", "report", "--db", db, "--format", "json"])
        out = capsys.readouterr().out
        assert '"total_interactions"' in out
        assert '"hallucination_rate"' in out

    def test_status(self, tmp_path, capsys):
        db = str(tmp_path / "test.db")
        _populate_db(db, n_approved=10, n_rejected=2)
        cli_main(["compliance", "status", "--db", db])
        out = capsys.readouterr().out
        assert "Interactions: 12" in out
        assert "Drift:" in out

    def test_drift(self, tmp_path, capsys):
        db = str(tmp_path / "test.db")
        _populate_db(db)
        cli_main(["compliance", "drift", "--db", db])
        out = capsys.readouterr().out
        assert "Drift:" in out
        assert "z=" in out

    def test_unknown_subcommand(self, tmp_path, capsys):
        db = str(tmp_path / "test.db")
        _populate_db(db)
        with pytest.raises(SystemExit, match="1"):
            cli_main(["compliance", "bogus", "--db", db])

    @pytest.mark.parametrize("subcommand", ["status", "drift", "report"])
    def test_all_subcommands_produce_output(self, tmp_path, capsys, subcommand):
        db = str(tmp_path / "test.db")
        _populate_db(db)
        cli_main(["compliance", subcommand, "--db", db])
        out = capsys.readouterr().out
        assert len(out) > 0

    @pytest.mark.parametrize("fmt", ["json"])
    def test_report_json_format(self, tmp_path, capsys, fmt):
        db = str(tmp_path / "test.db")
        _populate_db(db)
        cli_main(["compliance", "report", "--db", db, "--format", fmt])
        out = capsys.readouterr().out
        assert '"total_interactions"' in out


class TestCompliancePerformanceDoc:
    """Document compliance CLI pipeline performance."""

    def test_audit_log_creates_db(self, tmp_path):
        db = str(tmp_path / "perf.db")
        log = AuditLog(db)
        assert log is not None
        log.close()

    def test_compliance_status_fast(self, tmp_path, capsys):
        db = str(tmp_path / "perf.db")
        _populate_db(db)
        t0 = time.perf_counter()
        cli_main(["compliance", "status", "--db", db])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 5000, f"Compliance status took {elapsed_ms:.0f}ms"
