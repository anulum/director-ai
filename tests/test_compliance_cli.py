# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the compliance CLI subcommand."""

from __future__ import annotations

import time

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
        import pytest

        db = str(tmp_path / "test.db")
        _populate_db(db)
        with pytest.raises(SystemExit, match="1"):
            cli_main(["compliance", "bogus", "--db", db])
