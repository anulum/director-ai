# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Verification reporting CLI real-surface tests
"""Real file-path coverage for verification reporting CLI surfaces."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from director_ai.cli import main


def _write_json(path: Path, payload: object) -> None:
    """Write a JSON fixture through the same filesystem path the CLI consumes."""
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_kpis_reads_real_bundle_and_renders_all_formats(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """KPI command should parse a real input bundle through the dispatcher."""
    bundle = tmp_path / "kpis.json"
    _write_json(
        bundle,
        {
            "items": [
                {
                    "item_id": "halt-1",
                    "score": 0.42,
                    "guard_approved": False,
                    "domain": "medical",
                    "label": "hallucination",
                },
                {
                    "item_id": "allow-1",
                    "score": 0.88,
                    "guard_approved": True,
                    "domain": "medical",
                    "label": "grounded",
                },
            ],
            "latency_ms_samples": [20.0, 40.0, 80.0],
            "tenant_boundary_violations": 0,
            "unsigned_kb_writes_rejected": 2,
            "security_exception_debt": 1,
        },
    )

    main(["kpis", "--input", str(bundle), "--format", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["report"]["labelled_total"] == 2
    assert payload["overall"] in {"ok", "watch", "alert"}

    main(["kpis", "--input", str(bundle), "--format", "markdown"])
    assert "# Guardrail KPIs" in capsys.readouterr().out

    main(["kpis", "--input", str(bundle)])
    assert "Guardrail KPIs (overall:" in capsys.readouterr().out

    main(["kpis", "--input", str(bundle), "--ignored"])
    assert "Guardrail KPIs (overall:" in capsys.readouterr().out


def test_kpis_rejects_invalid_real_inputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """KPI command should fail clearly for malformed real input files."""
    missing = tmp_path / "missing.json"
    invalid_json = tmp_path / "invalid.json"
    non_object = tmp_path / "array.json"
    bad_item = tmp_path / "bad_item.json"
    bad_targets = tmp_path / "bad_targets.json"
    valid = tmp_path / "valid.json"
    invalid_json.write_text("{", encoding="utf-8")
    _write_json(non_object, [])
    _write_json(bad_item, {"items": [42]})
    _write_json(
        bad_targets,
        {
            "items": [
                {
                    "item_id": "item-1",
                    "score": 0.5,
                    "guard_approved": True,
                    "label": "grounded",
                }
            ],
            "targets": {"watch_fraction": 1.0},
        },
    )
    _write_json(valid, {"items": []})

    with pytest.raises(SystemExit) as no_input:
        main(["kpis"])
    assert no_input.value.code == 1
    assert "Usage: director-ai kpis" in capsys.readouterr().out

    with pytest.raises(SystemExit) as bad_format:
        main(["kpis", "--input", str(valid), "--format", "xml"])
    assert bad_format.value.code == 1
    assert "Unknown format 'xml'" in capsys.readouterr().out

    for file_path, expected in (
        (missing, "input bundle not found"),
        (invalid_json, "invalid JSON"),
        (non_object, "input bundle must be a JSON object"),
        (bad_item, "invalid item record"),
        (bad_targets, "invalid targets overlay"),
    ):
        with pytest.raises(SystemExit) as exc_info:
            main(["kpis", "--input", str(file_path)])
        assert exc_info.value.code == 1
        assert expected in capsys.readouterr().out


def test_forensics_reads_real_records_and_renders_all_formats(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Forensics command should parse real eval records through the dispatcher."""
    records = [
        {
            "case_id": "case-1",
            "approved": True,
            "label": "hallucination",
            "score": 0.72,
            "threshold": 0.6,
            "scorer": "lite",
            "model": "local-judge",
            "domain": "medical",
            "evidence_count": 0,
            "unsupported_claims": 0,
        }
    ]
    list_file = tmp_path / "records-list.json"
    object_file = tmp_path / "records-object.json"
    _write_json(list_file, records)
    _write_json(object_file, {"records": records})

    main(["forensics", "--input", str(object_file), "--format", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["misses_total"] == 1
    assert payload["cases"][0]["recommended_action"] == "refresh_or_add_governed_facts"

    main(["forensics", "--input", str(list_file), "--format", "markdown"])
    assert "# Guardrail Forensics" in capsys.readouterr().out

    main(["forensics", "--input", str(list_file)])
    assert "Guardrail Forensics" in capsys.readouterr().out

    main(["forensics", "--input", str(list_file), "--ignored"])
    assert "Guardrail Forensics" in capsys.readouterr().out


def test_forensics_rejects_invalid_real_inputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Forensics command should fail clearly for malformed real input files."""
    missing = tmp_path / "missing.json"
    invalid_json = tmp_path / "invalid.json"
    non_array_records = tmp_path / "non_array_records.json"
    non_mapping_record = tmp_path / "non_mapping_record.json"
    invalid_record = tmp_path / "invalid_record.json"
    valid = tmp_path / "valid.json"
    invalid_json.write_text("{", encoding="utf-8")
    _write_json(non_array_records, {"records": {}})
    _write_json(non_mapping_record, [42])
    _write_json(invalid_record, [{"approved": True, "score": "bad", "threshold": 0.6}])
    _write_json(valid, [])

    with pytest.raises(SystemExit) as no_input:
        main(["forensics"])
    assert no_input.value.code == 1
    assert "Usage: director-ai forensics" in capsys.readouterr().out

    with pytest.raises(SystemExit) as bad_format:
        main(["forensics", "--input", str(valid), "--format", "xml"])
    assert bad_format.value.code == 1
    assert "Unknown format 'xml'" in capsys.readouterr().out

    for file_path, expected in (
        (missing, "input records not found"),
        (invalid_json, "invalid JSON"),
        (non_array_records, "invalid forensics input"),
        (non_mapping_record, "invalid forensics input"),
        (invalid_record, "invalid forensics record"),
    ):
        with pytest.raises(SystemExit) as exc_info:
            main(["forensics", "--input", str(file_path)])
        assert exc_info.value.code == 1
        assert expected in capsys.readouterr().out


def test_compliance_report_html_reads_real_audit_database(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Compliance HTML output should read a real audit database path."""
    from director_ai.compliance.audit_log import AuditEntry, AuditLog

    db_path = tmp_path / "audit.db"
    audit_log = AuditLog(db_path)
    audit_log.log(
        AuditEntry(
            prompt="What is the approved answer?",
            response="A grounded answer.",
            model="local-judge",
            provider="local",
            score=0.91,
            approved=True,
            verdict_confidence=0.88,
            task_type="review",
            domain="medical",
            latency_ms=42.0,
            timestamp=1_800_000_000.0,
        )
    )
    audit_log.close()

    main(["compliance", "report", "--db", str(db_path), "--format", "html"])

    out = capsys.readouterr().out
    assert "<!DOCTYPE html>" in out
    assert "EU AI Act Article 15 Report" in out
    assert "Total Reviews" in out
