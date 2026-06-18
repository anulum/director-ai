# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — CLI forensics command tests
"""Behavioural tests for the scorer-miss forensics CLI command."""

from __future__ import annotations

import json

import pytest

import director_ai._cli_verify as verify_cli

_RECORDS = {
    "records": [
        {
            "director.eval.answer_id": "fn-1",
            "director.eval.approved": True,
            "director.eval.score": 0.82,
            "director.eval.threshold": 0.6,
            "director.eval.scorer": "nli",
            "director.eval.model": "model-a",
            "director.eval.evidence_count": 0,
            "label": "hallucination",
        },
        {
            "answer_id": "ok-1",
            "approved": False,
            "score": 0.2,
            "threshold": 0.6,
            "scorer": "nli",
            "label": "hallucination",
        },
    ]
}


def _write_payload(tmp_path, payload) -> str:
    path = tmp_path / "forensics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


class TestForensicsRecordsPayload:
    def test_accepts_top_level_records_object(self) -> None:
        records = verify_cli._forensics_records_from_payload(_RECORDS)
        assert len(records) == 2

    def test_accepts_top_level_array(self) -> None:
        records = verify_cli._forensics_records_from_payload(_RECORDS["records"])
        assert len(records) == 2

    def test_rejects_non_array_records(self) -> None:
        with pytest.raises(ValueError, match="records"):
            verify_cli._forensics_records_from_payload({"records": {"x": 1}})

    def test_rejects_non_object_record(self) -> None:
        with pytest.raises(ValueError, match="each record"):
            verify_cli._forensics_records_from_payload([1])


class TestForensicsCommand:
    def test_text_format_default(self, tmp_path, capsys) -> None:
        verify_cli._cmd_forensics(["--input", _write_payload(tmp_path, _RECORDS)])
        out = capsys.readouterr().out
        assert "Guardrail Forensics" in out
        assert "false_negatives: 1" in out

    def test_markdown_format(self, tmp_path, capsys) -> None:
        verify_cli._cmd_forensics(
            ["--input", _write_payload(tmp_path, _RECORDS), "--format", "markdown"]
        )
        out = capsys.readouterr().out
        assert "# Guardrail Forensics" in out
        assert "| fn-1 | false_negative |" in out

    def test_json_format(self, tmp_path, capsys) -> None:
        verify_cli._cmd_forensics(
            ["--input", _write_payload(tmp_path, _RECORDS), "--format", "json"]
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["misses_total"] == 1
        assert payload["privacy"]["raw_response_included"] is False

    def test_missing_input_flag_exits(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc:
            verify_cli._cmd_forensics([])
        assert exc.value.code == 1
        assert "Usage: director-ai forensics" in capsys.readouterr().out

    def test_unknown_format_exits(self, tmp_path, capsys) -> None:
        with pytest.raises(SystemExit) as exc:
            verify_cli._cmd_forensics(
                ["--input", _write_payload(tmp_path, _RECORDS), "--format", "yaml"]
            )
        assert exc.value.code == 1
        assert "Unknown format" in capsys.readouterr().out

    def test_invalid_record_exits(self, tmp_path, capsys) -> None:
        bad = {"records": [{"score": 0.5, "threshold": 0.6}]}
        with pytest.raises(SystemExit) as exc:
            verify_cli._cmd_forensics(["--input", _write_payload(tmp_path, bad)])
        assert exc.value.code == 1
        assert "invalid forensics record" in capsys.readouterr().out

    def test_main_routes_to_forensics(self, tmp_path, capsys) -> None:
        from director_ai import cli

        cli.main(["forensics", "--input", _write_payload(tmp_path, _RECORDS)])
        assert "Guardrail Forensics" in capsys.readouterr().out

    def test_help_lists_forensics(self, capsys) -> None:
        from director_ai import cli

        cli.main(["--help"])
        assert "forensics --input" in capsys.readouterr().out
