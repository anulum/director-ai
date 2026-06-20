# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — CLI `kpis` command tests
"""Behavioural coverage for the board-level KPI CLI export command."""

from __future__ import annotations

import json

import pytest

import director_ai._cli_verify as verify_cli

_BUNDLE = {
    "items": [
        {
            "item_id": "a",
            "score": 0.9,
            "guard_approved": False,
            "domain": "legal",
            "label": "hallucination",
        },
        {
            "item_id": "b",
            "score": 0.2,
            "guard_approved": True,
            "domain": "legal",
            "label": "grounded",
        },
        {
            "item_id": "c",
            "score": 0.8,
            "guard_approved": False,
            "domain": "med",
            "label": "grounded",
        },
    ],
    "latency_ms_samples": [10.0, 20.0, 30.0],
    "unsigned_kb_writes_rejected": 2,
    "security_exception_debt": 1,
}


def _write_bundle(tmp_path, payload) -> str:
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


class TestItemBuilder:
    def test_drops_unknown_fields_and_keeps_known(self):
        items = verify_cli._kpi_items_from_records(
            [{"item_id": "x", "score": 0.5, "guard_approved": True, "extra": 99}]
        )
        assert len(items) == 1
        assert items[0].item_id == "x"

    def test_rejects_non_object_record(self):
        with pytest.raises(ValueError, match="must be a JSON object"):
            verify_cli._kpi_items_from_records(["not-an-object"])


class TestKpisCommand:
    def test_text_format_default(self, tmp_path, capsys):
        verify_cli._cmd_kpis(["--input", _write_bundle(tmp_path, _BUNDLE)])
        out = capsys.readouterr().out
        assert "Guardrail KPIs (overall: alert)" in out
        assert "labelled_decisions: 3" in out

    def test_markdown_format(self, tmp_path, capsys):
        verify_cli._cmd_kpis(
            ["--input", _write_bundle(tmp_path, _BUNDLE), "--format", "markdown"]
        )
        out = capsys.readouterr().out
        assert "# Guardrail KPIs — overall: ALERT" in out
        assert "## Per-domain false-positive rate" in out

    def test_json_format(self, tmp_path, capsys):
        verify_cli._cmd_kpis(
            ["--input", _write_bundle(tmp_path, _BUNDLE), "--format", "json"]
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["overall"] == "alert"
        assert payload["report"]["labelled_total"] == 3
        assert payload["statuses"]["false_positive_rate[med]"] == "alert"

    def test_unknown_argument_is_ignored(self, tmp_path, capsys):
        verify_cli._cmd_kpis(
            ["--ignored", "--input", _write_bundle(tmp_path, _BUNDLE)]
        )
        assert "Guardrail KPIs" in capsys.readouterr().out

    def test_targets_overlay_applied(self, tmp_path, capsys):
        bundle = {
            "items": [
                {
                    "item_id": "g",
                    "score": 0.3,
                    "guard_approved": False,
                    "domain": "legal",
                    "label": "grounded",
                }
            ],
            "targets": {"max_false_positive_rate": 1.0},
        }
        verify_cli._cmd_kpis(
            ["--input", _write_bundle(tmp_path, bundle), "--format", "json"]
        )
        payload = json.loads(capsys.readouterr().out)
        # 100% FPR, but the maximally lenient target keeps it out of alert.
        assert payload["statuses"]["false_positive_rate"] != "alert"

    def test_missing_input_flag_exits(self, capsys):
        with pytest.raises(SystemExit) as exc:
            verify_cli._cmd_kpis([])
        assert exc.value.code == 1
        assert "Usage: director-ai kpis" in capsys.readouterr().out

    def test_unknown_format_exits(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc:
            verify_cli._cmd_kpis(
                ["--input", _write_bundle(tmp_path, _BUNDLE), "--format", "yaml"]
            )
        assert exc.value.code == 1
        assert "Unknown format" in capsys.readouterr().out

    def test_missing_file_exits(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc:
            verify_cli._cmd_kpis(["--input", str(tmp_path / "absent.json")])
        assert exc.value.code == 1
        assert "not found" in capsys.readouterr().out

    def test_invalid_json_exits(self, tmp_path, capsys):
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            verify_cli._cmd_kpis(["--input", str(path)])
        assert exc.value.code == 1
        assert "invalid JSON" in capsys.readouterr().out

    def test_non_object_bundle_exits(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc:
            verify_cli._cmd_kpis(["--input", _write_bundle(tmp_path, [1, 2, 3])])
        assert exc.value.code == 1
        assert "must be a JSON object" in capsys.readouterr().out

    def test_invalid_item_record_exits(self, tmp_path, capsys):
        bundle = {"items": [{"item_id": "x", "score": 5.0, "guard_approved": True}]}
        with pytest.raises(SystemExit) as exc:
            verify_cli._cmd_kpis(["--input", _write_bundle(tmp_path, bundle)])
        assert exc.value.code == 1
        assert "invalid item record" in capsys.readouterr().out

    def test_invalid_targets_overlay_exits(self, tmp_path, capsys):
        bundle = {"items": [], "targets": {"max_false_positive_rate": 5.0}}
        with pytest.raises(SystemExit) as exc:
            verify_cli._cmd_kpis(["--input", _write_bundle(tmp_path, bundle)])
        assert exc.value.code == 1
        assert "invalid targets overlay" in capsys.readouterr().out


class TestDispatch:
    def test_main_routes_to_kpis(self, tmp_path, capsys):
        from director_ai import cli

        cli.main(["kpis", "--input", _write_bundle(tmp_path, _BUNDLE)])
        assert "Guardrail KPIs" in capsys.readouterr().out

    def test_help_lists_kpis(self, capsys):
        from director_ai import cli

        cli.main(["--help"])
        assert "kpis --input" in capsys.readouterr().out
