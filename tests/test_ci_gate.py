# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — CI eval gate tests
"""Multi-angle tests for the CI quality gate.

Covers EvalCase label validation; JSONL loading (blank-line skip, default ids,
malformed-JSON / missing-field / empty-file errors with line numbers); the
metric maths (accuracy, hallucination catch rate, false-halt rate, plus the
None-when-absent branches); every threshold breach path including the
'threshold set but no matching cases' guards; score extraction from both a
``.score`` object and a bare float; the convenience wrapper; the report
serialisation/summary; and the CLI argument parser. A stub scorer keeps every
case model-free.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from director_ai._cli_gate import _cmd_ci_gate, _parse_args
from director_ai.ci_gate import (
    CaseOutcome,
    EvalCase,
    GateThresholds,
    gate_from_cases,
    load_cases,
    run_eval_gate,
)


@dataclass
class _Score:
    score: float


class _StubScorer:
    """Approves a response iff its text is in ``approve_texts``."""

    def __init__(self, approve_texts: set[str], *, score: float = 0.8):
        self._approve = approve_texts
        self._score = score

    def review(self, prompt: str, response: str) -> tuple[bool, _Score]:
        return response in self._approve, _Score(self._score)


class _CliConfig:
    def build_scorer(self) -> object:
        return object()


@dataclass(frozen=True)
class _CliReport:
    passed: bool

    def summary_lines(self) -> list[str]:
        return ["ci gate: PASS" if self.passed else "ci gate: FAIL"]

    def to_dict(self) -> dict[str, object]:
        return {"passed": self.passed, "total": 1}


def _case(resp: str, expected: str, cid: str = "") -> EvalCase:
    return EvalCase(prompt="q", response=resp, expected=expected, case_id=cid)


class TestEvalCase:
    def test_rejects_unknown_label(self):
        with pytest.raises(ValueError, match="expected must be one of"):
            EvalCase(prompt="q", response="r", expected="maybe")

    def test_accepts_both_labels(self):
        assert _case("r", "approve").expected == "approve"
        assert _case("r", "reject").expected == "reject"


class TestLoadCases:
    def test_loads_and_defaults_ids(self, tmp_path):
        p = tmp_path / "c.jsonl"
        p.write_text(
            '{"prompt": "a", "response": "b", "expected": "approve"}\n'
            "\n"  # blank line skipped
            '{"prompt": "c", "response": "d", "expected": "reject", "id": "x7"}\n',
            encoding="utf-8",
        )
        cases = load_cases(p)
        assert len(cases) == 2
        assert cases[0].case_id == "1"  # default = line order
        assert cases[1].case_id == "x7"

    def test_missing_field_reports_line(self, tmp_path):
        p = tmp_path / "c.jsonl"
        p.write_text('{"prompt": "a", "response": "b"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match=r":1: missing field\(s\) \['expected'\]"):
            load_cases(p)

    def test_invalid_json_reports_line(self, tmp_path):
        p = tmp_path / "c.jsonl"
        p.write_text(
            '{"prompt": "a", "response": "b", "expected": "approve"}\n{not json}\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=r":2: invalid JSON"):
            load_cases(p)

    def test_non_object_line(self, tmp_path):
        p = tmp_path / "c.jsonl"
        p.write_text("[1, 2, 3]\n", encoding="utf-8")
        with pytest.raises(ValueError, match="expected a JSON object"):
            load_cases(p)

    def test_bad_label_reports_line(self, tmp_path):
        p = tmp_path / "c.jsonl"
        p.write_text(
            '{"prompt": "a", "response": "b", "expected": "nope"}\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=r":1: .*expected must be one of"):
            load_cases(p)

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "c.jsonl"
        p.write_text("\n  \n", encoding="utf-8")
        with pytest.raises(ValueError, match="no cases found"):
            load_cases(p)


class TestMetrics:
    def test_perfect_run(self):
        cases = [_case("good", "approve"), _case("bad", "reject")]
        scorer = _StubScorer({"good"})
        report = run_eval_gate(cases, scorer, GateThresholds(min_accuracy=1.0))
        assert report.total == 2
        assert report.correct == 2
        assert report.accuracy == 1.0
        assert report.catch_rate == 1.0
        assert report.false_halt_rate == 0.0
        assert report.passed
        assert report.failures == ()

    def test_missed_hallucination_lowers_catch_rate(self):
        # Two reject cases; the scorer approves one of them (a miss).
        cases = [_case("h1", "reject"), _case("h2", "reject")]
        scorer = _StubScorer({"h1"})  # h1 wrongly approved
        report = run_eval_gate(cases, scorer, GateThresholds())
        assert report.catch_rate == 0.5
        assert report.false_halt_rate is None  # no approve cases
        assert report.accuracy == 0.5

    def test_false_halt_on_grounded_answer(self):
        cases = [_case("g1", "approve"), _case("g2", "approve")]
        scorer = _StubScorer({"g1"})  # g2 wrongly rejected
        report = run_eval_gate(cases, scorer, GateThresholds())
        assert report.false_halt_rate == 0.5
        assert report.catch_rate is None  # no reject cases

    def test_score_float_from_bare_number(self):
        class _Bare:
            def review(self, prompt, response):
                return True, 0.42

        report = run_eval_gate([_case("x", "approve")], _Bare(), GateThresholds())
        assert report.outcomes[0].score == 0.42

    def test_score_none_when_not_numeric(self):
        class _Weird:
            def review(self, prompt, response):
                return True, object()

        report = run_eval_gate([_case("x", "approve")], _Weird(), GateThresholds())
        assert report.outcomes[0].score is None

    def test_empty_cases_raises(self):
        with pytest.raises(ValueError, match="at least one case"):
            run_eval_gate([], _StubScorer(set()), GateThresholds())


class TestThresholds:
    def test_accuracy_breach_fails(self):
        cases = [_case("good", "approve"), _case("bad", "reject")]
        scorer = _StubScorer({"good", "bad"})  # approves both → 1 wrong
        report = run_eval_gate(cases, scorer, GateThresholds(min_accuracy=0.9))
        assert not report.passed
        assert any("accuracy" in f for f in report.failures)

    def test_catch_rate_breach_fails(self):
        cases = [_case("h1", "reject"), _case("h2", "reject")]
        scorer = _StubScorer({"h1"})  # catch rate 0.5
        report = run_eval_gate(cases, scorer, GateThresholds(min_catch_rate=0.8))
        assert not report.passed
        assert any("catch rate" in f for f in report.failures)

    def test_false_halt_breach_fails(self):
        cases = [_case("g1", "approve"), _case("g2", "approve")]
        scorer = _StubScorer({"g1"})  # false-halt 0.5
        report = run_eval_gate(cases, scorer, GateThresholds(max_false_halt_rate=0.1))
        assert not report.passed
        assert any("false-halt" in f for f in report.failures)

    def test_catch_rate_threshold_without_reject_cases_fails(self):
        cases = [_case("g", "approve")]
        scorer = _StubScorer({"g"})
        report = run_eval_gate(cases, scorer, GateThresholds(min_catch_rate=0.5))
        assert not report.passed
        assert any("no reject-labelled" in f for f in report.failures)

    def test_false_halt_threshold_without_approve_cases_fails(self):
        cases = [_case("h", "reject")]
        scorer = _StubScorer(set())  # rejects it
        report = run_eval_gate(cases, scorer, GateThresholds(max_false_halt_rate=0.1))
        assert not report.passed
        assert any("no approve-labelled" in f for f in report.failures)


class TestCalibrationMetrics:
    def test_ece_and_brier_computed_from_scores(self):
        # Two grounded (approve, label 1) cases scored 0.5: one bin, mean
        # confidence 0.5, empirical accuracy 1.0 -> ECE 0.5; Brier (0.5-1)^2.
        cases = [_case("g1", "approve"), _case("g2", "approve")]
        scorer = _StubScorer({"g1", "g2"}, score=0.5)
        report = run_eval_gate(cases, scorer, GateThresholds())
        assert report.ece == pytest.approx(0.5)
        assert report.brier == pytest.approx(0.25)

    def test_metrics_none_without_numeric_scores(self):
        class _NoScore:
            def review(self, prompt: str, response: str) -> tuple[bool, object]:
                return True, object()  # score is not numeric

        report = run_eval_gate([_case("x", "approve")], _NoScore(), GateThresholds())
        assert report.ece is None
        assert report.brier is None

    def test_max_ece_breach_fails(self):
        cases = [_case("g1", "approve"), _case("g2", "approve")]
        scorer = _StubScorer({"g1", "g2"}, score=0.5)  # ECE 0.5
        report = run_eval_gate(cases, scorer, GateThresholds(max_ece=0.1))
        assert not report.passed
        assert any("calibration ECE" in f for f in report.failures)

    def test_max_ece_pass_when_calibrated(self):
        # Score 1.0 on grounded cases: bin confidence 1.0 == accuracy 1.0, ECE 0.
        cases = [_case("g1", "approve"), _case("g2", "approve")]
        scorer = _StubScorer({"g1", "g2"}, score=1.0)
        report = run_eval_gate(cases, scorer, GateThresholds(max_ece=0.05))
        assert report.passed
        assert report.ece == pytest.approx(0.0)

    def test_max_ece_without_scores_fails(self):
        class _NoScore:
            def review(self, prompt: str, response: str) -> tuple[bool, object]:
                return True, object()

        report = run_eval_gate(
            [_case("x", "approve")], _NoScore(), GateThresholds(max_ece=0.1)
        )
        assert not report.passed
        assert any("max-ece set but no case" in f for f in report.failures)

    def test_to_dict_and_summary_include_calibration(self):
        cases = [_case("g1", "approve"), _case("g2", "approve")]
        report = run_eval_gate(
            cases, _StubScorer({"g1", "g2"}, score=0.5), GateThresholds()
        )
        d = report.to_dict()
        assert d["ece"] == pytest.approx(0.5)
        assert d["brier"] == pytest.approx(0.25)
        assert d["thresholds"]["max_ece"] is None
        assert any("calibration ECE" in line for line in report.summary_lines())


class TestReportRendering:
    def test_to_dict_roundtrip_fields(self):
        cases = [_case("good", "approve"), _case("bad", "reject")]
        report = gate_from_cases(cases, _StubScorer({"good"}), min_accuracy=0.5)
        d = report.to_dict()
        assert d["passed"] is True
        assert d["total"] == 2
        assert d["accuracy"] == 1.0
        assert isinstance(d["outcomes"], list) and len(d["outcomes"]) == 2
        assert d["thresholds"]["min_accuracy"] == 0.5

    def test_to_dict_can_omit_outcomes(self):
        report = gate_from_cases([_case("g", "approve")], _StubScorer({"g"}))
        assert "outcomes" not in report.to_dict(include_outcomes=False)

    def test_summary_lines_show_status_and_na(self):
        report = gate_from_cases([_case("g", "approve")], _StubScorer({"g"}))
        lines = report.summary_lines()
        assert lines[0].endswith("PASS")
        # No reject cases → catch rate is n/a.
        assert any("n/a" in line for line in lines)

    def test_case_outcome_shape(self):
        report = gate_from_cases([_case("g", "approve", "c1")], _StubScorer({"g"}))
        out = report.outcomes[0]
        assert isinstance(out, CaseOutcome)
        assert out.case_id == "c1"
        assert out.predicted == "approve"
        assert out.correct


class TestCliParser:
    def test_requires_dataset(self):
        assert _parse_args(["--min-accuracy", "0.9"]) is None

    def test_parses_all_flags(self):
        opts = _parse_args(
            [
                "--dataset",
                "c.jsonl",
                "--min-accuracy",
                "0.9",
                "--min-catch-rate",
                "0.8",
                "--max-false-halt",
                "0.1",
                "--profile",
                "medical",
                "--output",
                "g.json",
            ]
        )
        assert opts is not None
        assert opts.dataset == "c.jsonl"
        assert opts.min_accuracy == 0.9
        assert opts.min_catch_rate == 0.8
        assert opts.max_false_halt == 0.1
        assert opts.profile == "medical"
        assert opts.output == "g.json"

    def test_rejects_non_numeric_threshold(self):
        assert _parse_args(["--dataset", "c", "--min-accuracy", "high"]) is None

    def test_rejects_out_of_range_threshold(self):
        assert _parse_args(["--dataset", "c", "--min-accuracy", "1.5"]) is None

    def test_rejects_unknown_flag(self):
        assert _parse_args(["--dataset", "c", "--bogus"]) is None

    def test_rejects_non_numeric_optional_threshold(self):
        assert _parse_args(["--dataset", "c", "--min-catch-rate", "high"]) is None


class TestCliCommand:
    def test_usage_error_exits_two(self):
        with pytest.raises(SystemExit) as exc:
            _cmd_ci_gate(["--min-accuracy", "0.9"])

        assert exc.value.code == 2

    def test_load_error_exits_two(self, tmp_path, capsys):
        missing = tmp_path / "missing.jsonl"

        with pytest.raises(SystemExit) as exc:
            _cmd_ci_gate(["--dataset", str(missing)])

        assert exc.value.code == 2
        assert "Error:" in capsys.readouterr().out

    def test_profile_run_writes_report(self, tmp_path, monkeypatch, capsys):
        import director_ai.ci_gate as ci_gate_mod
        import director_ai.core.config as config_mod

        dataset = tmp_path / "cases.jsonl"
        output = tmp_path / "gate.json"
        dataset.write_text(
            '{"prompt": "p", "response": "r", "expected": "approve"}\n',
            encoding="utf-8",
        )
        seen: dict[str, object] = {}

        def fake_run_eval_gate(cases, scorer, thresholds):
            seen["case_count"] = len(cases)
            seen["scorer"] = scorer
            seen["min_accuracy"] = thresholds.min_accuracy
            seen["min_catch_rate"] = thresholds.min_catch_rate
            seen["max_false_halt_rate"] = thresholds.max_false_halt_rate
            return _CliReport(passed=True)

        monkeypatch.setattr(
            config_mod.DirectorConfig,
            "from_profile",
            classmethod(lambda cls, profile: _CliConfig()),
        )
        monkeypatch.setattr(ci_gate_mod, "run_eval_gate", fake_run_eval_gate)

        with pytest.raises(SystemExit) as exc:
            _cmd_ci_gate(
                [
                    "--dataset",
                    str(dataset),
                    "--profile",
                    "medical",
                    "--min-accuracy",
                    "0.9",
                    "--min-catch-rate",
                    "0.8",
                    "--max-false-halt",
                    "0.1",
                    "--output",
                    str(output),
                ]
            )

        assert exc.value.code == 0
        assert seen == {
            "case_count": 1,
            "scorer": seen["scorer"],
            "min_accuracy": 0.9,
            "min_catch_rate": 0.8,
            "max_false_halt_rate": 0.1,
        }
        assert output.read_text(encoding="utf-8").startswith('{\n  "passed": true')
        assert "report written" in capsys.readouterr().out

    def test_failed_env_run_exits_one(self, tmp_path, monkeypatch):
        import director_ai.ci_gate as ci_gate_mod
        import director_ai.core.config as config_mod

        dataset = tmp_path / "cases.jsonl"
        dataset.write_text(
            '{"prompt": "p", "response": "r", "expected": "reject"}\n',
            encoding="utf-8",
        )

        monkeypatch.setattr(
            config_mod.DirectorConfig,
            "from_env",
            classmethod(lambda cls: _CliConfig()),
        )
        monkeypatch.setattr(
            ci_gate_mod,
            "run_eval_gate",
            lambda cases, scorer, thresholds: _CliReport(passed=False),
        )

        with pytest.raises(SystemExit) as exc:
            _cmd_ci_gate(["--dataset", str(dataset)])

        assert exc.value.code == 1
