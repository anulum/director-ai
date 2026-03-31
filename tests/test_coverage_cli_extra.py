# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Coverage tests for cli.py — eval, bench, serve subcommands."""

from __future__ import annotations

import os
import sys
from types import ModuleType
from unittest.mock import patch

import pytest

from director_ai.cli import main


def _make_bench_module(**funcs):
    """Build a fake benchmarks.regression_suite module with named callables."""
    mod = ModuleType("benchmarks.regression_suite")
    for name, fn in funcs.items():
        fn.__name__ = name
        setattr(mod, name, fn)
    return mod


def _noop():
    pass


class TestCliEval:
    def test_eval_no_benchmarks(self, capsys):
        with (
            patch.dict(sys.modules, {"benchmarks": None, "benchmarks.run_all": None}),
            pytest.raises(SystemExit),
        ):
            main(["eval"])

    def test_eval_bad_max_samples(self, capsys):
        with (
            patch.dict(sys.modules, {"benchmarks": None, "benchmarks.run_all": None}),
            pytest.raises(SystemExit),
        ):
            main(["eval", "--max-samples", "abc"])


class TestCliBench:
    def test_bench_bad_seed(self, capsys):
        with pytest.raises(SystemExit):
            main(["bench", "--seed", "abc"])

    def test_bench_bad_max_samples(self, capsys):
        with pytest.raises(SystemExit):
            main(["bench", "--max-samples", "xyz"])

    def test_bench_unknown_dataset(self, capsys):
        mod = _make_bench_module(
            test_heuristic_accuracy=lambda: None,
            test_streaming_stability=lambda: None,
            test_latency_ceiling=lambda: None,
            test_metrics_integrity=lambda: None,
            test_evidence_schema=lambda: None,
        )
        bm = ModuleType("benchmarks")
        with (
            patch.dict(
                sys.modules,
                {
                    "benchmarks": bm,
                    "benchmarks.regression_suite": mod,
                },
            ),
            pytest.raises(SystemExit),
        ):
            main(["bench", "--dataset", "unknown"])

    def _reg_module(self):
        return _make_bench_module(
            test_heuristic_accuracy=lambda: None,
            test_streaming_stability=lambda: None,
            test_latency_ceiling=lambda: None,
            test_metrics_integrity=lambda: None,
            test_evidence_schema=lambda: None,
            test_e2e_heuristic_delta=lambda: None,
            test_false_halt_rate=lambda: None,
        )

    def test_bench_regression(self, capsys):
        mod = self._reg_module()
        bm = ModuleType("benchmarks")
        with patch.dict(
            sys.modules,
            {
                "benchmarks": bm,
                "benchmarks.regression_suite": mod,
            },
        ):
            main(["bench", "--dataset", "regression", "--seed", "42"])
            assert "passed" in capsys.readouterr().out

    def test_bench_with_output(self, capsys, tmp_path):
        mod = self._reg_module()
        bm = ModuleType("benchmarks")
        with patch.dict(
            sys.modules,
            {
                "benchmarks": bm,
                "benchmarks.regression_suite": mod,
            },
        ):
            out_file = str(tmp_path / "results.json")
            main(["bench", "--output", out_file])
            assert os.path.exists(out_file)

    def test_bench_with_failure(self, capsys):
        def failing():
            raise AssertionError("intentional fail")

        mod = _make_bench_module(
            test_heuristic_accuracy=failing,
            test_streaming_stability=lambda: None,
            test_latency_ceiling=lambda: None,
            test_metrics_integrity=lambda: None,
            test_evidence_schema=lambda: None,
            test_e2e_heuristic_delta=lambda: None,
            test_false_halt_rate=lambda: None,
        )
        bm = ModuleType("benchmarks")
        with (
            patch.dict(
                sys.modules,
                {
                    "benchmarks": bm,
                    "benchmarks.regression_suite": mod,
                },
            ),
            pytest.raises(SystemExit),
        ):
            main(["bench"])

    def test_bench_max_samples(self, capsys):
        mod = self._reg_module()
        bm = ModuleType("benchmarks")
        with patch.dict(
            sys.modules,
            {
                "benchmarks": bm,
                "benchmarks.regression_suite": mod,
            },
        ):
            main(["bench", "--max-samples", "2"])
            assert "passed" in capsys.readouterr().out

    def test_bench_no_benchmarks(self, capsys):
        with (
            patch.dict(
                sys.modules,
                {
                    "benchmarks": None,
                    "benchmarks.regression_suite": None,
                },
            ),
            pytest.raises(SystemExit),
        ):
            main(["bench"])

    def test_bench_streaming_dataset(self, capsys):
        mod = self._reg_module()
        bm = ModuleType("benchmarks")
        with patch.dict(
            sys.modules,
            {
                "benchmarks": bm,
                "benchmarks.regression_suite": mod,
            },
        ):
            main(["bench", "--dataset", "streaming"])
            assert "passed" in capsys.readouterr().out

    def test_bench_e2e_dataset(self, capsys):
        mod = self._reg_module()
        bm = ModuleType("benchmarks")
        with patch.dict(
            sys.modules,
            {
                "benchmarks": bm,
                "benchmarks.regression_suite": mod,
            },
        ):
            main(["bench", "--dataset", "e2e"])
            assert "passed" in capsys.readouterr().out


class TestCliServe:
    def test_serve_bad_port(self):
        with pytest.raises(SystemExit):
            main(["serve", "--port", "abc"])

    def test_serve_bad_workers(self):
        with pytest.raises(SystemExit):
            main(["serve", "--workers", "abc"])

    def test_serve_bad_workers_zero(self):
        with pytest.raises(SystemExit):
            main(["serve", "--workers", "0"])

    def test_serve_bad_transport(self):
        with pytest.raises(SystemExit):
            main(["serve", "--transport", "websocket"])

    def test_serve_no_uvicorn(self, capsys):
        with patch.dict(sys.modules, {"uvicorn": None}), pytest.raises(SystemExit):
            main(["serve"])
