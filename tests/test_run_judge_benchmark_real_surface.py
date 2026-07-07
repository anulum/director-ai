# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Judge Benchmark Real Surface Tests
"""Real subprocess coverage for the judge benchmark runner."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_SCRIPT = REPO_ROOT / "benchmarks" / "run_judge_benchmark.py"


def _write_torch_protocol_module(directory: Path) -> None:
    """Create a minimal torch protocol module for subprocess GPU introspection."""
    (directory / "torch.py").write_text(
        "\n".join(
            [
                "__version__ = 'protocol-torch-0'",
                "",
                "class _Cuda:",
                "    @staticmethod",
                "    def is_available():",
                "        return False",
                "",
                "cuda = _Cuda()",
            ],
        ),
        encoding="utf-8",
    )


def test_run_judge_benchmark_unit_guard_has_real_cli_companion() -> None:
    """Verify the helper-heavy guard is backed by this real CLI companion."""
    assert KNOWN_TEST_SURFACE_CLASSIFICATIONS["tests/test_run_judge_benchmark.py"] == (
        "unit-guard-with-companion",
        "ML/export/eval benchmark guard with companion "
        "tests/test_run_judge_benchmark_real_surface.py",
    )


def test_run_judge_benchmark_cli_writes_summary_to_requested_directory(
    tmp_path: Path,
) -> None:
    """Run the production benchmark CLI without downloading datasets or models."""
    protocol_dir = tmp_path / "protocol_modules"
    protocol_dir.mkdir()
    _write_torch_protocol_module(protocol_dir)
    results_dir = tmp_path / "judge_results"
    missing_judge_model = tmp_path / "missing-judge-model"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(protocol_dir), str(REPO_ROOT)]),
    }

    completed = subprocess.run(
        [
            sys.executable,
            str(BENCHMARK_SCRIPT),
            "--skip-latency",
            "--skip-nli-only",
            "--samples",
            "3",
            "--results-dir",
            str(results_dir),
            "--judge-model-path",
            str(missing_judge_model),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 0, completed.stderr
    summary_path = results_dir / "judge_bench_summary_3.json"
    assert summary_path.exists()
    assert not (
        REPO_ROOT / "benchmarks" / "results" / "judge_bench_summary_3.json"
    ).exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["local_judge"]["status"] == "failed"
    assert str(missing_judge_model) in summary["local_judge"]["error"]
    assert "nli_only" not in summary
    assert summary["hw"] == {"gpu": "none", "cuda": False}
