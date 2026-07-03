# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 evaluator real-surface tests
"""Real subprocess coverage for the Lite Scorer v2 evaluator CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "tools" / "eval_lite_scorer_v2.py"


def _write_dataset(path: Path) -> None:
    """Write a balanced held-out dataset for evaluator CLI coverage."""
    rows = [
        {
            "premise": "Alpha release evidence was signed.",
            "hypothesis": "alpha supported by release evidence",
            "label": "supported",
        },
        {
            "premise": "Beta deployment has no published smoke receipt.",
            "hypothesis": "beta denied by deployment receipt",
            "label": "unsupported",
        },
        {
            "premise": "Gamma model export produced an ONNX artefact.",
            "hypothesis": "gamma supported by onnx artefact",
            "label": True,
        },
        {
            "premise": "Delta latency was not benchmarked publicly.",
            "hypothesis": "delta denied by benchmark packet",
            "label": False,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_model_bundle(root: Path) -> Path:
    """Write the local ONNX bundle required by ``DistilledNLIBackend``."""
    model_dir = root / "model"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(b"local-onnx")
    return model_dir


def _write_external_protocol_modules(root: Path) -> Path:
    """Write local modules that preserve optional dependency protocols."""
    package_dir = root / "protocol_modules"
    package_dir.mkdir()
    (package_dir / "backfire_kernel.py").write_text(
        """
from __future__ import annotations

import math


def rust_sum_f64(values: list[float]) -> float:
    return float(sum(values))


def rust_softmax(flat: list[float], cols: int) -> list[float]:
    values = [float(value) for value in flat[:cols]]
    peak = max(values)
    exp_values = [math.exp(value - peak) for value in values]
    total = sum(exp_values)
    return [value / total for value in exp_values]


class BackfireConfig:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class RustCoherenceScorer:
    def __init__(self, *, config: BackfireConfig | None = None) -> None:
        self.config = config
""".lstrip(),
        encoding="utf-8",
    )
    (package_dir / "onnxruntime.py").write_text(
        """
from __future__ import annotations

import time


class _Input:
    def __init__(self, name: str) -> None:
        self.name = name


class InferenceSession:
    def __init__(self, path: str, *, providers: list[str]) -> None:
        self.path = path
        self.providers = providers

    def get_inputs(self) -> list[_Input]:
        return [_Input("input_ids"), _Input("attention_mask")]

    def run(self, _outputs: object, inputs: dict[str, list[list[int]]]) -> list[list[list[float]]]:
        time.sleep(0.0001)
        token = float(inputs["input_ids"][0][0])
        return [[[token, 0.0]]]
""".lstrip(),
        encoding="utf-8",
    )
    (package_dir / "transformers.py").write_text(
        """
from __future__ import annotations


class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, _model_path: str, *, revision: str) -> "AutoTokenizer":
        return cls()

    def __call__(
        self,
        _premise: str,
        hypothesis: str,
        **_kwargs: object,
    ) -> dict[str, list[list[int]]]:
        token = 3 if "supported" in hypothesis else -3
        return {"input_ids": [[token]], "attention_mask": [[1]]}
""".lstrip(),
        encoding="utf-8",
    )
    return package_dir


def _run_eval_cli(
    tmp_path: Path,
    *,
    dataset: Path,
    model_dir: Path,
    output: Path,
    threshold: str = "0.5",
) -> subprocess.CompletedProcess[str]:
    """Run the production evaluator CLI with local optional-dep protocols."""
    protocol_dir = _write_external_protocol_modules(tmp_path)
    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath_parts = [str(protocol_dir), str(ROOT / "src")]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["DIRECTOR_ONNX_ALLOWED_DIRS"] = str(model_dir)

    return subprocess.run(
        [
            sys.executable,
            str(EVALUATOR),
            "--dataset",
            str(dataset),
            "--model-path",
            str(model_dir),
            "--threshold",
            threshold,
            "--latency-sample-count",
            "4",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
    )


def test_lite_scorer_v2_eval_unit_guard_has_real_surface_companion() -> None:
    """The eval unit guard should be backed by real subprocess CLI coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_lite_scorer_v2_eval.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_lite_scorer_v2_eval_real_surface.py" in category


def test_lite_scorer_v2_eval_real_surface_declares_protocol_fake() -> None:
    """The companion should declare its external optional-dependency fakes."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_lite_scorer_v2_eval_real_surface.py"
    ]

    assert classification == "approved-protocol-fake"
    assert "onnxruntime" in category
    assert "transformers" in category
    assert "backfire-kernel" in category


def test_lite_scorer_v2_eval_cli_scores_dataset_and_writes_result(
    tmp_path: Path,
) -> None:
    """The production evaluator CLI should score JSONL data end to end."""
    dataset = tmp_path / "heldout.jsonl"
    output = tmp_path / "benchmarks" / "results" / "lite_scorer_v2_eval.json"
    _write_dataset(dataset)
    model_dir = _write_model_bundle(tmp_path)

    result = _run_eval_cli(
        tmp_path,
        dataset=dataset,
        model_dir=model_dir,
        output=output,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    stdout_payload = json.loads(result.stdout)
    assert stdout_payload == {
        "balanced_accuracy": 1.0,
        "dataset": dataset.as_posix(),
        "latency_p50_ms": stdout_payload["latency_p50_ms"],
        "latency_p95_ms": stdout_payload["latency_p95_ms"],
        "latency_sample_count": 4,
        "rows": 4,
        "threshold": 0.5,
        "true_negative_rate": 1.0,
        "true_positive_rate": 1.0,
    }
    assert stdout_payload["latency_p50_ms"] > 0.0
    assert stdout_payload["latency_p95_ms"] >= stdout_payload["latency_p50_ms"]

    output_payload = json.loads(output.read_text(encoding="utf-8"))
    assert output_payload["heldout_eval_dataset"] == dataset.as_posix()
    assert output_payload["heldout_eval_rows"] == 4
    assert output_payload["heldout_eval_balanced_accuracy"] == 1.0
    assert output_payload["heldout_eval_threshold"] == 0.5
    assert output_payload["latency_sample_count"] == 4
    assert "public_score_claim" not in output_payload


def test_lite_scorer_v2_eval_cli_reports_invalid_dataset_to_stderr(
    tmp_path: Path,
) -> None:
    """CLI failures should not pollute stdout JSON consumers."""
    dataset = tmp_path / "heldout.jsonl"
    output = tmp_path / "result.json"
    dataset.write_text("", encoding="utf-8")
    model_dir = _write_model_bundle(tmp_path)

    result = _run_eval_cli(
        tmp_path,
        dataset=dataset,
        model_dir=model_dir,
        output=output,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == f"{dataset}: dataset must contain at least one row\n"
    assert not output.exists()
