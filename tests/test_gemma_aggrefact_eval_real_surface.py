# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Real subprocess coverage for the Gemma AggreFact evaluator CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_ROOT / "benchmarks" / "gemma_aggrefact_eval.py"


def _write_protocol_modules(root: Path) -> None:
    """Write local protocol-compatible dataset and llama-cpp modules."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "datasets.py").write_text(
        """
import os


class Dataset:
    def __init__(self, rows):
        self._rows = list(rows)

    def select(self, indices):
        return Dataset([self._rows[index] for index in indices])

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)


def load_dataset(name, *, split):
    if name != "lytang/LLM-AggreFact" or split != "test":
        raise AssertionError((name, split))
    if os.environ.get("DIRECTOR_TEST_EMPTY_AGGREFACT") == "1":
        return Dataset([])
    return Dataset(
        [
            {
                "doc": "Sky is blue.",
                "claim": "Sky is blue.",
                "label": 1,
                "dataset": "AggreFact-CNN",
            },
            {
                "document": "Sky is blue.",
                "hypothesis": "Sky is red.",
                "annotations": 0,
                "dataset": "AggreFact-CNN",
            },
            {
                "doc": "Water is wet.",
                "claim": "Water is wet.",
                "label": 1,
                "dataset": "RAGTruth",
            },
            {
                "document": "Water is wet.",
                "hypothesis": "Fire is cold.",
                "annotations": 0,
                "dataset": "RAGTruth",
            },
        ]
    )
""",
        encoding="utf-8",
    )
    (root / "llama_cpp.py").write_text(
        """
import json
import os


class Llama:
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        calls_path = os.environ.get("DIRECTOR_TEST_LLAMA_CALLS")
        if calls_path:
            with open(calls_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"init": kwargs}) + "\\n")

    def create_chat_completion(self, *, messages, max_tokens, temperature):
        content = messages[0]["content"]
        verdict = "SUPPORTED"
        if "Sky is red" in content or "Fire is cold" in content:
            verdict = "NOT_SUPPORTED"
        calls_path = os.environ.get("DIRECTOR_TEST_LLAMA_CALLS")
        if calls_path:
            with open(calls_path, "a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                            "content": content,
                        }
                    )
                    + "\\n"
                )
        return {"choices": [{"message": {"content": verdict}}]}
""",
        encoding="utf-8",
    )


def _run_eval_cli(
    tmp_path: Path,
    *,
    empty_dataset: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run the production evaluator script with local protocol modules."""
    protocol_root = tmp_path / "protocol"
    _write_protocol_modules(protocol_root)
    output = tmp_path / "results" / "gemma_eval.json"
    calls_path = tmp_path / "llama_calls.jsonl"
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{protocol_root}{os.pathsep}"
        f"{PROJECT_ROOT / 'benchmarks'}{os.pathsep}"
        f"{env.get('PYTHONPATH', '')}"
    )
    env["DIRECTOR_TEST_LLAMA_CALLS"] = str(calls_path)
    if empty_dataset:
        env["DIRECTOR_TEST_EMPTY_AGGREFACT"] = "1"
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--backend",
            "llama-cpp",
            "--model",
            "/tmp/fake-gemma.gguf",
            "--max-samples",
            "4",
            "--output",
            str(output),
            "--n-ctx",
            "2048",
            "--n-threads",
            "3",
            "--log-every",
            "2",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL protocol-call records from disk."""
    return [
        cast(dict[str, Any], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_gemma_aggrefact_eval_unit_guard_has_real_cli_companion() -> None:
    """The fake-heavy unit guard must stay backed by this subprocess companion."""
    assert KNOWN_TEST_SURFACE_CLASSIFICATIONS["tests/test_gemma_aggrefact_eval.py"] == (
        "unit-guard-with-companion",
        "ML/export/eval Gemma AggreFact evaluator guard with companion "
        "tests/test_gemma_aggrefact_eval_real_surface.py",
    )


def test_gemma_aggrefact_eval_cli_writes_report_from_protocol_modules(
    tmp_path: Path,
) -> None:
    """The production CLI should write the AggreFact report through real imports."""
    result = _run_eval_cli(tmp_path)

    assert result.returncode == 0, result.stderr
    report = _read_json(tmp_path / "results" / "gemma_eval.json")
    assert report["model"] == "/tmp/fake-gemma.gguf"
    assert report["backend"] == "llama-cpp"
    assert report["samples"] == 4
    assert report["predictions"] == [1, 0, 1, 0]
    assert report["labels"] == [1, 0, 1, 0]
    assert report["datasets_per_sample"] == [
        "AggreFact-CNN",
        "AggreFact-CNN",
        "RAGTruth",
        "RAGTruth",
    ]
    assert report["global_balanced_accuracy"] == 1.0
    assert report["per_dataset"]["AggreFact-CNN"] == {
        "samples": 2,
        "balanced_accuracy": 1.0,
    }
    assert report["per_dataset"]["RAGTruth"] == {
        "samples": 2,
        "balanced_accuracy": 1.0,
    }
    calls = _read_jsonl(tmp_path / "llama_calls.jsonl")
    assert calls[0]["init"]["model_path"] == "/tmp/fake-gemma.gguf"
    assert calls[0]["init"]["n_ctx"] == 2048
    assert calls[0]["init"]["n_threads"] == 3
    assert {call["temperature"] for call in calls[1:]} == {0.0}


def test_gemma_aggrefact_eval_cli_rejects_empty_dataset(tmp_path: Path) -> None:
    """An empty loaded dataset should fail closed before report arithmetic."""
    result = _run_eval_cli(tmp_path, empty_dataset=True)

    assert result.returncode == 1
    assert "dataset is empty" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "results" / "gemma_eval.json").exists()
