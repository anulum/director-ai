# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real subprocess coverage for the Gemma AggreFact self-consistency evaluator."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PROJECT_ROOT / "benchmarks" / "gemma_aggrefact_self_consistency.py"


def _write_protocol_modules(module_dir: Path) -> Path:
    """Write protocol-compatible local dataset and llama-cpp modules."""
    module_dir.mkdir(parents=True, exist_ok=True)
    call_log = module_dir / "llama_calls.jsonl"
    (module_dir / "datasets.py").write_text(
        """
from __future__ import annotations

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
            {
                "doc": "Contract is signed.",
                "claim": "Contract is signed.",
                "label": 1,
                "dataset": "Wice",
            },
            {
                "document": "Contract is signed.",
                "hypothesis": "Contract is void.",
                "annotations": 0,
                "dataset": "Wice",
            },
        ]
    )
""",
        encoding="utf-8",
    )
    (module_dir / "llama_cpp.py").write_text(
        f"""
from __future__ import annotations

import json
from pathlib import Path

CALL_LOG = Path({str(call_log)!r})


class Llama:
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        with CALL_LOG.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({{"init": kwargs}}, sort_keys=True) + "\\n")

    def create_chat_completion(
        self,
        *,
        messages,
        max_tokens,
        temperature,
        top_p,
    ):
        content = messages[0]["content"]
        event = {{
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "content": content,
        }}
        with CALL_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\\n")
        verdict = (
            "NOT_SUPPORTED"
            if (
                "Sky is red" in content
                or "Fire is cold" in content
                or "Contract is void" in content
            )
            else "SUPPORTED"
        )
        return {{"choices": [{{"message": {{"content": verdict}}}}]}}
""",
        encoding="utf-8",
    )
    return call_log


def _run_self_consistency_cli(
    tmp_path: Path,
    *,
    empty_dataset: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    """Run the production self-consistency evaluator with protocol modules."""
    protocol_dir = tmp_path / "protocol"
    call_log = _write_protocol_modules(protocol_dir)
    report_path = tmp_path / "results" / "gemma_self_consistency.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(protocol_dir), str(PROJECT_ROOT / "benchmarks"), env.get("PYTHONPATH", "")]
    )
    if empty_dataset:
        env["DIRECTOR_TEST_EMPTY_AGGREFACT"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--model",
            "/models/local-self-consistency.gguf",
            "--max-samples",
            "6",
            "--k",
            "3",
            "--temperature",
            "0.4",
            "--top-p",
            "0.9",
            "--output",
            str(report_path),
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
    return completed, report_path, call_log


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL protocol-call records from disk."""
    return [
        cast(dict[str, Any], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_gemma_aggrefact_self_consistency_unit_guard_has_real_cli_companion() -> None:
    """Ensure the self-consistency unit guard has this subprocess companion."""
    assert KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_gemma_aggrefact_self_consistency.py"
    ] == (
        "unit-guard-with-companion",
        "ML/export/eval Gemma AggreFact self-consistency evaluator guard with "
        "companion tests/test_gemma_aggrefact_self_consistency_real_surface.py",
    )


def test_gemma_aggrefact_self_consistency_cli_writes_report_from_protocol_modules(
    tmp_path: Path,
) -> None:
    """Exercise the self-consistency evaluator through its benchmark CLI contract."""
    completed, report_path, call_log = _run_self_consistency_cli(tmp_path)

    assert completed.returncode == 0, completed.stderr
    report = _read_json(report_path)
    assert report["schema_version"] == 2
    assert report["model"] == "/models/local-self-consistency.gguf"
    assert report["samples"] == 6
    assert report["k"] == 3
    assert report["temperature"] == 0.4
    assert report["top_p"] == 0.9
    assert report["global_balanced_accuracy"] == 1.0
    assert report["unknown_predictions"] == 0
    assert report["predictions"] == [1, 0, 1, 0, 1, 0]
    assert report["support_fractions"] == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    assert report["labels"] == [1, 0, 1, 0, 1, 0]
    assert report["datasets_per_sample"] == [
        "AggreFact-CNN",
        "AggreFact-CNN",
        "RAGTruth",
        "RAGTruth",
        "Wice",
        "Wice",
    ]
    assert report["families_per_sample"] == [
        "summ",
        "summ",
        "rag",
        "rag",
        "claim",
        "claim",
    ]
    assert report["per_dataset"] == {
        "AggreFact-CNN": {"samples": 2, "balanced_accuracy": 1.0},
        "RAGTruth": {"samples": 2, "balanced_accuracy": 1.0},
        "Wice": {"samples": 2, "balanced_accuracy": 1.0},
    }
    assert report["per_family"] == {
        "claim": {"samples": 2, "balanced_accuracy": 1.0},
        "rag": {"samples": 2, "balanced_accuracy": 1.0},
        "summ": {"samples": 2, "balanced_accuracy": 1.0},
    }

    calls = _read_jsonl(call_log)
    assert calls[0]["init"]["model_path"] == "/models/local-self-consistency.gguf"
    assert calls[0]["init"]["n_ctx"] == 2048
    assert calls[0]["init"]["n_threads"] == 3
    assert calls[0]["init"]["logits_all"] is False
    assert len(calls[1:]) == 18
    assert {call["temperature"] for call in calls[1:]} == {0.4}
    assert {call["top_p"] for call in calls[1:]} == {0.9}
    assert {call["max_tokens"] for call in calls[1:]} == {8}
    prompt_text = "\n".join(str(call["content"]) for call in calls[1:])
    assert "careful summarisation evaluator" in prompt_text
    assert "retrieval-augmented generation outputs" in prompt_text
    assert "fact-checking assistant" in prompt_text
    assert "Sky is red." in prompt_text
    assert "Fire is cold." in prompt_text
    assert "Contract is void." in prompt_text


def test_gemma_aggrefact_self_consistency_cli_rejects_empty_dataset(
    tmp_path: Path,
) -> None:
    """Reject empty self-consistency inputs before reporting or dividing by zero."""
    completed, report_path, _call_log = _run_self_consistency_cli(
        tmp_path,
        empty_dataset=True,
    )

    assert completed.returncode == 1
    assert "dataset is empty" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not report_path.exists()
