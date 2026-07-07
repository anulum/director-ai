# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real subprocess coverage for the Gemma AggreFact logprob evaluator."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PROJECT_ROOT / "benchmarks" / "gemma_aggrefact_logprob.py"


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
        ]
    )
""",
        encoding="utf-8",
    )
    (module_dir / "llama_cpp.py").write_text(
        f"""
from __future__ import annotations

import json
import math
from pathlib import Path

CALL_LOG = Path({str(call_log)!r})


class Llama:
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        with CALL_LOG.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({{"init": kwargs}}, sort_keys=True) + "\\n")

    def tokenize(self, value, *, add_bos):
        text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        return [200] if "NOT" in text else [100]

    def create_chat_completion(
        self,
        *,
        messages,
        max_tokens,
        temperature,
        logprobs,
        top_logprobs,
    ):
        content = messages[0]["content"]
        score = 0.2 if "Sky is red" in content or "Fire is cold" in content else 0.8
        verdict = "NOT_SUPPORTED" if score < 0.5 else "SUPPORTED"
        event = {{
            "max_tokens": max_tokens,
            "temperature": temperature,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            "content": content,
        }}
        with CALL_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\\n")
        return {{
            "choices": [
                {{
                    "message": {{"content": verdict}},
                    "logprobs": {{
                        "content": [
                            {{
                                "top_logprobs": [
                                    {{"token": "SUPPORTED", "logprob": math.log(score)}},
                                    {{"token": "NOT", "logprob": math.log(1 - score)}},
                                ]
                            }}
                        ]
                    }},
                }}
            ]
        }}
""",
        encoding="utf-8",
    )
    return call_log


def _run_logprob_cli(
    tmp_path: Path,
    *,
    empty_dataset: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    """Run the production logprob evaluator script with local protocol modules."""
    protocol_dir = tmp_path / "protocol"
    call_log = _write_protocol_modules(protocol_dir)
    report_path = tmp_path / "results" / "gemma_logprob.json"
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
            "/models/local-logprob.gguf",
            "--max-samples",
            "4",
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


def test_gemma_aggrefact_logprob_unit_guard_has_real_cli_companion() -> None:
    """Ensure the logprob unit guard is backed by this subprocess companion."""
    assert KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_gemma_aggrefact_logprob.py"
    ] == (
        "unit-guard-with-companion",
        "ML/export/eval Gemma AggreFact logprob evaluator guard with companion "
        "tests/test_gemma_aggrefact_logprob_real_surface.py",
    )


def test_gemma_aggrefact_logprob_cli_writes_report_from_protocol_modules(
    tmp_path: Path,
) -> None:
    """Exercise the logprob evaluator through its benchmark CLI contract."""
    completed, report_path, call_log = _run_logprob_cli(tmp_path)

    assert completed.returncode == 0, completed.stderr
    report = _read_json(report_path)
    assert report["model"] == "/models/local-logprob.gguf"
    assert report["samples"] == 4
    assert report["global_balanced_accuracy_t05"] == 1.0
    assert report["global_balanced_accuracy_optimal"] == 1.0
    assert report["per_dataset_avg_balanced_accuracy"] == 1.0
    assert report["invalid_scores"] == 0
    assert report["scores"] == [0.8, 0.2, 0.8, 0.2]
    assert report["labels"] == [1, 0, 1, 0]
    assert report["datasets"] == [
        "AggreFact-CNN",
        "AggreFact-CNN",
        "RAGTruth",
        "RAGTruth",
    ]
    assert report["per_dataset"] == {
        "AggreFact-CNN": {
            "samples": 2,
            "balanced_accuracy": 1.0,
            "threshold": 0.25,
        },
        "RAGTruth": {"samples": 2, "balanced_accuracy": 1.0, "threshold": 0.25},
    }

    calls = _read_jsonl(call_log)
    assert calls[0]["init"]["model_path"] == "/models/local-logprob.gguf"
    assert calls[0]["init"]["n_ctx"] == 2048
    assert calls[0]["init"]["n_threads"] == 3
    assert calls[0]["init"]["logits_all"] is True
    assert {call["temperature"] for call in calls[1:]} == {0.0}
    assert {call["max_tokens"] for call in calls[1:]} == {4}
    assert {call["logprobs"] for call in calls[1:]} == {True}
    assert {call["top_logprobs"] for call in calls[1:]} == {10}
    prompt_text = "\n".join(str(call["content"]) for call in calls[1:])
    assert "Answer with exactly one word: SUPPORTED or NOT_SUPPORTED." in prompt_text
    assert "Sky is red." in prompt_text
    assert "Fire is cold." in prompt_text


def test_gemma_aggrefact_logprob_cli_rejects_empty_dataset(tmp_path: Path) -> None:
    """Reject empty logprob inputs before reporting or dividing by zero."""
    completed, report_path, _call_log = _run_logprob_cli(tmp_path, empty_dataset=True)

    assert completed.returncode == 1
    assert "dataset is empty" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not report_path.exists()
