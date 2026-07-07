# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real subprocess coverage for the routed Gemma AggreFact HiSS evaluator."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PROJECT_ROOT / "benchmarks" / "gemma_aggrefact_hiss_routed.py"


def _write_protocol_modules(
    module_dir: Path,
    *,
    dataset_rows: list[dict[str, object]],
) -> Path:
    """Write local modules that preserve the dataset and llama-cpp protocols."""
    module_dir.mkdir(parents=True, exist_ok=True)
    rows_json = json.dumps(dataset_rows)
    call_log = module_dir / "llama_calls.jsonl"
    (module_dir / "datasets.py").write_text(
        f"""
from __future__ import annotations

ROWS = {rows_json}


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
    return Dataset(ROWS)
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

    def create_chat_completion(self, *, messages, max_tokens, temperature):
        content = messages[0]["content"]
        event = {{
            "max_tokens": max_tokens,
            "temperature": temperature,
            "content": content,
        }}
        with CALL_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\\n")
        if "Break the CLAIM" in content:
            claim = content.split("CLAIM:", 1)[1].split("Sub-claims:", 1)[0].strip()
            return {{
                "choices": [
                    {{
                        "message": {{
                            "content": "1. " + claim + "\\n2. " + claim + " corroborating detail"
                        }}
                    }}
                ]
            }}
        verdict = "NOT_SUPPORTED" if "claim_no_" in content else "SUPPORTED"
        return {{"choices": [{{"message": {{"content": verdict}}}}]}}
""",
        encoding="utf-8",
    )
    return call_log


def _run_hiss_routed_cli(
    tmp_path: Path,
    *,
    dataset_rows: list[dict[str, object]],
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    """Run the routed HiSS evaluator script with local protocol modules."""
    protocol_dir = tmp_path / "protocol"
    call_log = _write_protocol_modules(protocol_dir, dataset_rows=dataset_rows)
    report_path = tmp_path / "hiss_routed_report.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(protocol_dir), str(PROJECT_ROOT / "benchmarks"), env.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--model",
            "/models/local-hiss-routed.gguf",
            "--max-samples",
            str(max(len(dataset_rows), 1)),
            "--output",
            str(report_path),
            "--n-ctx",
            "2048",
            "--n-threads",
            "3",
            "--log-every",
            "2",
            "--min-decompose-words",
            "8",
            "--support-frac",
            "0.5",
            "--max-subclaims",
            "2",
            "--max-decompose-tokens",
            "44",
            "--max-verify-tokens",
            "7",
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


def _hiss_routed_rows() -> list[dict[str, object]]:
    """Return routed HiSS rows spanning short, long, and fallback schemas."""
    return [
        {
            "doc": "summary source supports the concise claim",
            "claim": "claim_yes_summ",
            "label": 1,
            "dataset": "AggreFact-CNN",
        },
        {
            "document": "summary source rejects the concise claim",
            "hypothesis": "claim_no_summ",
            "annotations": 0,
            "dataset": "AggreFact-CNN",
        },
        {
            "doc": "retrieved context supports a longer generated answer",
            "claim": (
                "claim_yes_rag includes several grounded details that should be "
                "decomposed for verification"
            ),
            "label": 1,
            "dataset": "RAGTruth",
        },
        {
            "document": "claim evidence rejects a longer statement",
            "hypothesis": (
                "claim_no_claim includes several unsupported details that should be "
                "decomposed for verification"
            ),
            "annotations": 0,
            "dataset": "Wice",
        },
    ]


def test_gemma_aggrefact_hiss_routed_unit_guard_has_real_cli_companion() -> None:
    """Ensure the routed HiSS unit guard is backed by this subprocess companion."""
    assert KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_gemma_aggrefact_hiss_routed.py"
    ] == (
        "unit-guard-with-companion",
        "ML/export/eval Gemma AggreFact HiSS routed evaluator guard with companion "
        "tests/test_gemma_aggrefact_hiss_routed_real_surface.py",
    )


def test_gemma_aggrefact_hiss_routed_cli_routes_and_decomposes(
    tmp_path: Path,
) -> None:
    """Exercise routed HiSS evaluation through the benchmark CLI contract."""
    completed, report_path, call_log = _run_hiss_routed_cli(
        tmp_path,
        dataset_rows=_hiss_routed_rows(),
    )

    assert completed.returncode == 0, completed.stderr
    report = _read_json(report_path)
    assert report["schema_version"] == 2
    assert report["model"] == "/models/local-hiss-routed.gguf"
    assert report["samples"] == 4
    assert report["min_decompose_words"] == 8
    assert report["support_frac"] == 0.5
    assert report["max_subclaims"] == 2
    assert report["skipped_decompose"] == 2
    assert report["global_balanced_accuracy"] == 1.0
    assert report["predictions"] == [1, 0, 1, 0]
    assert report["labels"] == [1, 0, 1, 0]
    assert report["support_fractions"] == [1.0, 0.0, 1.0, 0.0]
    assert report["subclaim_counts"] == [1, 1, 2, 2]
    assert report["decomposed_flags"] == [False, False, True, True]
    assert report["families_per_sample"] == ["summ", "summ", "rag", "claim"]
    assert report["per_family"] == {
        "claim": {"samples": 1, "balanced_accuracy": 0.0},
        "rag": {"samples": 1, "balanced_accuracy": 0.0},
        "summ": {"samples": 2, "balanced_accuracy": 1.0},
    }

    calls = _read_jsonl(call_log)
    assert calls[0]["init"]["model_path"] == "/models/local-hiss-routed.gguf"
    assert calls[0]["init"]["n_ctx"] == 2048
    assert calls[0]["init"]["n_threads"] == 3
    assert {call["temperature"] for call in calls[1:]} == {0.0}
    assert {call["max_tokens"] for call in calls[1:3]} == {7}
    assert {call["max_tokens"] for call in calls[3::3]} == {44}
    prompt_text = "\n".join(str(call["content"]) for call in calls[1:])
    assert "careful summarisation evaluator" in prompt_text
    assert "retrieval-augmented generation outputs" in prompt_text
    assert "fact-checking assistant" in prompt_text
    assert "Break the CLAIM" in prompt_text


def test_gemma_aggrefact_hiss_routed_cli_rejects_empty_dataset(
    tmp_path: Path,
) -> None:
    """Reject empty routed HiSS inputs before writing an invalid report."""
    completed, report_path, _call_log = _run_hiss_routed_cli(
        tmp_path,
        dataset_rows=[],
    )

    assert completed.returncode == 1
    assert "dataset is empty" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not report_path.exists()
