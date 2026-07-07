# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Sotek. All rights reserved.
# © Code 2020–2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Real subprocess coverage for the routed Gemma AggreFact evaluator."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PROJECT_ROOT / "benchmarks" / "gemma_aggrefact_routed.py"


def _write_protocol_modules(
    module_dir: Path,
    *,
    dataset_rows: list[dict[str, object]],
) -> Path:
    """Write local modules that preserve the external package protocols."""
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

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)


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

    def create_chat_completion(self, *, messages, max_tokens, temperature):
        content = messages[0]["content"]
        event = {{
            "init": self._kwargs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "content": content,
        }}
        with CALL_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\\n")
        verdict = "NOT_SUPPORTED" if "claim_no_" in content else "SUPPORTED"
        return {{"choices": [{{"message": {{"content": verdict}}}}]}}
""",
        encoding="utf-8",
    )
    return call_log


def _run_routed_cli(
    tmp_path: Path,
    *,
    dataset_rows: list[dict[str, object]],
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    """Run the routed evaluator script with local protocol modules."""
    protocol_dir = tmp_path / "protocol"
    call_log = _write_protocol_modules(protocol_dir, dataset_rows=dataset_rows)
    report_path = tmp_path / "routed_report.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(protocol_dir), str(PROJECT_ROOT / "benchmarks"), env.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--model",
            "/models/local-routed.gguf",
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
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed, report_path, call_log


def _routed_rows() -> list[dict[str, object]]:
    """Return routed rows covering summary, RAG, and claim families."""
    return [
        {
            "doc": "source supports the first summary claim",
            "claim": "claim_yes_summ",
            "label": 1,
            "dataset": "AggreFact-CNN",
        },
        {
            "doc": "source rejects the second summary claim",
            "claim": "claim_no_summ",
            "label": 0,
            "dataset": "AggreFact-CNN",
        },
        {
            "doc": "retrieved context supports the first generated answer",
            "claim": "claim_yes_rag",
            "label": 1,
            "dataset": "RAGTruth",
        },
        {
            "doc": "retrieved context rejects the second generated answer",
            "claim": "claim_no_rag",
            "label": 0,
            "dataset": "RAGTruth",
        },
        {
            "doc": "claim evidence supports the first statement",
            "claim": "claim_yes_claim",
            "label": 1,
            "dataset": "Wice",
        },
        {
            "doc": "claim evidence rejects the second statement",
            "claim": "claim_no_claim",
            "label": 0,
            "dataset": "Wice",
        },
    ]


def test_gemma_aggrefact_routed_unit_guard_has_real_cli_companion() -> None:
    """Ensure the routed unit guard is backed by this subprocess companion."""
    assert KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_gemma_aggrefact_routed.py"
    ] == (
        "unit-guard-with-companion",
        "ML/export/eval Gemma AggreFact routed evaluator guard with companion "
        "tests/test_gemma_aggrefact_routed_real_surface.py",
    )


def test_gemma_aggrefact_routed_cli_routes_family_prompts(
    tmp_path: Path,
) -> None:
    """Exercise the routed evaluator through its benchmark CLI contract."""
    completed, report_path, call_log = _run_routed_cli(
        tmp_path,
        dataset_rows=_routed_rows(),
    )

    assert completed.returncode == 0, completed.stderr
    report = cast(dict[str, Any], json.loads(report_path.read_text(encoding="utf-8")))

    assert report["model"] == "/models/local-routed.gguf"
    assert report["samples"] == 6
    assert report["global_balanced_accuracy"] == 1.0
    assert report["predictions"] == [1, 0, 1, 0, 1, 0]
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
    assert report["per_family"] == {
        "claim": {"samples": 2, "balanced_accuracy": 1.0},
        "rag": {"samples": 2, "balanced_accuracy": 1.0},
        "summ": {"samples": 2, "balanced_accuracy": 1.0},
    }
    assert report["per_dataset"] == {
        "AggreFact-CNN": {"samples": 2, "balanced_accuracy": 1.0},
        "RAGTruth": {"samples": 2, "balanced_accuracy": 1.0},
        "Wice": {"samples": 2, "balanced_accuracy": 1.0},
    }

    calls = [
        cast(dict[str, Any], json.loads(line))
        for line in call_log.read_text(encoding="utf-8").splitlines()
    ]
    assert calls[0]["init"]["model_path"] == "/models/local-routed.gguf"
    assert calls[0]["init"]["n_ctx"] == 2048
    assert calls[0]["init"]["n_threads"] == 3
    assert {call["temperature"] for call in calls} == {0.0}
    prompt_text = "\n".join(str(call["content"]) for call in calls)
    assert "careful summarisation evaluator" in prompt_text
    assert "retrieval-augmented generation outputs" in prompt_text
    assert "fact-checking assistant" in prompt_text


def test_gemma_aggrefact_routed_cli_rejects_empty_dataset(tmp_path: Path) -> None:
    """Reject empty routed inputs before reporting or dividing by zero."""
    completed, report_path, _call_log = _run_routed_cli(tmp_path, dataset_rows=[])

    assert completed.returncode == 1
    assert "dataset is empty" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not report_path.exists()
