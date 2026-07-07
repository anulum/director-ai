# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Paladin-mini benchmark real-surface tests
"""Real subprocess coverage for the Paladin-mini AggreFact benchmark CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "paladin_mini_aggrefact.py"


def _write_protocol_modules(root: Path) -> Path:
    """Write local protocol modules for the benchmark's external ML packages."""
    module_dir = root / "protocol_modules"
    module_dir.mkdir()
    (module_dir / "torch.py").write_text(
        """
from __future__ import annotations

bfloat16 = "bfloat16"


class _NoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def no_grad() -> _NoGrad:
    return _NoGrad()
""".lstrip(),
        encoding="utf-8",
    )
    (module_dir / "datasets.py").write_text(
        """
from __future__ import annotations


class _Dataset:
    def __init__(self, rows):
        self._rows = list(rows)

    def select(self, indices):
        return _Dataset([self._rows[index] for index in indices])

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)


def load_dataset(name, split):
    if name != "lytang/LLM-AggreFact" or split != "test":
        raise AssertionError(f"unexpected dataset request: {name} {split}")
    return _Dataset(
        [
            {"doc": "ctx", "claim": "c1", "label": 1, "dataset": "AggreFact-CNN"},
            {"doc": "ctx", "claim": "c2", "label": 0, "dataset": "AggreFact-CNN"},
            {"doc": "ctx", "claim": "c3", "label": 1, "dataset": "RAGTruth"},
            {"doc": "ctx", "claim": "c4", "label": 0, "dataset": "RAGTruth"},
        ]
    )
""".lstrip(),
        encoding="utf-8",
    )
    (module_dir / "transformers.py").write_text(
        """
from __future__ import annotations


class _Inputs:
    shape = (1, 3)

    def to(self, device):
        return self


class _GeneratedRow:
    def __getitem__(self, key):
        return ["token"]


class _GeneratedOutput:
    def __getitem__(self, index):
        return _GeneratedRow()


class _Tokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return _Inputs()

    def decode(self, token_ids, *, skip_special_tokens):
        return "SUPPORTED"


class _Model:
    device = "cpu"

    def eval(self):
        return None

    def generate(self, inputs, **kwargs):
        return _GeneratedOutput()


class AutoTokenizer:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return _Tokenizer()


class AutoModelForCausalLM:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return _Model()
""".lstrip(),
        encoding="utf-8",
    )
    return module_dir


def _run_benchmark(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run the production benchmark CLI against local protocol modules."""
    module_dir = _write_protocol_modules(tmp_path)
    output = tmp_path / "paladin_mini.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{module_dir}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return subprocess.run(
        [
            sys.executable,
            str(BENCHMARK),
            "--max-samples",
            "4",
            "--output",
            str(output),
            "--log-every",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
        cwd=ROOT,
        env=env,
    )


def _load_result(path: Path) -> dict[str, Any]:
    """Load a benchmark JSON object from ``path``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(dict[str, Any], payload)


def test_paladin_mini_unit_guard_has_real_cli_companion() -> None:
    """The Paladin-mini unit guard should be backed by CLI coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_paladin_mini.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_paladin_mini_real_surface.py" in category


def test_paladin_mini_benchmark_cli_writes_schema_and_metrics(tmp_path: Path) -> None:
    """The production benchmark CLI should write comparable AggreFact metrics."""
    result = _run_benchmark(tmp_path)

    assert result.returncode == 0, result.stderr
    payload = _load_result(tmp_path / "paladin_mini.json")
    assert payload["model"] == "qualifire/context-grounding-paladin-mini"
    assert payload["backend"] == "transformers"
    assert payload["samples"] == 4
    assert payload["global_balanced_accuracy"] == 0.5
    assert payload["predictions"] == [1, 1, 1, 1]
    assert payload["labels"] == [1, 0, 1, 0]
    assert payload["datasets_per_sample"] == [
        "AggreFact-CNN",
        "AggreFact-CNN",
        "RAGTruth",
        "RAGTruth",
    ]
    assert payload["unknown_predictions"] == 0
    assert payload["per_dataset"] == {
        "AggreFact-CNN": {"balanced_accuracy": 0.5, "samples": 2},
        "RAGTruth": {"balanced_accuracy": 0.5, "samples": 2},
    }
