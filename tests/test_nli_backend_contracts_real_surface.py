# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — NLI backend real-surface contract tests
"""Real subprocess coverage for the public NLI backend scorer contract."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TypedDict, cast

from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class NliProtocolResult(TypedDict):
    """JSON payload emitted by the subprocess scorer probe."""

    available: bool
    cost: float
    scores: list[float]
    single: float
    tokens: int
    with_confidence: list[list[float]]


def _write_protocol_modules(module_dir: Path) -> Path:
    """Write protocol-compatible ``torch`` and ``transformers`` modules."""
    module_dir.mkdir(parents=True, exist_ok=True)
    event_log = module_dir / "nli_protocol_events.jsonl"
    (module_dir / "torch.py").write_text(
        """
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

float16 = "float16"
bfloat16 = "bfloat16"
float32 = "float32"


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return self

    def numel(self):
        return int(self.values.size)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


@contextmanager
def no_grad():
    yield


def softmax(tensor, dim):
    values = tensor.values if isinstance(tensor, Tensor) else np.asarray(tensor)
    shifted = values - values.max(axis=dim, keepdims=True)
    exp = np.exp(shifted)
    return Tensor(exp / exp.sum(axis=dim, keepdims=True))
""",
        encoding="utf-8",
    )
    (module_dir / "transformers.py").write_text(
        f"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

EVENT_LOG = Path({str(event_log)!r})


def _record(event):
    with EVENT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, default=str, sort_keys=True) + "\\n")


class _Tokenizer:
    def __call__(self, *args, **kwargs):
        _record({{"event": "tokenize", "args": args, "kwargs": kwargs}})
        rows = []
        if len(args) == 2 and isinstance(args[0], list):
            pairs = list(zip(args[0], args[1], strict=True))
        elif len(args) == 2:
            pairs = [(args[0], args[1])]
        elif len(args) == 1 and isinstance(args[0], list):
            pairs = [(text, text) for text in args[0]]
        else:
            pairs = [(args[0], args[0])]
        for _premise, hypothesis in pairs:
            text = str(hypothesis)
            if "red" in text or "void" in text:
                rows.append([2, 1, 1])
            else:
                rows.append([1, 1, 1])
        return {{
            "input_ids": torch.Tensor(rows),
            "attention_mask": torch.Tensor([[1, 1, 1] for _ in rows]),
        }}


class _Model:
    def __init__(self):
        self.config = SimpleNamespace(
            id2label={{0: "entailment", 1: "neutral", 2: "contradiction"}}
        )
        self._device = "cpu"

    def parameters(self):
        return iter([SimpleNamespace(device=self._device)])

    def to(self, device):
        self._device = device
        _record({{"event": "model_to", "device": device}})
        return self

    def eval(self):
        _record({{"event": "model_eval"}})
        return self

    def __call__(self, **inputs):
        rows = inputs["input_ids"].values
        logits = []
        for row in rows:
            if int(row[0]) == 2:
                logits.append([-4.0, 0.0, 4.0])
            else:
                logits.append([4.0, 0.0, -4.0])
        return SimpleNamespace(logits=torch.Tensor(logits))


class AutoTokenizer:
    @staticmethod
    def from_pretrained(model_source, **kwargs):
        _record(
            {{"event": "tokenizer_load", "model_source": model_source, "kwargs": kwargs}}
        )
        return _Tokenizer()


class AutoModelForSequenceClassification:
    @staticmethod
    def from_pretrained(model_source, **kwargs):
        if os.environ.get("DIRECTOR_TEST_NLI_MODEL_FAIL") == "1":
            raise RuntimeError("protocol model load failed")
        _record(
            {{"event": "model_load", "model_source": model_source, "kwargs": kwargs}}
        )
        return _Model()
""",
        encoding="utf-8",
    )
    return event_log


def _run_protocol_scorer(
    tmp_path: Path,
    *,
    fail_model: bool = False,
) -> tuple[subprocess.CompletedProcess[str], NliProtocolResult | None, Path]:
    """Run the public NLI scorer API in a subprocess."""
    protocol_dir = tmp_path / "protocol"
    event_log = _write_protocol_modules(protocol_dir)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(protocol_dir), str(PROJECT_ROOT / "src"), env.get("PYTHONPATH", "")]
    )
    if fail_model:
        env["DIRECTOR_TEST_NLI_MODEL_FAIL"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from __future__ import annotations

import json

from director_ai.core.scoring.nli import NLIScorer, nli_available

scorer = NLIScorer(
    model_name="protocol-nli",
    revision="abcdef123456",
    backend="deberta",
    device="cpu",
    torch_dtype="float32",
)
pairs = [
    ("The sky is blue.", "The sky is blue."),
    ("The sky is blue.", "The sky is red."),
]
scores = scorer.score_batch(pairs)
with_confidence = scorer.score_batch_with_confidence(pairs)
single = scorer.score("The contract is signed.", "The contract is void.")
print(
    json.dumps(
        {
            "available": nli_available(),
            "cost": scorer.last_estimated_cost,
            "scores": scores,
            "single": single,
            "tokens": scorer.last_token_count,
            "with_confidence": with_confidence,
        },
        sort_keys=True,
    )
)
""",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return completed, None, event_log
    return (
        completed,
        cast(NliProtocolResult, json.loads(completed.stdout)),
        event_log,
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read JSONL protocol events from disk."""
    return [
        cast(dict[str, object], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_nli_backend_contract_unit_guard_has_real_surface_companion() -> None:
    """Ensure the NLI backend unit guard has this public scorer companion."""
    assert KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_nli_backend_contracts.py"
    ] == (
        "unit-guard-with-companion",
        "ML/export/eval NLI backend guard with companion "
        "tests/test_nli_backend_contracts_real_surface.py",
    )


def test_nli_public_scorer_uses_protocol_model_backend(tmp_path: Path) -> None:
    """Exercise the public NLI scorer through protocol dependency modules."""
    completed, payload, event_log = _run_protocol_scorer(tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert payload is not None
    assert payload["available"] is True
    assert payload["scores"][0] < 0.05
    assert payload["scores"][1] > 0.95
    assert payload["single"] > 0.95
    assert payload["tokens"] == 15
    assert payload["cost"] > 0.0
    assert payload["with_confidence"][0][0] < 0.05
    assert payload["with_confidence"][0][1] > 0.85
    assert payload["with_confidence"][1][0] > 0.95
    assert payload["with_confidence"][1][1] > 0.85
    assert "Traceback" not in completed.stderr

    events = _read_jsonl(event_log)
    assert events[0] == {
        "event": "tokenizer_load",
        "kwargs": {"revision": "abcdef123456", "use_fast": False},
        "model_source": "protocol-nli",
    }
    assert events[1]["event"] == "model_load"
    assert cast(dict[str, object], events[1]["kwargs"])["torch_dtype"] == "float32"
    assert events[2] == {"device": "cpu", "event": "model_to"}
    assert events[3] == {"event": "model_eval"}
    tokenize_events = [event for event in events if event["event"] == "tokenize"]
    assert len(tokenize_events) == 3
    first_tokenize_kwargs = cast(dict[str, object], tokenize_events[0]["kwargs"])
    assert first_tokenize_kwargs["padding"] is True
    assert first_tokenize_kwargs["return_tensors"] == "pt"
    assert first_tokenize_kwargs["max_length"] == 512


def test_nli_public_scorer_falls_back_when_protocol_model_load_fails(
    tmp_path: Path,
) -> None:
    """Keep the public scorer usable when the model backend cannot load."""
    completed, payload, _event_log = _run_protocol_scorer(tmp_path, fail_model=True)

    assert completed.returncode == 0, completed.stderr
    assert payload is not None
    assert payload["available"] is True
    assert payload["scores"] == [0.2, 0.275]
    assert payload["with_confidence"] == [[0.2, 0.5], [0.275, 0.5]]
    assert payload["single"] == 0.275
    assert payload["tokens"] == 0
    assert "protocol model load failed" in completed.stderr
    assert "Traceback" not in completed.stderr
