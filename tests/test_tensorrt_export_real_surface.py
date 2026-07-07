# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - TensorRT export real-surface tests
"""Real-surface coverage for TensorRT export CLI and public wiring."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from director_ai.core import export_tensorrt as core_export_tensorrt
from director_ai.core.nli import export_tensorrt as compatibility_export_tensorrt
from director_ai.core.scoring.nli import export_tensorrt as runtime_export_tensorrt
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_onnxruntime_protocol_module(directory: Path) -> None:
    """Create a local ONNX Runtime protocol module for CLI subprocess export."""
    (directory / "onnxruntime.py").write_text(
        "\n".join(
            [
                "import json",
                "import os",
                "",
                "class GraphOptimizationLevel:",
                "    ORT_ENABLE_ALL = 'ORT_ENABLE_ALL'",
                "",
                "def get_available_providers():",
                "    return [",
                "        'TensorrtExecutionProvider',",
                "        'CUDAExecutionProvider',",
                "        'CPUExecutionProvider',",
                "    ]",
                "",
                "class SessionOptions:",
                "    def __init__(self):",
                "        self.graph_optimization_level = None",
                "        self.log_severity_level = None",
                "",
                "class _Input:",
                "    def __init__(self, name):",
                "        self.name = name",
                "",
                "class InferenceSession:",
                "    def __init__(self, model_file, opts, providers):",
                "        self.model_file = model_file",
                "        self.opts = opts",
                "        self.providers = providers",
                "",
                "    def get_providers(self):",
                "        return ['TensorrtExecutionProvider']",
                "",
                "    def get_inputs(self):",
                "        return [_Input('input_ids'), _Input('attention_mask')]",
                "",
                "    def run(self, outputs, feed):",
                "        record = {",
                "            'model_file': self.model_file,",
                "            'graph_optimization_level': self.opts.graph_optimization_level,",
                "            'log_severity_level': self.opts.log_severity_level,",
                "            'providers': self.providers,",
                "            'feed': {",
                "                key: {'shape': list(value.shape), 'dtype': str(value.dtype)}",
                "                for key, value in feed.items()",
                "            },",
                "        }",
                "        with open(os.environ['DIRECTOR_TRT_PROTOCOL_RECORD'], 'w', encoding='utf-8') as handle:",
                "            json.dump(record, handle, default=str, sort_keys=True)",
                "        return []",
            ],
        ),
        encoding="utf-8",
    )


def _write_transformers_protocol_module(directory: Path) -> None:
    """Create a local Transformers protocol module for tokenizer warmup."""
    (directory / "transformers.py").write_text(
        "\n".join(
            [
                "import numpy as np",
                "",
                "class _Tokenizer:",
                "    def __call__(",
                "        self,",
                "        texts,",
                "        *,",
                "        return_tensors,",
                "        truncation,",
                "        padding,",
                "        max_length,",
                "    ):",
                "        if return_tensors != 'np':",
                "            raise AssertionError(return_tensors)",
                "        batch = len(texts)",
                "        width = min(max_length, 4)",
                "        return {",
                "            'input_ids': np.ones((batch, width), dtype=np.int32),",
                "            'attention_mask': np.ones((batch, width), dtype=np.int64),",
                "            'ignored': np.ones((batch, width), dtype=np.int64),",
                "        }",
                "",
                "class AutoTokenizer:",
                "    @staticmethod",
                "    def from_pretrained(path, revision, local_files_only):",
                "        if revision != 'local-artifact':",
                "            raise AssertionError(revision)",
                "        if local_files_only is not True:",
                "            raise AssertionError(local_files_only)",
                "        return _Tokenizer()",
            ],
        ),
        encoding="utf-8",
    )


def test_tensorrt_export_unit_guard_declares_real_surface_companion() -> None:
    """The TensorRT unit guard should name this real CLI companion surface."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_tensorrt_export.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_tensorrt_export_real_surface.py" in reason


def test_public_tensorrt_export_paths_share_runtime_exporter() -> None:
    """Public compatibility export paths should resolve to one runtime function."""
    assert core_export_tensorrt is runtime_export_tensorrt
    assert compatibility_export_tensorrt is runtime_export_tensorrt


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("max_batch", 0),
        ("max_batch", True),
        ("max_seq_len", 0),
        ("max_seq_len", False),
        ("warmup_pairs", 0),
        ("warmup_pairs", True),
    ],
)
def test_public_tensorrt_export_rejects_invalid_profile_config(
    parameter: str,
    value: int,
) -> None:
    """TensorRT export should reject impossible profile settings early."""
    with pytest.raises(ValueError, match=parameter):
        if parameter == "max_batch":
            runtime_export_tensorrt(onnx_dir="unused", max_batch=value)
        elif parameter == "max_seq_len":
            runtime_export_tensorrt(onnx_dir="unused", max_seq_len=value)
        else:
            runtime_export_tensorrt(onnx_dir="unused", warmup_pairs=value)


def test_tensorrt_export_cli_builds_cache_with_documented_onnx_dir(
    tmp_path: Path,
) -> None:
    """Run the production TensorRT export CLI through local protocol modules."""
    protocol_dir = tmp_path / "protocol_modules"
    protocol_dir.mkdir()
    _write_onnxruntime_protocol_module(protocol_dir)
    _write_transformers_protocol_module(protocol_dir)
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    (onnx_dir / "model.onnx").write_bytes(b"onnx")
    output_dir = tmp_path / "trt-cache"
    record_path = tmp_path / "trt-record.json"
    env = {
        **os.environ,
        "DIRECTOR_TRT_PROTOCOL_RECORD": str(record_path),
        "PYTHONPATH": os.pathsep.join([str(protocol_dir), str(REPO_ROOT / "src")]),
    }

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "director_ai.cli",
            "export",
            "--format",
            "tensorrt",
            "--onnx-dir",
            str(onnx_dir),
            "--output",
            str(output_dir),
            "--no-fp16",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 0, completed.stderr
    assert output_dir.is_dir()
    assert f"Building TensorRT engine cache from {onnx_dir}" in completed.stdout
    assert f"Done. Cache at {output_dir}." in completed.stdout
    record = json.loads(record_path.read_text(encoding="utf-8"))
    provider_name, provider_options = record["providers"][0]
    assert provider_name == "TensorrtExecutionProvider"
    assert provider_options["trt_fp16_enable"] is False
    assert provider_options["trt_engine_cache_path"] == str(output_dir)
    assert provider_options["trt_profile_min_shapes"] == (
        "input_ids=1x1,attention_mask=1x1"
    )
    assert record["model_file"] == str(onnx_dir / "model.onnx")
    assert record["feed"] == {
        "attention_mask": {"dtype": "int64", "shape": [4, 4]},
        "input_ids": {"dtype": "int64", "shape": [4, 4]},
    }
