# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Real subprocess coverage for torch device selection."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_subprocess_selector_skips_unsupported_visible_cuda_device(
    tmp_path: Path,
) -> None:
    """Public selector should pick the first CUDA device supported by torch."""
    fake_torch_root = tmp_path / "fake_torch"
    _write_fake_torch(
        fake_torch_root,
        available=True,
        capabilities=[(6, 1), (8, 6)],
        arches=["sm_70", "sm_80", "sm_86"],
    )

    completed = _run_device_script(
        fake_torch_root,
        """
        import json
        from director_ai.core._device import select_torch_device

        print(json.dumps({"device": select_torch_device()}))
        """,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"device": "cuda:1"}


def test_subprocess_selector_force_cpu_and_cache_release(
    tmp_path: Path,
) -> None:
    """Force-CPU selection should coexist with public CUDA cache release."""
    fake_torch_root = tmp_path / "fake_torch"
    cache_marker = tmp_path / "empty_cache_called.txt"
    _write_fake_torch(
        fake_torch_root,
        available=True,
        capabilities=[(8, 0)],
        arches=["sm_80"],
        empty_cache_marker=cache_marker,
    )

    completed = _run_device_script(
        fake_torch_root,
        """
        import json
        from director_ai.core._device import release_torch_cuda, select_torch_device

        selected = select_torch_device("cuda:0")
        release_torch_cuda()
        print(json.dumps({"device": selected}))
        """,
        extra_env={"DIRECTOR_FORCE_CPU": "1"},
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"device": "cpu"}
    assert cache_marker.read_text(encoding="utf-8") == "called"


def _run_device_script(
    fake_torch_root: Path,
    script: str,
    *,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("DIRECTOR_FORCE_CPU", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(fake_torch_root),
            str(REPO_ROOT / "src"),
            env.get("PYTHONPATH", ""),
        ]
    )
    if extra_env is not None:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_fake_torch(
    root: Path,
    *,
    available: bool,
    capabilities: list[tuple[int, int]],
    arches: list[str],
    empty_cache_marker: Path | None = None,
) -> None:
    package = root / "torch"
    package.mkdir(parents=True)
    marker = "" if empty_cache_marker is None else str(empty_cache_marker)
    module_source = f"""
        from pathlib import Path

        _ARCHES = {_python_literal(arches)}
        _AVAILABLE = {_python_literal(available)}
        _CAPABILITIES = {_python_literal(capabilities)}
        _EMPTY_CACHE_MARKER = {_python_literal(marker)}


        class _Cuda:
            def is_available(self):
                return _AVAILABLE

            def device_count(self):
                return len(_CAPABILITIES)

            def get_arch_list(self):
                return list(_ARCHES)

            def get_device_capability(self, index=0):
                return tuple(_CAPABILITIES[index])

            def empty_cache(self):
                if _EMPTY_CACHE_MARKER:
                    Path(_EMPTY_CACHE_MARKER).write_text("called", encoding="utf-8")


        cuda = _Cuda()
        """
    (package / "__init__.py").write_text(
        textwrap.dedent(module_source),
        encoding="utf-8",
    )


def _python_literal(value: object) -> str:
    return repr(value)
