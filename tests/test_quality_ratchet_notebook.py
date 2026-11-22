# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — native notebook typing adapter refusals
"""Refuse an unqualified notebook analysis before, during and after MyPy.

The checks made before the engine check run in process. The changes made
while the analysis runs need the pinned interpreter, so the real command
runs under ``DIRECTOR_QUALITY_PYTHON`` while this process changes the files
it reads, timed by inotify events on those files.
"""

from __future__ import annotations

import ctypes
import os
import select
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tests.quality_ratchet_python_support import quality_python
from tests.test_quality_ratchet_notebook_cli import _cli, _write
from tests.test_quality_ratchet_typing_cli import commit_typing_sources
from tests.test_quality_ratchet_typing_cli import typing_repository as typing_repository
from tools.quality_ratchet_contracts import Finding
from tools.quality_ratchet_notebook import _rebind, analyse_notebooks
from tools.quality_ratchet_notebook_source import NotebookSource

IN_CLOSE_NOWRITE = 0x10
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf")])
def test_the_analysis_needs_a_positive_finite_deadline(timeout: float) -> None:
    """Refuse a deadline that cannot bound the analysis."""
    with pytest.raises(ValueError, match="deadline must be positive and finite"):
        analyse_notebooks(ROOT, sys.executable, lambda _root: (), timeout=timeout)


def test_a_selection_without_notebooks_is_refused() -> None:
    """Refuse a selection that names no notebook."""
    with pytest.raises(ValueError, match="source inventory is empty"):
        analyse_notebooks(
            ROOT, sys.executable, lambda _root: ("tools/__init__.py",), timeout=60
        )


def test_a_position_past_the_projection_is_refused() -> None:
    """Refuse a MyPy line that has no original notebook cell line."""
    projection = NotebookSource("value = 1\n", ((1, 1),), frozenset(), ())
    finding = Finding("generated.py", "assignment", 2, 1, "message", None)
    with pytest.raises(ValueError, match="has no original cell line"):
        _rebind((finding,), {"generated.py": ("notebooks/a.ipynb", projection)}, set())


def test_a_mypy_interpreter_other_than_the_pinned_build_is_refused(
    typing_repository: Path, tmp_path: Path
) -> None:
    """Refuse by name a MyPy interpreter that is not the verified build."""
    _write(typing_repository, 'label: str = "score"')
    commit_typing_sources(typing_repository)
    output = tmp_path / "capture.json"
    result = _cli(
        typing_repository,
        "capture",
        "--output",
        str(output),
        "--python",
        sys.executable,
    )
    assert result.returncode == 2
    assert "The MyPy Python binary differs from its verified artefact" in result.stderr
    assert not output.exists()


def watch_reads(path: Path) -> tuple[int, Callable[[int], None]]:
    """Watch one file for completed reads.

    Parameters
    ----------
    path : Path
        The file.

    Returns
    -------
    tuple[int, Callable[[int], None]]
        The inotify descriptor, and a function waiting for that many more
        completed reads.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    libc.inotify_init1.argtypes = [ctypes.c_int]
    libc.inotify_init1.restype = ctypes.c_int
    libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    libc.inotify_add_watch.restype = ctypes.c_int
    descriptor = int(libc.inotify_init1(os.O_CLOEXEC))
    assert descriptor >= 0, ctypes.get_errno()
    watch = int(libc.inotify_add_watch(descriptor, os.fsencode(path), IN_CLOSE_NOWRITE))
    assert watch >= 0, ctypes.get_errno()

    def wait(count: int) -> None:
        """Wait for ``count`` completed reads; each event is 16 bytes here."""
        seen = 0
        while seen < count:
            ready, _, _ = select.select([descriptor], [], [], 60)
            assert ready, "The analysis never read the watched file"
            seen += len(os.read(descriptor, 4096)) // 16

    return descriptor, wait


def replace(path: Path) -> None:
    """Replace a file's bytes with a changed copy.

    Parameters
    ----------
    path : Path
        The file.
    """
    replacement = path.with_name(path.name + ".next")
    replacement.write_bytes(path.read_bytes() + b"\n")
    replacement.replace(path)


@pytest.mark.parametrize(
    ("reads", "message"),
    [
        (1, "Notebook source changed before native analysis"),
        (2, "Notebook source, generator, configuration or environment changed"),
    ],
)
def test_a_notebook_changed_while_it_is_analysed_is_refused(
    typing_repository: Path, tmp_path: Path, reads: int, message: str
) -> None:
    """Refuse a notebook changed after its first read, or after its second."""
    notebook = _write(typing_repository, 'label: str = "score"')
    commit_typing_sources(typing_repository)
    output = tmp_path / "capture.json"
    descriptor, wait = watch_reads(notebook)
    try:

        def change() -> None:
            """Change the notebook once the analysis has read it enough."""
            wait(reads)
            replace(notebook)

        with ThreadPoolExecutor(max_workers=1) as executor:
            job = executor.submit(change)
            result = _cli(typing_repository, "capture", "--output", str(output))
            job.result(timeout=60)
    finally:
        os.close(descriptor)
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert not output.exists()


def test_a_generated_input_changed_while_mypy_runs_is_refused(
    typing_repository: Path, tmp_path: Path
) -> None:
    """Refuse a generated compiler input that changes before the analysis ends."""
    _write(typing_repository, 'label: str = "score"')
    commit_typing_sources(typing_repository)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    output = tmp_path / "capture.json"

    def change() -> None:
        """Append to the generated input once the adapter has written it."""
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            generated = sorted(scratch.glob("quality-notebooks-*/notebook_*.py"))
            if generated and generated[0].stat().st_size:
                with generated[0].open("ab") as stream:
                    stream.write(b"\n")
                return
            time.sleep(0.005)
        raise AssertionError("The analysis never wrote its generated input")

    with ThreadPoolExecutor(max_workers=1) as executor:
        job = executor.submit(change)
        result = subprocess.run(
            [
                quality_python(),
                "-B",
                "-m",
                "tools.check_quality_ratchet",
                "--profile",
                "notebook-types",
                "--root",
                str(typing_repository),
                "capture",
                "--output",
                str(output),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env={**os.environ, "TMPDIR": str(scratch)},
        )
        job.result(timeout=60)
    assert result.returncode == 2
    assert "Generated notebook compiler inputs changed" in result.stderr, result.stderr
    assert not output.exists()


SWAPPED = """
import sys
from pathlib import Path
import tools.quality_ratchet_notebook as adapter
real = adapter.probed_engine
probes = []
def probed(result):
    probes.append(result)
    if len(probes) > 1:
        raise ValueError("The MyPy Python binary differs from its verified artefact")
    return real(result)
adapter.probed_engine = probed
try:
    adapter.capture_notebook_typing(Path(sys.argv[1]), sys.executable)
except ValueError as error:
    print(len(probes), error)
"""


def test_the_mypy_interpreter_is_checked_again_after_the_analysis(
    typing_repository: Path,
) -> None:
    """Refuse a MyPy interpreter that no longer passes when the analysis ends."""
    _write(typing_repository, 'label: str = "score"')
    commit_typing_sources(typing_repository)
    result = subprocess.run(
        [quality_python(), "-B", "-c", SWAPPED, str(typing_repository)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        "2 The MyPy Python binary differs from its verified artefact"
    )
