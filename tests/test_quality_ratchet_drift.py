# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — actual concurrent native source custody tests
"""Mutate real source custody after the collectors finish their initial byte reads."""

from __future__ import annotations

import ctypes
import os
import select
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tools.quality_ratchet_native import capture_python_documentation
from tools.quality_ratchet_typing import capture_python_typing


@pytest.fixture
def native_repository(tmp_path: Path) -> Path:
    """Create a real Git repository accepted by both actual native collectors.

    Parameters
    ----------
    tmp_path : Path
        Runner-owned isolated allocation.

    Returns
    -------
    Path
        Actual repository with original maintained source and configuration.
    """
    root = tmp_path / "native-drift"
    root.mkdir()
    subprocess.run(
        ["git", "init", "--initial-branch=main"],
        cwd=root,
        capture_output=True,
        check=True,
        timeout=10,
    )
    (root / "src").mkdir()
    (root / "requirements").mkdir()
    (root / "src/original.py").write_text(
        '"""Actual maintained source."""\nVALUE: int = 1\n'
    )
    (root / "pyproject.toml").write_text("[tool.mypy]\nstrict=true\n")
    (root / "requirements/ci-lint.txt").write_text("ruff==0.16.10\n")
    (root / "requirements/ci-dev.txt").write_text("mypy==1.20.2\n")
    (root / "requirements/ci-types.txt").write_text("types-requests==2.33.0.20260906\n")
    return root


@pytest.mark.parametrize("profile", ("documentation", "typing"))
@pytest.mark.parametrize("mutation", ("source", "configuration", "inventory"))
def test_actual_concurrent_custody_changes_refuse(
    native_repository: Path,
    profile: str,
    mutation: str,
) -> None:
    """Refuse genuine changes after initial capture while running unmodified tools.

    Parameters
    ----------
    native_repository : Path
        Actual selected repository.
    profile : str
        Native Ruff or MyPy collector under test.
    mutation : str
        Actual source bytes, configuration bytes or source membership change.
    """
    root = native_repository
    watched = {
        "documentation": "requirements/ci-lint.txt",
        "typing": "requirements/ci-types.txt",
    }[profile]
    target = (
        root
        / {
            "source": "src/original.py",
            "configuration": "pyproject.toml",
            "inventory": "src/added.py",
        }[mutation]
    )
    payload = {
        "source": (root / "src/original.py").read_bytes() + b"\n",
        "configuration": (root / "pyproject.toml").read_bytes() + b"\n",
        "inventory": b'"""Actual added source."""\nVALUE: int = 2\n',
    }[mutation]
    libc = ctypes.CDLL(None, use_errno=True)
    libc.inotify_init1.argtypes = [ctypes.c_int]
    libc.inotify_init1.restype = ctypes.c_int
    libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    libc.inotify_add_watch.restype = ctypes.c_int
    descriptor = int(libc.inotify_init1(os.O_CLOEXEC))
    assert descriptor >= 0, ctypes.get_errno()
    try:
        watch = int(
            libc.inotify_add_watch(descriptor, os.fsencode(root / watched), 0x10)
        )
        assert watch >= 0, ctypes.get_errno()

        def mutate_after_original_read() -> None:
            """Wait for the actual initial close-read event, then change owned bytes."""
            ready, _, _ = select.select([descriptor], [], [], 10)
            assert ready, "Collector never completed its original identity read"
            assert os.read(descriptor, 4096)
            replacement = target.with_name(target.name + ".next")
            replacement.write_bytes(payload)
            replacement.replace(target)

        with ThreadPoolExecutor(max_workers=1) as executor:
            mutation_job = executor.submit(mutate_after_original_read)
            with pytest.raises(ValueError, match="changed during"):
                {
                    "documentation": capture_python_documentation,
                    "typing": capture_python_typing,
                }[profile](root, sys.executable)
            mutation_job.result(timeout=10)
        assert target.read_bytes() == payload
    finally:
        os.close(descriptor)
