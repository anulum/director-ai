# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — actual concurrent Go source and configuration custody tests
"""Refuse real custody changes and native build failures without process substitutes."""

from __future__ import annotations

import ctypes
import os
import select
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tests.test_quality_ratchet_golang import go_repository as go_repository
from tests.test_quality_ratchet_golang import go_tools as go_tools
from tools.quality_ratchet_golang import capture_go_documentation
from tools.quality_ratchet_golang_protocol import qualify_go_build_report


@pytest.mark.parametrize("mutation", ("source", "configuration", "inventory"))
def test_concurrent_go_custody_changes_refuse(
    go_repository: Path, go_tools: tuple[str, str], mutation: str
) -> None:
    """Reject actual changes after identity capture while original native tools execute.

    Parameters
    ----------
    go_repository : Path
        Actual selected Git repository with original module configuration.
    go_tools : tuple[str, str]
        Actual Go compiler and Revive executable.
    mutation : str
        Maintained source, module configuration or inventory change.
    """
    root = go_repository
    target = (
        root
        / {
            "source": "gateway/go/value.go",
            "configuration": "gateway/go/go.mod",
            "inventory": "gateway/go/added.go",
        }[mutation]
    )
    payload = {
        "source": (root / "gateway/go/value.go").read_bytes() + b"\n",
        "configuration": (root / "gateway/go/go.mod").read_bytes() + b"\n",
        "inventory": b"package fixture\n// added returns another count.\nfunc added() int {return 2}\n",
    }[mutation]
    libc = ctypes.CDLL(None, use_errno=True)
    libc.inotify_init1.argtypes = [ctypes.c_int]
    libc.inotify_init1.restype = ctypes.c_int
    libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    libc.inotify_add_watch.restype = ctypes.c_int
    descriptor = int(libc.inotify_init1(os.O_CLOEXEC))
    assert descriptor >= 0, ctypes.get_errno()
    try:
        watched = "gateway/go/value.go" if mutation == "source" else "gateway/go/go.sum"
        watch = int(
            libc.inotify_add_watch(descriptor, os.fsencode(root / watched), 0x10)
        )
        assert watch >= 0, ctypes.get_errno()

        def mutate_after_original_read() -> None:
            """Wait for the actual final configuration read, then atomically replace bytes."""
            for _ in range(2 if mutation == "source" else 1):
                ready, _, _ = select.select([descriptor], [], [], 10)
                assert ready, "Collector never completed the original native read"
                assert os.read(descriptor, 4096)
            replacement = target.with_name(target.name + ".next")
            replacement.write_bytes(payload)
            replacement.replace(target)

        with ThreadPoolExecutor(max_workers=1) as executor:
            mutation_job = executor.submit(mutate_after_original_read)
            with pytest.raises(ValueError, match="changed during"):
                capture_go_documentation(root, *go_tools)
            mutation_job.result(timeout=10)
        assert target.read_bytes() == payload
    finally:
        os.close(descriptor)


def test_actual_unusable_go_build_cache_refuses_publication(
    go_repository: Path, go_tools: tuple[str, str], tmp_path: Path
) -> None:
    """Keep an actual native build failure from becoming a complete quality snapshot.

    Parameters
    ----------
    go_repository : Path
        Actual original source repository.
    go_tools : tuple[str, str]
        Actual Go compiler and Revive executable.
    tmp_path : Path
        Runner-owned allocation containing a real nondirectory build-cache path.
    """
    cache = tmp_path / "not-a-cache-directory"
    cache.write_bytes(b"An actual regular file cannot serve as the Go build cache.\n")
    helper = Path(__file__).resolve().parents[1] / "tools/quality_ratchet_go"
    built = subprocess.run(
        [
            go_tools[0],
            "build",
            "-o",
            str(tmp_path / "parser"),
            str(helper / "main.go"),
            str(helper / "documentation.go"),
        ],
        env=dict(os.environ, GOCACHE=str(cache)),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert built.returncode != 0 and built.stderr
    with pytest.raises(ValueError, match="could not be built"):
        qualify_go_build_report(built)
    command = (
        "import sys; from pathlib import Path; "
        "from tools.quality_ratchet_golang import capture_go_documentation; "
        "capture_go_documentation(Path(sys.argv[1]), sys.argv[2], sys.argv[3])"
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", command, str(go_repository), *go_tools],
        env=dict(os.environ, GOCACHE=str(cache)),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode != 0 and not completed.stdout
    assert "Native Go documentation parser could not be built" in completed.stderr
