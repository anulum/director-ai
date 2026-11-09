# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — native TypeScript generator recipe execution
"""Bound native generator execution and exact filesystem input identities."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path


def _remaining(deadline: float) -> float:
    """Refuse exhausted shared budgets, including inventories and final custody."""
    value = deadline - time.monotonic()
    if value <= 0:
        raise ValueError(
            "Generated TypeScript verification exhausted its whole deadline"
        )
    return value


def _digest(path: Path, deadline: float) -> str:
    """Hash complete bytes while keeping the caller's deadline in force."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            _remaining(deadline)
            result.update(chunk)
    return result.hexdigest()


def _identity(value: object) -> str:
    """Hash a complete deterministic JSON representation, excluding NaN values."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _directory(path: Path) -> Path:
    """Require the exact existing physical directory rather than a symbolic alias."""
    absolute = path.absolute()
    if absolute.resolve(strict=True) != absolute or not absolute.is_dir():
        raise ValueError(
            "Generated TypeScript directory is not an exact real directory"
        )
    return absolute


def _dependencies(root: Path, deadline: float) -> dict[str, str]:
    """Bind every installed dependency byte and internal npm link to the recipe."""
    root = _directory(root)
    files: dict[str, str] = {}
    for directory, directories, names in os.walk(root, followlinks=False):
        base = Path(directory)
        _remaining(deadline)
        for name in directories + names:
            path = base / name
            metadata = path.lstat()
            relative = path.relative_to(root).as_posix()
            if stat.S_ISLNK(metadata.st_mode):
                if not path.resolve(strict=True).is_relative_to(root):
                    raise ValueError(
                        "Installed TypeScript dependency link leaves its graph"
                    )
                files[relative] = _identity({"link": os.readlink(path)})
            elif stat.S_ISREG(metadata.st_mode):
                files[relative] = _digest(path, deadline)
            elif not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("Installed TypeScript dependency is nonregular")
    if not files:
        raise ValueError("Installed TypeScript dependency graph is empty")
    return dict(sorted(files.items()))


def run_native(
    arguments: list[str],
    cwd: Path,
    deadline: float,
    environment: dict[str, str],
    receipt: Path,
    *,
    accepted_returncodes: tuple[int, ...] = (0,),
    record_output: bool = True,
) -> str:
    """Execute a native recipe command under a shared deadline and keep its receipt.

    Parameters
    ----------
    arguments : list[str]
        Actual argument vector, without shell evaluation.
    cwd : Path
        Real native working directory.
    deadline : float
        Caller-owned absolute monotonic deadline shared with input verification.
        Termination and output cleanup use at most five additional seconds.
    environment : dict[str, str]
        Explicit child environment with task-owned temporary and cache locations.
    receipt : Path
        New original command receipt path; existing files are never replaced.
    accepted_returncodes : tuple[int, ...]
        Native success statuses. Generators use only zero; Git optional metadata
        explicitly permits its documented absent-metadata status.
    record_output : bool
        Whether stdout and stderr may enter the receipt. Sensitive metadata probes
        suppress both until the caller validates a safe context value.

    Returns
    -------
    str
        Actual stdout, without trimming or synthesis.

    Raises
    ------
    ValueError, OSError
        If execution, deadline, output custody or the native result refuses.
    KeyboardInterrupt
        After bounded group cleanup and retaining the interruption receipt.
        The receipt records whether the leader was reaped and streams completed.

    Notes
    -----
    Termination targets the launched process group. Descendants that establish
    another session remain outside that group; their inherited output streams
    cannot extend the cleanup deadline. Callers retain responsibility for those
    escaped descendants.
    """
    _remaining(deadline)
    process = subprocess.Popen(
        arguments,
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    expired = False
    interruption: KeyboardInterrupt | None = None
    output_complete = True
    try:
        stdout, stderr = process.communicate(
            timeout=max(0.0, deadline - time.monotonic())
        )
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        expired = isinstance(error, subprocess.TimeoutExpired)
        if isinstance(error, KeyboardInterrupt):
            interruption = error
        cleanup_deadline = time.monotonic() + 5
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(
                timeout=max(0.0, cleanup_deadline - time.monotonic() - 1)
            )
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            try:
                stdout, stderr = process.communicate(
                    timeout=max(0.0, cleanup_deadline - time.monotonic())
                )
            except subprocess.TimeoutExpired as pending:
                output_complete = False
                stdout = (pending.stdout or b"").decode(errors="replace")
                stderr = (pending.stderr or b"").decode(errors="replace")
                assert process.stdout is not None and process.stderr is not None
                process.stdout.close()
                process.stderr.close()
    reaped = process.poll() is not None
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "argv": arguments,
                "cwd": str(cwd),
                "pid": process.pid,
                "exit": process.returncode,
                "stdout": stdout if record_output else None,
                "stderr": stderr if record_output else None,
                "whole_deadline_expired": expired,
                "interrupted": interruption is not None,
                "output_complete": output_complete,
                "reaped": reaped,
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    if interruption is not None:
        raise interruption
    if (
        expired
        or not output_complete
        or not reaped
        or process.returncode not in accepted_returncodes
    ):
        raise ValueError("Actual native TypeScript generator did not qualify")
    return stdout


@dataclass(frozen=True)
class NativeToolchain:
    """Capture the actual selected executable and complete installed SDK graph.

    Attributes
    ----------
    executable : Path
        Real selected Node executable, after resolving its command lookup.
    executable_sha256 : str
        Exact executable or deliberately selected launcher bytes.
    executable_mode : int
        Original executable permission bits, independently of its byte digest.
    dependencies : dict[str, str]
        Every installed regular file and internal npm link identity, by real path.
        Link values bind the original link target; ordinary values bind file bytes.
    """

    executable: Path
    executable_sha256: str
    executable_mode: int
    dependencies: dict[str, str]


def capture_toolchain(sdk: Path, node: str, *, deadline: float) -> NativeToolchain:
    """Capture real installed inputs without claiming native compatibility.

    Parameters
    ----------
    sdk : Path
        Physical SDK installation owning its real node_modules directory.
    node : str
        Selected executable command or explicit launcher path.
    deadline : float
        Absolute monotonic deadline shared with native verification.

    Returns
    -------
    NativeToolchain
        Complete byte identities. The caller must still run native metadata and
        generators, qualify the profile and revalidate custody before acceptance.

    Raises
    ------
    ValueError, OSError
        If tools, complete dependency bytes or the deadline cannot qualify.
    """
    dependencies = _dependencies(sdk / "node_modules", deadline)
    name = shutil.which(node)
    if name is None:
        raise ValueError("Actual native Node executable is absent")
    executable = Path(name).resolve(strict=True)
    digest = _digest(executable, deadline)
    _remaining(deadline)
    return NativeToolchain(
        executable, digest, stat.S_IMODE(executable.stat().st_mode), dependencies
    )
