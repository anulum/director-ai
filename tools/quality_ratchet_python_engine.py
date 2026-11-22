# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — verified Python engine artefacts for the notebook profile
"""Bind the notebook profile to the verified Python build, per platform.

The notebook projection runs inside the ratchet's own interpreter, and its
behaviour depends on that interpreter's tokenizer: Python 3.12 changed how
``tokenize`` reports a leading ``!``. The profile identity therefore names,
for each supported platform, the digests of the interpreter binary and of
the pure-Python ``tokenize.py`` and ``token.py`` it imports, all taken from
one python-build-standalone release archive, together with that archive's
digest. The release's published ``SHA256SUMS`` is the archive's verification
record, and every job checks the archive against the committed pin in
``tools/install_quality_python.sh`` before extracting it.

At run time both interpreters of the profile are checked against their
platform's entry: the running interpreter, which projects the notebooks, and
the one that runs MyPy. The binary checked is the one that executes,
``sys.executable`` resolved; in the symlinked virtual environments the jobs
create it is the archive's ``bin/python3.11``. A platform without an entry
is refused by name and never falls back to another platform's entry. The
check reads no environment variable and has no test mode or override.

Only ``linux-x64`` is listed: it is the platform of the workstation, the lab
server and the hosted runners. Another platform joins through a declared
tool succession with its own verification record.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import token
import tokenize
from pathlib import Path

PYTHON_ARTEFACTS: dict[str, dict[str, str]] = {
    "linux-x64": {
        "archive": "224ca8be2d28841e7a382a27c51e74f4ed0aabe249a79294684e782363165948",
        "binary": "ff9b2ac4362b81c68622ffb1730f7138eb269177d9846399c521a44c5daffb0e",
        "token": "2cabb6a81cafa84b475fc852c20b6bad45955ec0f88063720a831495215f776b",
        "tokenize": "79aa94ebe1761811ea9b2da2ada409a8cb9abd7c6eac241d0c181a36684bb609",
    },
}
"""Expected CPython 3.11.17 build 20261021 archive and file digests by platform."""

ENGINE_FILES = ("binary", "tokenize", "token")
"""The run-time-checked members of each platform entry, in checking order."""

PROBE_CODE = (
    "import json, os, platform, sys, token, tokenize; "
    "print(json.dumps([sys.platform, platform.machine(), "
    "os.path.realpath(sys.executable), "
    "os.path.realpath(tokenize.__file__), os.path.realpath(token.__file__)]))"
)
"""Report an interpreter's platform and the files :data:`ENGINE_FILES` names."""

_ARCHITECTURES = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}


def python_platform(system: str, machine: str) -> str:
    """Name a platform the way :data:`PYTHON_ARTEFACTS` keys it.

    Parameters
    ----------
    system : str
        ``sys.platform`` of the interpreter, for example ``linux``.
    machine : str
        ``platform.machine()`` of the interpreter, for example ``x86_64``.

    Returns
    -------
    str
        The platform, for example ``linux-x64``; an unknown machine keeps
        its own lower-case name, so it never matches another entry.
    """
    architecture = machine.lower()
    return f"{system}-{_ARCHITECTURES.get(architecture, architecture)}"


def _verify(name: str, files: dict[str, str], who: str) -> str:
    """Check one interpreter's files against its platform's entry."""
    expected = PYTHON_ARTEFACTS.get(name)
    if expected is None:
        raise ValueError(f"Python platform {name} has no verified engine artefact")
    for member in ENGINE_FILES:
        path = Path(files[member]).resolve(strict=True)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[member]:
            raise ValueError(
                f"The {who} Python {member} differs from its verified artefact"
            )
    return expected["binary"]


def running_engine() -> tuple[str, str]:
    """Check the interpreter that runs the notebook projection.

    Returns
    -------
    tuple[str, str]
        The platform, for example ``linux-x64``, and the executed binary's
        SHA-256, which equals that platform's :data:`PYTHON_ARTEFACTS` entry.

    Raises
    ------
    ValueError, OSError
        If the platform has no verified artefact, or the interpreter binary,
        ``tokenize.py`` or ``token.py`` differs from it.
    """
    name = python_platform(sys.platform, platform.machine())
    files = {
        "binary": sys.executable,
        "tokenize": tokenize.__file__,
        "token": token.__file__,
    }
    return name, _verify(name, files, "running")


def probed_engine(result: subprocess.CompletedProcess[str]) -> tuple[str, str]:
    """Check the interpreter that runs MyPy from its :data:`PROBE_CODE` report.

    Parameters
    ----------
    result : subprocess.CompletedProcess[str]
        The completed ``python -I -B -c PROBE_CODE`` process.

    Returns
    -------
    tuple[str, str]
        The probed interpreter's platform and its binary's SHA-256.

    Raises
    ------
    ValueError, OSError
        If the probe failed or returned a malformed report, the platform has
        no verified artefact, or a checked file differs from it.
    """
    if result.returncode != 0 or result.stderr:
        raise ValueError("Native Python engine probe failed")
    value = json.loads(result.stdout)
    if (
        not isinstance(value, list)
        or len(value) != 5
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError("Native Python engine probe returned an invalid identity")
    name = python_platform(value[0], value[1])
    files = dict(zip(ENGINE_FILES, value[2:], strict=True))
    return name, _verify(name, files, "MyPy")
