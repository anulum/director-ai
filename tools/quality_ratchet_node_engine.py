# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — verified Node engine artefacts for the TypeScript profile
"""Bind the TypeScript profile to the verified Node binary, per platform.

The profile identity names the expected digest of the ``node`` binary for
each supported platform, extracted from that platform's official release
tarball, together with the tarball's digest as the signature-verified
``SHASUMS256.txt`` lists it. The identity is therefore the same on every
host. At run time the binary that actually executes, the resolved
``process.execPath`` of the running process, must match its platform's
entry. A platform without an entry is refused by name and never falls back
to another platform's entry.

Only ``linux-x64`` is listed: it is the platform of the workstation, the lab
server and the hosted runners. Another platform joins through a declared
tool succession with its own verification record.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

NODE_ARTEFACTS: dict[str, dict[str, str]] = {
    "linux-x64": {
        "binary": "d931c649850ffa7155b8ceffb400dc916ef325866e509cc52ae36e33e8fbb9a9",
        "tarball": "5c063e46bab0fbc1c17cc445d7ff129cbdcd0471be6ed83b6ab864f8143bc864",
    },
}
"""Expected Node 26.13.1 binary and release tarball digests by platform."""

_PROBE = "process.stdout.write(JSON.stringify([process.platform, process.arch, process.execPath]))"


def node_engine(node: str, root: Path, *, timeout: float) -> tuple[str, str]:
    """Return the running Node's platform and its verified binary digest.

    Parameters
    ----------
    node : str
        The Node executable to run.
    root : Path
        The repository root, the probe's working directory, as for every
        other native call of the profile.
    timeout : float
        Positive seconds for the probe.

    Returns
    -------
    tuple[str, str]
        The platform, for example ``linux-x64``, and the executed binary's
        SHA-256, which equals that platform's :data:`NODE_ARTEFACTS` entry.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If the probe fails, the platform has no verified artefact, or the
        executed binary differs from it.
    """
    probe = subprocess.run(
        [node, "-e", _PROBE],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if probe.returncode != 0 or probe.stderr:
        raise ValueError("Native Node engine probe failed")
    value = json.loads(probe.stdout)
    if (
        not isinstance(value, list)
        or len(value) != 3
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError("Native Node engine probe returned an invalid identity")
    platform = f"{value[0]}-{value[1]}"
    expected = NODE_ARTEFACTS.get(platform)
    if expected is None:
        raise ValueError(f"Node platform {platform} has no verified engine artefact")
    digest = hashlib.sha256(Path(value[2]).resolve(strict=True).read_bytes())
    if digest.hexdigest() != expected["binary"]:
        raise ValueError("The running Node binary differs from its verified artefact")
    return platform, digest.hexdigest()
