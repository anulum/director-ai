# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — JarvisLabs GPU runner for the KIMI red-team reproduction
"""Provision a JarvisLabs GPU, run the KIMI red-team reproduction, tear down.

Reuses the provisioning + rsync mechanics from
:mod:`tools.jarvislabs_train` but runs the detection-efficacy harness
(``benchmarks/kimi_redteam_reproduction.py``) against the current source
instead of training, then downloads the JSON artefact. The instance is
destroyed in a ``finally`` block so a failure never leaks a paid GPU.

Usage::

    export JARVISLABS_TOKEN=...
    python tools/run_kimi_redteam_gpu.py --gpu RTX5000 --out kimi_repro.json
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jarvislabs_train import provision_instance, upload_code  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("DirectorAI.KimiRedteamGPU")

REPO_DIR = Path(__file__).resolve().parent.parent

REMOTE_SCRIPT = r"""#!/usr/bin/env bash
set -euo pipefail
echo "=== KIMI red-team reproduction (JarvisLabs) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || echo "no GPU?"
cd /home/director-ai
python -m pip install --quiet --upgrade pip
# Editable install carries proxy.py + all source (the wheel exclude does not
# apply to an editable/source install), plus the NLI extra for FactCG.
python -m pip install --quiet -e '.[nli]'
python benchmarks/kimi_redteam_reproduction.py --out /home/director-ai/kimi_repro.json
echo "=== done; artefact at /home/director-ai/kimi_repro.json ==="
"""


def _ssh_parts(ssh_str: str) -> tuple[str, str]:
    parts = ssh_str.split()
    port = parts[parts.index("-p") + 1] if "-p" in parts else "22"
    return port, parts[-1]


def run_redteam(ssh_str: str) -> None:
    """Write and execute the red-team script on the remote, streaming output."""
    port, host = _ssh_parts(ssh_str)
    ssh_base = f"ssh -p {port} -o StrictHostKeyChecking=no {host}"
    script_path = "/home/director-ai/run_kimi_redteam.sh"
    subprocess.run(
        f"{ssh_base} 'cat > {script_path}' << 'REMOTE_EOF'\n{REMOTE_SCRIPT}\nREMOTE_EOF",
        shell=True,
        check=True,
    )
    subprocess.run(f'{ssh_base} "chmod +x {script_path}"', shell=True, check=True)
    logger.info("Running red-team on the GPU (streaming)...")
    subprocess.run(f'{ssh_base} "bash {script_path}"', shell=True, check=True)


def download_artefact(ssh_str: str, out: str) -> None:
    """Download the JSON artefact from the remote."""
    port, host = _ssh_parts(ssh_str)
    dest = REPO_DIR / "benchmarks" / "results" / out
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        f"scp -P {port} -o StrictHostKeyChecking=no "
        f"{host}:/home/director-ai/kimi_repro.json {dest}",
        shell=True,
        check=True,
    )
    logger.info("Artefact downloaded to %s", dest)


def destroy_instance(instance_id: int) -> None:
    """Destroy the instance so a paid GPU never leaks."""
    try:
        from jlclient.jarvisclient import User

        inst = User.get_instance(instance_id=instance_id)
        inst.destroy()
        logger.info("Instance %s destroyed.", instance_id)
    except Exception as exc:  # noqa: BLE001 - teardown must never raise
        logger.error(
            "FAILED to destroy instance %s: %s — DESTROY MANUALLY", instance_id, exc
        )


def main(argv: list[str] | None = None) -> int:
    """Provision, run, download, and always destroy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="RTX5000", help="JarvisLabs GPU type")
    parser.add_argument("--storage", type=int, default=20)
    parser.add_argument("--out", default="kimi_redteam_reproduction.json")
    args = parser.parse_args(argv)

    token = os.environ.get("JARVISLABS_TOKEN")
    if not token:
        logger.error("JARVISLABS_TOKEN not set")
        return 2

    info = provision_instance(gpu_type=args.gpu, storage=args.storage, token=token)
    instance_id = info["instance_id"]
    try:
        upload_code(info["ssh_str"], hf_token=os.environ.get("HF_TOKEN", ""))
        run_redteam(info["ssh_str"])
        download_artefact(info["ssh_str"], args.out)
    finally:
        destroy_instance(instance_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
