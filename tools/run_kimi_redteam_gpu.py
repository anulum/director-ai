# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — JarvisLabs GPU runner for the KIMI red-team reproduction
"""Provision a JarvisLabs GPU, run the KIMI red-team reproduction, tear down.

Provisions a small GPU via the current jlclient API, has the remote
git-clone the tag and run the detection-efficacy harness
(``benchmarks/kimi_redteam_reproduction.py``) in a fresh Python 3.11 conda
env, downloads the JSON artefact, and destroys the instance in a ``finally``
block so a failure never leaks a paid GPU. The remote recipe (region, GPU
availability, py3.11, ensurepip) is encoded in :data:`REMOTE_SCRIPT`.

Usage::

    export JARVISLABS_TOKEN=...
    python tools/run_kimi_redteam_gpu.py --gpu A30 --tag v3.18.1
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("DirectorAI.KimiRedteamGPU")

REPO_DIR = Path(__file__).resolve().parent.parent

# jlclient 0.1 (git HEAD) auto-resolves RTX-class GPUs to a deprecated India
# region; the live non-Europe region is india-noida-01 (Europe only offers
# expensive H100/H200). Pin it explicitly for a small, cheap GPU.
NOIDA_REGION = "india-noida-01"

# The JarvisLabs pytorch template ships Python 3.10 (miniconda base +
# py3.10 env), but director-ai requires >=3.11. So the remote recipe builds a
# fresh 3.11 env via conda-forge (the default anaconda channel needs a ToS
# accept and fails), bootstraps pip with ensurepip (conda-forge's minimal
# python has no pip), git-clones the tag (rsyncing the 15 GB working tree is
# wasteful — the repo's committed datasets alone are ~3.5 GB), and runs the
# harness. An editable install carries proxy.py + all source (the wheel
# exclude does not apply to a source install).
REMOTE_SCRIPT = r"""#!/usr/bin/env bash
set -uo pipefail
echo "=== KIMI red-team reproduction (JarvisLabs) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || echo "no GPU?"
rm -rf /home/director-ai
git clone --depth 1 --branch __TAG__ https://github.com/anulum/director-ai /home/director-ai
cd /home/director-ai
GIT_SHA="$(git rev-parse HEAD)"
echo "cloned commit: $GIT_SHA"
CONDA=/root/miniconda3/bin/conda
$CONDA create -n py311 python=3.11 -y -c conda-forge --override-channels
CR="$CONDA run -n py311 --no-capture-output"
$CR python -m ensurepip --upgrade
$CR python -m pip install --quiet "torch>=2.8,<3" "transformers>=5.0.0rc3,<6" numpy requests
$CR python -m pip install --quiet -e . --no-deps
$CR python -c "import torch; print('cuda:', torch.cuda.is_available())"
$CR python __SCRIPT__ --out /home/director-ai/kimi_repro.json --git-sha "$GIT_SHA"
echo "=== done; artefact at /home/director-ai/kimi_repro.json ==="
"""


def provision(gpu_type: str, storage: int, token: str) -> Any:
    """Create a GPU instance and return the running Instance object.

    Uses the current jlclient API directly (``Instance.create`` returns an
    Instance on success or an ``{'error_message': ...}`` dict on failure) and
    pins :data:`NOIDA_REGION` because auto-resolution picks a deprecated region.
    """
    from jlclient import jarvisclient
    from jlclient.jarvisclient import Instance

    jarvisclient.token = token
    logger.info("Creating %s x1 (%dGB) in %s...", gpu_type, storage, NOIDA_REGION)
    inst = Instance.create(
        "GPU",
        gpu_type=gpu_type,
        template="pytorch",
        num_gpus=1,
        storage=storage,
        name="director-ai-kimi-redteam",
        is_reserved=True,
        duration="hour",
        region=NOIDA_REGION,
    )
    if isinstance(inst, dict):
        raise RuntimeError(
            f"instance creation failed: {inst.get('error_message', inst)}"
        )

    logger.info("Instance machine_id=%s; waiting for Running...", inst.machine_id)
    for i in range(60):
        inst._refresh()
        if getattr(inst, "status", "") == "Running" and getattr(inst, "ssh_str", ""):
            logger.info("Running. SSH: %s", inst.ssh_str)
            return inst
        logger.info("  status=%s (%d/60)...", getattr(inst, "status", "?"), i + 1)
        time.sleep(10)
    raise TimeoutError("instance did not reach Running in 10 minutes")


def _ssh_parts(ssh_str: str) -> tuple[str, str]:
    parts = ssh_str.split()
    port = parts[parts.index("-p") + 1] if "-p" in parts else "22"
    return port, parts[-1]


def _render_remote_script(tag: str, script_path: str) -> str:
    """Fill the remote recipe for *tag* and *script_path*.

    The remote resolves the exact cloned commit (``git rev-parse HEAD``) and
    passes it to the benchmark as ``--git-sha`` so the artefact records the true
    commit that produced the numbers, not just the (movable) tag.
    """
    return REMOTE_SCRIPT.replace("__TAG__", tag).replace("__SCRIPT__", script_path)


def run_redteam(ssh_str: str, tag: str, script_path: str) -> None:
    """Write and execute the red-team script on the remote, streaming output.

    The remote clones *tag* from GitHub itself, so no local upload is needed.
    *script_path* is the repo-relative benchmark run on the GPU; it must accept
    ``--out`` and ``--git-sha`` and write its JSON artefact there.
    """
    port, host = _ssh_parts(ssh_str)
    ssh_base = f"ssh -p {port} -o StrictHostKeyChecking=no {host}"
    script = _render_remote_script(tag, script_path)
    remote = "/tmp/run_kimi_redteam.sh"
    subprocess.run(
        f"{ssh_base} 'cat > {remote}' << 'REMOTE_EOF'\n{script}\nREMOTE_EOF",
        shell=True,
        check=True,
    )
    subprocess.run(f'{ssh_base} "chmod +x {remote}"', shell=True, check=True)
    logger.info("Running red-team on the GPU (streaming)...")
    subprocess.run(f'{ssh_base} "bash {remote}"', shell=True, check=True)


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


def _live_machine_ids(list_instances: Callable[[], list[Any]]) -> set[str]:
    """Return the ``machine_id``\\ s currently live at source (JarvisLabs)."""
    return {str(getattr(inst, "machine_id", "")) for inst in list_instances()}


def destroy_instance(
    inst: Any,
    *,
    list_instances: Callable[[], list[Any]] | None = None,
    retries: int = 3,
    backoff_seconds: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Destroy the instance and confirm at source that it is really gone.

    A lone ``inst.destroy()`` is not a safe teardown: the JarvisLabs call can
    raise a transient network error (e.g. ``RemoteDisconnected``) *after* the
    request is on the wire, so the destroy looks failed while the paid GPU keeps
    running — the exact way an A30 once leaked. So this retries on failure and,
    critically, re-verifies liveness at source via *list_instances*
    (``User.get_instances`` by default): the instance counts as destroyed only
    once its ``machine_id`` no longer appears in the live set. Returns ``True``
    on a confirmed teardown, ``False`` if the GPU may still be leaked (logged
    ``CRITICAL`` so a watcher or operator destroys it manually).
    """
    machine_id = str(getattr(inst, "machine_id", "?"))
    if list_instances is None:
        from jlclient.jarvisclient import User

        list_instances = User.get_instances

    for attempt in range(1, retries + 1):
        try:
            result = inst.destroy()
            logger.info(
                "Instance %s destroy attempt %d/%d: %s",
                machine_id,
                attempt,
                retries,
                result,
            )
        except Exception as exc:  # noqa: BLE001 - teardown must never raise
            logger.error(
                "Instance %s destroy attempt %d/%d raised: %s",
                machine_id,
                attempt,
                retries,
                exc,
            )

        try:
            still_live = machine_id in _live_machine_ids(list_instances)
        except Exception as exc:  # noqa: BLE001 - fail closed on an unverifiable teardown
            logger.error(
                "Instance %s liveness re-check failed: %s (assuming still live)",
                machine_id,
                exc,
            )
            still_live = True

        if not still_live:
            logger.info(
                "Instance %s confirmed destroyed (0-live at source)", machine_id
            )
            return True
        if attempt < retries:
            sleep(backoff_seconds)

    logger.critical(
        "LEAKED GPU: instance %s STILL LIVE after %d destroy attempts — "
        "DESTROY MANUALLY IN THE UI",
        machine_id,
        retries,
    )
    return False


def main(argv: list[str] | None = None) -> int:
    """Provision, run, download, and always destroy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu", default="A30", help="JarvisLabs GPU type (A30/L4/A100 in noida)"
    )
    parser.add_argument("--storage", type=int, default=20)
    parser.add_argument("--tag", default="v3.18.1", help="git tag to clone + run")
    parser.add_argument(
        "--script",
        default="benchmarks/kimi_redteam_reproduction.py",
        help="repo-relative benchmark to run on the GPU (must accept --out)",
    )
    parser.add_argument("--out", default="kimi_redteam_reproduction.json")
    args = parser.parse_args(argv)

    token = os.environ.get("JARVISLABS_TOKEN")
    if not token:
        logger.error("JARVISLABS_TOKEN not set")
        return 2

    inst = provision(args.gpu, args.storage, token)
    teardown_confirmed = True
    try:
        run_redteam(inst.ssh_str, args.tag, args.script)
        download_artefact(inst.ssh_str, args.out)
    finally:
        teardown_confirmed = destroy_instance(inst)
    return 0 if teardown_confirmed else 3


if __name__ == "__main__":
    raise SystemExit(main())
