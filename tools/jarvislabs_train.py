# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — JarvisLabs GPU Training Orchestrator

"""Provision a JarvisLabs GPU instance, upload code, run fine-tuning,
download results, and destroy the instance.

Usage::

    export JARVISLABS_TOKEN=...
    export HF_TOKEN=hf_...
    python tools/jarvislabs_train.py

    # Or specify GPU type:
    python tools/jarvislabs_train.py --gpu A6000 --storage 50
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("DirectorAI.JarvisLabsTrain")

REPO_DIR = Path(__file__).resolve().parent.parent

# Training script to execute on remote GPU
REMOTE_TRAIN_SCRIPT = r"""#!/usr/bin/env bash
set -euo pipefail

echo "=== Director-AI GPU Training Pipeline (JarvisLabs) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' || echo 'N/A')"
echo ""

cd /home/director-ai

# Install
echo ">>> Installing Director-AI..."
pip install -e ".[finetune,nli]" --quiet 2>&1 | tail -3
pip install datasets nltk --quiet 2>&1 | tail -1

# Prepare data
echo ">>> Preparing training data..."
python tools/prepare_finetune_data.py --dataset all --eval-ratio 0.1

for f in data/aggrefact_train.jsonl data/aggrefact_eval.jsonl \
         data/medical_train.jsonl data/medical_eval.jsonl \
         data/legal_train.jsonl data/legal_eval.jsonl; do
    if [ -f "$f" ]; then
        echo "  OK: $f ($(wc -l < "$f") lines)"
    else
        echo "  MISSING: $f"
    fi
done

mkdir -p models

# Run 1: General factuality
echo ""
echo "=== Run 1/3: General Factuality (AggreFact) ==="
START1=$(date +%s)
director-ai finetune data/aggrefact_train.jsonl \
    --eval data/aggrefact_eval.jsonl \
    --output models/factcg-aggrefact \
    --epochs 3 --lr 2e-5 --batch-size 16 2>&1 | tee models/run1_aggrefact.log
END1=$(date +%s)
echo "Run 1 time: $(( (END1-START1)/60 ))m $(( (END1-START1)%60 ))s"

# Run 2: Medical domain
echo ""
echo "=== Run 2/3: Medical Domain (MedNLI + PubMedQA) ==="
START2=$(date +%s)
director-ai finetune data/medical_train.jsonl \
    --eval data/medical_eval.jsonl \
    --output models/factcg-medical \
    --epochs 3 --lr 2e-5 --batch-size 16 2>&1 | tee models/run2_medical.log
END2=$(date +%s)
echo "Run 2 time: $(( (END2-START2)/60 ))m $(( (END2-START2)%60 ))s"

# Run 3: Legal domain
echo ""
echo "=== Run 3/3: Legal Domain (ContractNLI) ==="
START3=$(date +%s)
director-ai finetune data/legal_train.jsonl \
    --eval data/legal_eval.jsonl \
    --output models/factcg-legal \
    --epochs 3 --lr 2e-5 --batch-size 16 2>&1 | tee models/run3_legal.log
END3=$(date +%s)
echo "Run 3 time: $(( (END3-START3)/60 ))m $(( (END3-START3)%60 ))s"

# Benchmark
echo ""
echo "=== Benchmarking ==="
python -m benchmarks.aggrefact_eval \
    --model models/factcg-aggrefact --sweep 2>&1 | tee models/bench_aggrefact.log || true

echo ""
echo "=== All training complete ==="
TOTAL_TIME=$(( (END3-START1)/60 ))
echo "Total training time: ${TOTAL_TIME} minutes"
echo "Models in: /home/director-ai/models/"
ls -lh models/

# Create tarball for download
echo ">>> Creating models tarball..."
tar czf /home/director-ai/models_trained.tar.gz -C models .
echo "Download: scp ... /home/director-ai/models_trained.tar.gz"
echo "Size: $(du -h /home/director-ai/models_trained.tar.gz | cut -f1)"
"""


def _run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)


def provision_instance(
    gpu_type: str = "A5000",
    num_gpus: int = 1,
    storage: int = 50,
    token: str | None = None,
) -> dict:
    """Create JarvisLabs GPU instance and wait for it to be ready."""
    from jlclient import jarvisclient
    from jlclient.jarvisclient import Instance, User

    jarvisclient.token = token or os.environ["JARVISLABS_TOKEN"]

    logger.info(
        "Creating JarvisLabs instance: %s x%d, %dGB storage",
        gpu_type,
        num_gpus,
        storage,
    )
    instance = Instance.create(
        "GPU",
        gpu_type=gpu_type,
        num_gpus=num_gpus,
        storage=storage,
        template="pytorch",
        name="director-ai-training",
        is_reserved=True,
    )

    logger.info("Instance created: id=%s", instance.instance_id)
    logger.info("SSH: %s", getattr(instance, "ssh_str", "pending..."))

    # Wait for instance to be running
    for i in range(60):
        inst = User.get_instance(instance_id=instance.instance_id)
        status = getattr(inst, "status", "unknown")
        if status == "Running":
            ssh_str = getattr(inst, "ssh_str", "")
            logger.info("Instance running. SSH: %s", ssh_str)
            return {
                "instance_id": inst.instance_id,
                "ssh_str": ssh_str,
                "status": status,
                "gpu_type": gpu_type,
            }
        logger.info("  Status: %s (waiting %d/60)...", status, i + 1)
        time.sleep(10)

    raise TimeoutError("Instance did not reach Running state in 10 minutes")


def upload_code(ssh_str: str, hf_token: str) -> None:
    """Upload Director-AI code to the instance via rsync/scp."""
    # Parse SSH string: ssh -p PORT user@host
    parts = ssh_str.split()
    port_idx = parts.index("-p") + 1 if "-p" in parts else None
    port = parts[port_idx] if port_idx else "22"
    host = parts[-1]  # user@host

    logger.info("Uploading code to %s (port %s)...", host, port)

    # rsync the repo (exclude heavy/unnecessary dirs)
    rsync_cmd = (
        f'rsync -avz --progress -e "ssh -p {port} -o StrictHostKeyChecking=no" '
        f"--exclude .git --exclude .venv --exclude __pycache__ "
        f'--exclude "*.pyc" --exclude director_guard --exclude .coordination '
        f'--exclude "*.onnx" --exclude "*.pt" --exclude "*.safetensors" '
        f"{REPO_DIR}/ {host}:/home/director-ai/"
    )
    subprocess.run(rsync_cmd, shell=True, check=True)

    # Set HF_TOKEN on remote
    ssh_base = f"ssh -p {port} -o StrictHostKeyChecking=no {host}"
    subprocess.run(
        f'{ssh_base} "echo export HF_TOKEN={hf_token} >> ~/.bashrc"',
        shell=True,
        check=True,
    )
    logger.info("Code uploaded and HF_TOKEN set.")


def run_training(ssh_str: str, hf_token: str) -> None:
    """Execute training on the remote instance."""
    parts = ssh_str.split()
    port_idx = parts.index("-p") + 1 if "-p" in parts else None
    port = parts[port_idx] if port_idx else "22"
    host = parts[-1]

    ssh_base = f"ssh -p {port} -o StrictHostKeyChecking=no {host}"

    # Write training script
    script_path = "/home/director-ai/run_training.sh"
    subprocess.run(
        f"{ssh_base} 'cat > {script_path}' << 'REMOTE_EOF'\n{REMOTE_TRAIN_SCRIPT}\nREMOTE_EOF",
        shell=True,
        check=True,
    )
    subprocess.run(f'{ssh_base} "chmod +x {script_path}"', shell=True, check=True)

    # Run training (streams output)
    logger.info("Starting training (streaming output)...")
    subprocess.run(
        f'{ssh_base} "export HF_TOKEN={hf_token} && bash {script_path}"',
        shell=True,
    )


def download_models(ssh_str: str, local_dir: str = "models") -> None:
    """Download trained models from the instance."""
    parts = ssh_str.split()
    port_idx = parts.index("-p") + 1 if "-p" in parts else None
    port = parts[port_idx] if port_idx else "22"
    host = parts[-1]

    local_path = REPO_DIR / local_dir
    local_path.mkdir(exist_ok=True)

    logger.info("Downloading models to %s...", local_path)
    subprocess.run(
        f"scp -P {port} -o StrictHostKeyChecking=no "
        f"{host}:/home/director-ai/models_trained.tar.gz "
        f"{local_path}/models_trained.tar.gz",
        shell=True,
        check=True,
    )

    # Extract
    subprocess.run(
        f"tar xzf {local_path}/models_trained.tar.gz -C {local_path}",
        shell=True,
        check=True,
    )
    logger.info("Models extracted to %s", local_path)

    # Also grab logs
    for log in [
        "run1_aggrefact.log",
        "run2_medical.log",
        "run3_legal.log",
        "bench_aggrefact.log",
    ]:
        subprocess.run(
            f"scp -P {port} -o StrictHostKeyChecking=no "
            f"{host}:/home/director-ai/models/{log} {local_path}/{log}",
            shell=True,
            check=False,
        )


def destroy_instance(instance_id: int, token: str | None = None) -> None:
    """Destroy the instance to stop billing."""
    from jlclient import jarvisclient
    from jlclient.jarvisclient import User

    jarvisclient.token = token or os.environ["JARVISLABS_TOKEN"]
    inst = User.get_instance(instance_id=instance_id)
    inst.destroy()
    logger.info("Instance %s destroyed.", instance_id)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="JarvisLabs GPU training orchestrator")
    parser.add_argument("--gpu", default="A5000", help="GPU type (A5000, A6000, A100)")
    parser.add_argument("--storage", type=int, default=50, help="Storage in GB")
    parser.add_argument(
        "--skip-provision",
        action="store_true",
        help="Skip provisioning",
    )
    parser.add_argument("--instance-id", type=int, help="Existing instance ID")
    parser.add_argument("--ssh-str", type=str, help="Existing SSH string")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--destroy-only", action="store_true")
    args = parser.parse_args()

    jl_token = os.environ.get("JARVISLABS_TOKEN")
    hf_token = os.environ.get("HF_TOKEN", "")
    if not jl_token:
        print("ERROR: Set JARVISLABS_TOKEN environment variable")
        sys.exit(1)

    if args.destroy_only:
        if not args.instance_id:
            print("ERROR: --instance-id required for --destroy-only")
            sys.exit(1)
        destroy_instance(args.instance_id, jl_token)
        sys.exit(0)

    if args.download_only:
        if not args.ssh_str:
            print("ERROR: --ssh-str required for --download-only")
            sys.exit(1)
        download_models(args.ssh_str)
        sys.exit(0)

    # Full pipeline
    if args.skip_provision:
        if not args.ssh_str or not args.instance_id:
            print("ERROR: --ssh-str and --instance-id required with --skip-provision")
            sys.exit(1)
        info = {"instance_id": args.instance_id, "ssh_str": args.ssh_str}
    else:
        info = provision_instance(
            gpu_type=args.gpu,
            storage=args.storage,
            token=jl_token,
        )

    try:
        upload_code(info["ssh_str"], hf_token)
        run_training(info["ssh_str"], hf_token)
        download_models(info["ssh_str"])
        logger.info("Training complete. Models downloaded.")
    finally:
        yn = input(f"\nDestroy instance {info['instance_id']}? [Y/n] ").strip().lower()
        if yn != "n":
            destroy_instance(info["instance_id"], jl_token)
        else:
            logger.info("Instance kept alive. Remember to destroy it to stop billing!")
            logger.info(
                "  python tools/jarvislabs_train.py --destroy-only --instance-id %s",
                info["instance_id"],
            )
