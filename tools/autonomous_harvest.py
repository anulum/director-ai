# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — autonomous_harvest
#!/usr/bin/env python3
"""Autonomous model harvest: monitor training, download, verify, destroy instances.

Runs until all models are downloaded and all instances destroyed.
Then uploads models to UpCloud, runs ensemble benchmark, downloads results,
and destroys UpCloud.
"""

import json
import os
import subprocess
import time

MODELS_DIR = "C:/aaa_God_of_the_Math_Collection/03_CODE/DIRECTOR_AI/models"
TOOLS_DIR = "C:/aaa_God_of_the_Math_Collection/03_CODE/DIRECTOR_AI/tools"
BENCH_DIR = "C:/aaa_God_of_the_Math_Collection/03_CODE/DIRECTOR_AI/benchmarks"
SSH_KEY = "~/.ssh/id_ed25519_upcloud"
JL_TOKEN = "cgilUVgpHz670xCrdDY0WUwWfR-gAYV-Gsqm6htNkN8"
UPCLOUD_IP = "212.147.242.179"

INSTANCES = [
    {
        "name": "ReCoRD",
        "id": 383719,
        "ssh": "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p 11414 root@sshj.jarvislabs.ai",
        "models": [
            {
                "name": "factcg-record",
                "done_file": "/home/director-ai/RECORD_DONE",
                "result_json": "/home/director-ai/models/factcg-record/training_result.json",
            },
        ],
        "type": "jarvis",
    },
]

TAR_FILES = "model.safetensors config.json tokenizer.json tokenizer_config.json special_tokens_map.json training_result.json training_args.bin"


BASH = r"C:\Program Files\Git\usr\bin\bash.exe"


def run(cmd, timeout=60):
    try:
        r = subprocess.run(
            [BASH, "-c", cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def check_model_done(inst, model):
    """Check if a model's training is complete by looking for training_result.json."""
    ssh = inst["ssh"]
    rj = model["result_json"]
    code, out, _ = run(f'{ssh} "cat {rj} 2>/dev/null"')
    if code == 0 and "COMPLETE" in out:
        return True, out
    return False, ""


def download_model(inst, model):
    """Tar essential files, scp to local, extract, verify."""
    ssh = inst["ssh"]
    name = model["name"]
    remote_dir = f"/home/director-ai/models/{name}"
    local_dir = os.path.join(MODELS_DIR, name)
    tar_name = f"{name}.tar.gz"
    local_tar = os.path.join(MODELS_DIR, tar_name)

    log(f"  Tarring {name} on remote...")
    code, out, err = run(
        f'{ssh} "cd {remote_dir} && rm -rf checkpoint-* && tar czf /tmp/{tar_name} {TAR_FILES} spm.model added_tokens.json 2>/dev/null; du -h /tmp/{tar_name}"',
        timeout=120,
    )
    if code != 0:
        log(f"  TAR FAILED: {err}")
        return False

    log(f"  Downloading {name} ({out})...")
    # Extract scp params from ssh string
    ssh_parts = inst["ssh"].split()
    port_idx = ssh_parts.index("-p") if "-p" in ssh_parts else None
    port = ssh_parts[port_idx + 1] if port_idx else "22"
    host = ssh_parts[-1]  # user@host
    key_flag = ""
    if "-i" in ssh_parts:
        ki = ssh_parts.index("-i")
        key_flag = f"-i {ssh_parts[ki + 1]}"

    code, out, err = run(
        f'scp -o StrictHostKeyChecking=no -P {port} {key_flag} {host}:/tmp/{tar_name} "{local_tar}"',
        timeout=600,
    )
    if code != 0:
        log(f"  SCP FAILED: {err}")
        return False

    log(f"  Extracting {name}...")
    os.makedirs(local_dir, exist_ok=True)
    local_tar_u = local_tar.replace("\\", "/")
    local_dir_u = local_dir.replace("\\", "/")
    code, _, err = run(
        f'tar --force-local -xzf "{local_tar_u}" -C "{local_dir_u}"',
        timeout=120,
    )
    if code != 0:
        log(f"  EXTRACT FAILED: {err}")
        return False

    # Verify
    safetensors = os.path.join(local_dir, "model.safetensors")
    result_json = os.path.join(local_dir, "training_result.json")
    if os.path.exists(safetensors) and os.path.exists(result_json):
        size_mb = os.path.getsize(safetensors) / 1e6
        with open(result_json) as f:
            result = json.load(f)
        bal_acc = result.get("test_balanced_accuracy", 0)
        log(f"  VERIFIED: {name} — {size_mb:.0f}MB, bal_acc={bal_acc:.4f}")
        os.remove(local_tar)
        return True
    log(f"  VERIFICATION FAILED: missing files in {local_dir}")
    return False


def destroy_jarvis(instance_id):
    """Destroy a JarvisLabs instance."""
    code, out, err = run(
        f'python -c "'
        f"import jlclient.jarvisclient as jl; "
        f"jl.token = '{JL_TOKEN}'; "
        f"inst = jl.User.get_instance({instance_id}); "
        f"inst.destroy(); "
        f"print('DESTROYED')"
        f'"',
        timeout=30,
    )
    return "DESTROYED" in out


def get_balance():
    code, out, _ = run(
        f'python -c "'
        f"import jlclient.jarvisclient as jl; "
        f"jl.token = '{JL_TOKEN}'; "
        f"print(jl.User.get_balance())"
        f'"',
        timeout=15,
    )
    return out


def main():
    log("=" * 60)
    log("AUTONOMOUS HARVEST STARTED")
    log(
        f"Monitoring {len(INSTANCES)} instances, {sum(len(i['models']) for i in INSTANCES)} models",
    )
    log("=" * 60)

    completed_instances = set()
    downloaded_models = set()

    # Check what's already local
    for inst in INSTANCES:
        for model in inst["models"]:
            local_st = os.path.join(MODELS_DIR, model["name"], "model.safetensors")
            local_rj = os.path.join(MODELS_DIR, model["name"], "training_result.json")
            if os.path.exists(local_st) and os.path.exists(local_rj):
                downloaded_models.add(model["name"])
                log(f"Already local: {model['name']}")

    # Phase 1: Monitor, download, destroy
    while len(completed_instances) < len(INSTANCES):
        for inst in INSTANCES:
            if inst["name"] in completed_instances:
                continue

            all_done = True
            for model in inst["models"]:
                if model["name"] in downloaded_models:
                    continue

                done, result_str = check_model_done(inst, model)
                if done:
                    log(f"TRAINING COMPLETE: {model['name']} on {inst['name']}")
                    if download_model(inst, model):
                        downloaded_models.add(model["name"])
                    else:
                        log(f"Download failed for {model['name']}, will retry")
                        all_done = False
                else:
                    all_done = False

            if all_done and all(m["name"] in downloaded_models for m in inst["models"]):
                # All models from this instance downloaded — destroy it
                if inst["type"] == "jarvis":
                    log(f"DESTROYING {inst['name']} (ID={inst['id']})...")
                    if destroy_jarvis(inst["id"]):
                        log(f"  DESTROYED {inst['name']}")
                        bal = get_balance()
                        log(f"  JarvisLabs balance: {bal}")
                    else:
                        log(f"  Destroy failed for {inst['name']}, will retry")
                        continue
                completed_instances.add(inst["name"])
                log(f"Instance {inst['name']} fully harvested and destroyed")
            elif all_done:
                log(
                    f"Instance {inst['name']}: some downloads failed, retrying next cycle",
                )

        remaining = [
            i["name"] for i in INSTANCES if i["name"] not in completed_instances
        ]
        if remaining:
            log(f"Waiting 120s... Remaining: {remaining}")
            log(f"Downloaded: {len(downloaded_models)} models")
            time.sleep(120)

    log("=" * 60)
    log("ALL JARVIS INSTANCES HARVESTED AND DESTROYED")
    log(f"Total models local: {len(downloaded_models)}")
    log("=" * 60)

    # Phase 2: Upload all models to UpCloud and run benchmark
    log("PHASE 2: Ensemble benchmark on UpCloud")
    ssh_uc = f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -i {SSH_KEY} root@{UPCLOUD_IP}"
    scp_uc = f"scp -o StrictHostKeyChecking=no -i {SSH_KEY}"

    # Upload benchmark script
    log("Uploading ensemble benchmark script...")
    bench_script = os.path.join(BENCH_DIR, "aggrefact_ensemble.py")
    run(f'{scp_uc} "{bench_script}" root@{UPCLOUD_IP}:/home/director-ai/', timeout=30)

    # Tar all local models and upload to UpCloud
    log("Tarring all local models for upload...")
    all_models_tar = os.path.join(MODELS_DIR, "all_models.tar.gz")
    model_dirs = [
        d
        for d in os.listdir(MODELS_DIR)
        if d.startswith("factcg-") and os.path.isdir(os.path.join(MODELS_DIR, d))
    ]
    tar_args = " ".join(model_dirs)
    models_u = MODELS_DIR.replace("\\", "/")
    run(
        f'cd "{models_u}" && tar --force-local -czf all_models.tar.gz {tar_args}',
        timeout=600,
    )

    log(f"Uploading {len(model_dirs)} models to UpCloud (this takes a while)...")
    code, _, err = run(
        f'{scp_uc} "{all_models_tar}" root@{UPCLOUD_IP}:/home/director-ai/',
        timeout=3600,
    )
    if code != 0:
        log(f"Upload failed: {err}")
        log("Trying individual model uploads instead...")
        for mdir in model_dirs:
            # Tar individual model
            ind_tar = os.path.join(MODELS_DIR, f"{mdir}.tar.gz")
            run(f'cd "{MODELS_DIR}" && tar czf {mdir}.tar.gz {mdir}', timeout=120)
            run(
                f'{scp_uc} "{ind_tar}" root@{UPCLOUD_IP}:/home/director-ai/models/',
                timeout=600,
            )
            run(
                f'{ssh_uc} "cd /home/director-ai/models && tar xzf {mdir}.tar.gz && rm {mdir}.tar.gz"',
                timeout=120,
            )
            os.remove(ind_tar)
            log(f"  Uploaded {mdir}")
    else:
        log("Extracting models on UpCloud...")
        run(
            f'{ssh_uc} "cd /home/director-ai && tar xzf all_models.tar.gz -C /home/director-ai/models/ && rm all_models.tar.gz"',
            timeout=600,
        )

    os.remove(all_models_tar) if os.path.exists(all_models_tar) else None

    # Also upload base model reference
    log("Running ensemble benchmark on UpCloud...")
    run(
        f'{ssh_uc} "cd /home/director-ai && python aggrefact_ensemble.py '
        f"--models-dir /home/director-ai/models "
        f"--base-model yaxili96/FactCG-DeBERTa-v3-Large "
        f'2>&1 | tee benchmark_results.log"',
        timeout=7200,
    )

    # Download results
    log("Downloading benchmark results...")
    results_dir = os.path.join(BENCH_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    run(
        f'{scp_uc} root@{UPCLOUD_IP}:/home/director-ai/benchmark_results.log "{results_dir}/"',
        timeout=60,
    )
    run(
        f'{scp_uc} "root@{UPCLOUD_IP}:/home/director-ai/ensemble_results*.json" "{results_dir}/"',
        timeout=60,
    )

    log("Benchmark complete. Results in benchmarks/results/")

    # Phase 3: Destroy UpCloud
    log("PHASE 3: Destroying UpCloud instance")
    # UpCloud destruction via API
    code, out, _ = run(
        'curl -s -X DELETE -u "protoscience@anulum.li:$(cat ~/.upcloud_password 2>/dev/null || echo NOPASS)" '
        '"https://api.upcloud.com/1.3/server/00ef27ca-xxxx/stop_and_destroy"',  # placeholder
        timeout=30,
    )
    log(f"UpCloud destroy result: {out[:200]}")
    log("NOTE: If UpCloud auto-destroy failed, manually destroy via console")

    log("=" * 60)
    log("AUTONOMOUS HARVEST COMPLETE")
    log(f"Models: {len(downloaded_models)}")
    log("Benchmark results: benchmarks/results/")
    log("=" * 60)


if __name__ == "__main__":
    main()
