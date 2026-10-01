# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Vertex AI entrypoint for the contradiction LoRA fine-tune

"""Self-contained Vertex AI custom-training entrypoint.

Downloads the contradiction dataset from GCS, runs the LoRA fine-tune
(``training.train_contradiction``), merges the adapter into a standalone model,
and uploads both the adapter and the merged model back to GCS. Held-out
evaluation runs separately (it needs the full director_ai package); this job
only produces the trained artefacts so the training container stays minimal.

Environment:
    DATA_BUCKET   GCS bucket holding ``data_contradiction/`` (default below)
    OUT_BUCKET    GCS bucket for the artefacts (default below)
    OUT_PREFIX    Prefix under OUT_BUCKET (default ``contradiction-lora-vertex``)
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("vertex.contradiction")

BASE_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
DATA_BUCKET = os.environ.get("DATA_BUCKET", "gotm-director-ai-data")
OUT_BUCKET = os.environ.get("OUT_BUCKET", "gotm-director-ai-training")
OUT_PREFIX = os.environ.get("OUT_PREFIX", "contradiction-lora-vertex")

REPO = Path(__file__).resolve().parent.parent
DATA_DST = REPO / "training" / "data_contradiction"
ADAPTER = REPO / "training" / "output" / "contradiction-lora-vertex"
MERGED = REPO / "training" / "output" / "contradiction-lora-vertex-merged"


def _download_data() -> None:
    from google.cloud import storage

    client = storage.Client()
    blobs = list(client.list_blobs(DATA_BUCKET, prefix="data_contradiction/"))
    if not blobs:
        raise RuntimeError(f"no data under gs://{DATA_BUCKET}/data_contradiction/")
    for blob in blobs:
        rel = blob.name[len("data_contradiction/") :]
        if not rel:
            continue
        dst = DATA_DST / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dst))
    logger.info("Downloaded %d data files to %s", len(blobs), DATA_DST)


def _upload_dir(local: Path, prefix: str) -> None:
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(OUT_BUCKET)
    n = 0
    for path in local.rglob("*"):
        if path.is_file():
            bucket.blob(f"{prefix}/{path.relative_to(local)}").upload_from_filename(
                str(path)
            )
            n += 1
    logger.info("Uploaded %d files: %s -> gs://%s/%s", n, local, OUT_BUCKET, prefix)


def _train() -> None:
    # fp32 is kept (DeBERTa-v3 is unstable in reduced precision; not worth a NaN
    # gamble on a tight credit budget). Cost is bounded instead: 2 epochs cap the
    # step count regardless of early stopping, and eval every 400 steps halves the
    # expensive (~minutes each) evaluation passes. All overridable via env so the
    # config can change without rebuilding the container.
    # All hyperparams env-overridable so the same image fits both L4 (24 GB, large
    # batch, no checkpoint) and T4 (16 GB, small batch + gradient checkpointing).
    epochs = os.environ.get("EPOCHS", "2")
    eval_steps = os.environ.get("EVAL_STEPS", "400")
    max_train = os.environ.get("MAX_TRAIN", "0")
    batch_size = os.environ.get("BATCH_SIZE", "8")
    grad_accum = os.environ.get("GRAD_ACCUM", "4")
    max_length = os.environ.get("MAX_LENGTH", "512")
    cmd = [
        sys.executable,
        "-m",
        "training.train_contradiction",
        "--rank",
        "32",
        "--lora-alpha",
        "64",
        "--max-length",
        max_length,
        "--epochs",
        epochs,
        "--batch-size",
        batch_size,
        "--grad-accum",
        grad_accum,
        "--lr",
        "7e-4",
        "--max-train",
        max_train,
        "--eval-steps",
        eval_steps,
        "--logging-steps",
        "25",
        "--output-dir",
        str(ADAPTER),
    ]
    # Gradient checkpointing trades compute for memory — needed on the 16 GB T4.
    if os.environ.get("GRAD_CHECKPOINT", "0").strip().lower() in ("1", "true", "yes"):
        cmd.append("--grad-checkpoint")
    logger.info("Training: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO), check=True)


def _merge() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    base = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=3)
    merged = PeftModel.from_pretrained(base, str(ADAPTER)).merge_and_unload()
    MERGED.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(MERGED))
    AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False).save_pretrained(
        str(MERGED)
    )
    del base, merged
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Merged adapter -> %s", MERGED)


def main() -> None:
    """Download data, train and merge the LoRA adapter, then upload artefacts."""
    _download_data()
    _train()
    _upload_dir(ADAPTER, f"{OUT_PREFIX}-adapter")
    _merge()
    _upload_dir(MERGED, f"{OUT_PREFIX}-merged")
    logger.info("Done. Artefacts under gs://%s/%s-*", OUT_BUCKET, OUT_PREFIX)


if __name__ == "__main__":
    main()
