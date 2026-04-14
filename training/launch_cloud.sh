#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Director-AI — Launch distillation v5 on cloud GPU
#
# Usage:
#   ./training/launch_cloud.sh [strategy]
#
# strategy: A, B, C, or all (default: all)
#
# Supports: JarvisLabs, Vertex AI, or any Docker-capable GPU host.

set -euo pipefail

STRATEGY="${1:-all}"
IMAGE="distil-v5"
TAG="$(date +%Y%m%d-%H%M)"

echo "=== Distillation v5 — Strategy: ${STRATEGY} ==="
echo "Building Docker image..."

docker build -f training/Dockerfile.distil -t "${IMAGE}:${TAG}" .

echo "Image built: ${IMAGE}:${TAG}"
echo ""
echo "=== Run options ==="
echo ""
echo "# Local GPU:"
echo "docker run --gpus all -v \$(pwd)/training/output:/app/training/output ${IMAGE}:${TAG} --strategy ${STRATEGY}"
echo ""
echo "# JarvisLabs (upload image, then SSH):"
echo "# jlcli create --name distil-v5 --gpu A100 --framework pytorch"
echo "# scp -r training/ user@<instance>:~/training/"
echo "# ssh user@<instance> 'cd ~/; python training/distil_v5_cloud.py --strategy ${STRATEGY}'"
echo ""
echo "# Vertex AI Custom Training:"
echo "# gcloud ai custom-jobs create \\"
echo "#   --region=europe-west4 \\"
echo "#   --display-name=distil-v5-${STRATEGY} \\"
echo "#   --worker-pool-spec=machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri=gcr.io/\${PROJECT}/${IMAGE}:${TAG}"
echo ""
echo "=== Expected runtime ==="
echo "Strategy A (base, 10 epochs):     ~45 min on T4, ~15 min on A100"
echo "Strategy B (6 sweep configs):     ~3 hours on T4, ~1 hour on A100"
echo "Strategy C (re-label + 2 trains): ~2 hours on T4, ~40 min on A100"
echo "Strategy all:                     ~6 hours on T4, ~2 hours on A100"
