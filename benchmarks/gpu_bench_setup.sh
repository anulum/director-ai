#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Director-AI — JarvisLabs GPU Benchmark Bootstrap
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ---------------------------------------------------------------------
# Run on a fresh JarvisLabs instance (Ubuntu + NVIDIA driver preinstalled).
# Clones repo, installs deps, exports ONNX model, runs cross-GPU benchmark.
set -euo pipefail

git clone https://github.com/anulum/director-ai.git && cd director-ai
pip install --no-deps -e .
pip install onnxruntime-gpu==1.20.1 "optimum[onnxruntime-gpu]==1.24.0"
python -c "from director_ai.core.nli import export_onnx; export_onnx(output_dir='benchmarks/results/factcg_onnx')"
python -m benchmarks.gpu_bench --onnx-path benchmarks/results/factcg_onnx --iterations 50
