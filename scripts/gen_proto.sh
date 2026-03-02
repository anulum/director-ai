#!/usr/bin/env bash
# Generate Python gRPC stubs from proto/director.proto
set -euo pipefail

PROTO_DIR="$(cd "$(dirname "$0")/.." && pwd)/proto"
OUT_DIR="$(cd "$(dirname "$0")/.." && pwd)/src/director_ai"

python -m grpc_tools.protoc \
    -I"${PROTO_DIR}" \
    --python_out="${OUT_DIR}" \
    --grpc_python_out="${OUT_DIR}" \
    "${PROTO_DIR}/director.proto"

echo "Generated stubs in ${OUT_DIR}"
