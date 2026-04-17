#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — regenerate language stubs from the proto schema
#
# Idempotent. Run whenever `schemas/proto/**/*.proto` changes.
#
# Environment:
#   PROTOC                 path to protoc             (default: protoc on PATH)
#   PROTOC_GEN_GO          path to protoc-gen-go      (default: go/bin)
#   PROTOC_GEN_GO_GRPC     path to protoc-gen-go-grpc (default: go/bin)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROTO_SRC="${SCRIPT_DIR}/proto"
PY_OUT="${REPO_ROOT}/src/director_ai/proto"
GO_OUT="${REPO_ROOT}/gateway/go/proto"

mkdir -p "${PY_OUT}" "${GO_OUT}"

PROTO_FILES=$(find "${PROTO_SRC}" -name '*.proto' | sort)
if [[ -z "${PROTO_FILES}" ]]; then
  echo "error: no .proto files under ${PROTO_SRC}" >&2
  exit 1
fi

PROTOC="${PROTOC:-$(command -v protoc)}"
if [[ -z "${PROTOC}" ]]; then
  echo "error: protoc not found (set PROTOC)" >&2
  exit 1
fi

echo "→ Python stubs → ${PY_OUT}"
python -m grpc_tools.protoc \
  --proto_path="${PROTO_SRC}" \
  --python_out="${PY_OUT}" \
  --pyi_out="${PY_OUT}" \
  --grpc_python_out="${PY_OUT}" \
  ${PROTO_FILES}

# grpc_tools writes absolute imports like ``from director.v1 import
# director_pb2``, which breaks when the package lives under
# ``director_ai.proto``. Rewrite the imports to the package-relative
# form.
find "${PY_OUT}" -type f -name '*_pb2*.py' -exec \
  sed -i 's|^from director\.v1 |from director_ai.proto.director.v1 |' {} \;

# Ensure every emitted directory has an __init__.py so Python treats
# them as packages.
find "${PY_OUT}" -type d -exec bash -c 'test -f "$1/__init__.py" || : > "$1/__init__.py"' _ {} \;

echo "→ Go stubs → ${GO_OUT}"
PATH_WITH_PLUGINS="${HOME}/go/bin:/usr/local/go/bin:${PATH}"
PATH="${PATH_WITH_PLUGINS}" "${PROTOC}" \
  --proto_path="${PROTO_SRC}" \
  --go_out="${GO_OUT}" \
  --go_opt=paths=source_relative \
  --go-grpc_out="${GO_OUT}" \
  --go-grpc_opt=paths=source_relative \
  ${PROTO_FILES}

echo "✓ stubs regenerated"
