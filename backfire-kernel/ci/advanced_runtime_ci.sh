#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Backfire kernel CI entrypoint

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

cargo fmt --all -- --check
cargo check -p backfire-ffi
cargo test --workspace

if [[ "${BACKFIRE_KERNEL_BUILD_WHEEL:-0}" == "1" ]]; then
    command -v maturin >/dev/null
    maturin build --release -m crates/backfire-ffi/Cargo.toml --out target/wheel-contract
fi
