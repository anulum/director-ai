#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — verified rustup and Rust toolchain installation

# The rust-docs quality identity hashes the cargo and rustc executables the
# ratchet runs. Through rustup those are its proxies, which are the rustup binary
# itself, so a capture reproduces the recorded baseline only under the rustup
# release it was recorded with. This installs exactly that release and the
# pinned toolchain into CARGO_HOME, then refuses unless every proxy is the
# verified binary. RUNNER_TEMP holds the download; GITHUB_PATH gets the bin.
set -euo pipefail

: "${RUNNER_TEMP:?RUNNER_TEMP must name the job temporary directory}"
: "${GITHUB_PATH:?GITHUB_PATH must name the job path file}"
for command in curl sha256sum mktemp wc uname; do
    command -v "$command" >/dev/null
done
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    printf '%s\n' 'rustup installer requires Linux x86_64.' >&2
    exit 2
fi

rustup_version=1.29.3
rustup_sha256=b1c8cb14e30889e7e2999e6b69e62d9e9b9ffc0f4df8180564d306d9680e6c55
rustup_bytes=21347088
toolchain=1.98.1
cargo_home=${CARGO_HOME:-${HOME}/.cargo}
download=$(mktemp -d "${RUNNER_TEMP}/director-rustup.XXXXXXXX")
init="${download}/rustup-init"
curl --proto '=https' --tlsv1.2 --location --fail --silent --show-error \
    --retry 3 --retry-connrefused --connect-timeout 20 --max-time 300 \
    --max-filesize "$rustup_bytes" \
    "https://static.rust-lang.org/rustup/archive/${rustup_version}/x86_64-unknown-linux-gnu/rustup-init" \
    --output "$init"
if [[ "$(wc -c <"$init")" -ne "$rustup_bytes" ]]; then
    printf '%s\n' 'rustup-init size differs from the pinned release.' >&2
    exit 1
fi
printf '%s  %s\n' "$rustup_sha256" "$init" | sha256sum --check --strict -
chmod u+x "$init"
CARGO_HOME="$cargo_home" "$init" -y --no-modify-path --profile minimal \
    --default-toolchain none
rm -- "$init"
CARGO_HOME="$cargo_home" "${cargo_home}/bin/rustup" toolchain install "$toolchain" --profile minimal
CARGO_HOME="$cargo_home" "${cargo_home}/bin/rustup" default "$toolchain"
for proxy in rustup cargo rustc; do
    printf '%s  %s\n' "$rustup_sha256" "${cargo_home}/bin/${proxy}" | sha256sum --check --strict -
done
rustup_actual=$("${cargo_home}/bin/rustup" --version 2>/dev/null)
if [[ "$rustup_actual" != "rustup ${rustup_version} "* ]]; then
    printf '%s\n' 'rustup version differs from the pinned release.' >&2
    exit 1
fi
rustc_actual=$("${cargo_home}/bin/rustc" --version)
if [[ "$rustc_actual" != "rustc ${toolchain} "* ]]; then
    printf '%s\n' 'rustc version differs from the pinned toolchain.' >&2
    exit 1
fi
printf '%s\n%s\n' "$rustup_actual" "$rustc_actual"
printf '%s\n' "${cargo_home}/bin" >>"$GITHUB_PATH"
