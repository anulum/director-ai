#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — verified Lean toolchain installation

# RUNNER_TEMP owns the installation; GITHUB_PATH receives its bin directory.
set -euo pipefail

: "${RUNNER_TEMP:?RUNNER_TEMP must name the job temporary directory}"
: "${GITHUB_PATH:?GITHUB_PATH must name the job path file}"
for command in curl sha256sum tar zstd mktemp wc uname; do
    command -v "$command" >/dev/null
done
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    printf '%s\n' 'Lean installer requires Linux x86_64.' >&2
    exit 2
fi

lean_version=4.34.1
lean_sha256=47bf4bbd78f70c2e9670598ab7124d92b6efb7330ff33e5fbb4030f6fd72e4e4
lean_bytes=580432872
lean_root=$(mktemp -d "${RUNNER_TEMP}/director-lean.XXXXXXXX")
lean_archive="${lean_root}/lean.tar.zst"
curl --proto '=https' --tlsv1.2 --location --fail --silent --show-error \
    --retry 3 --retry-connrefused --connect-timeout 20 --max-time 600 \
    --max-filesize "$lean_bytes" \
    "https://github.com/leanprover/lean4/releases/download/v${lean_version}/lean-${lean_version}-linux.tar.zst" \
    --output "$lean_archive"
if [[ "$(wc -c <"$lean_archive")" -ne "$lean_bytes" ]]; then
    printf '%s\n' 'Lean archive size differs from the pinned release asset.' >&2
    exit 1
fi
printf '%s  %s\n' "$lean_sha256" "$lean_archive" | sha256sum --check --strict -
tar --zstd --extract --file "$lean_archive" --directory "$lean_root" --strip-components=1
lean_actual=$("${lean_root}/bin/lean" --version)
if [[ "$lean_actual" != "Lean (version ${lean_version},"* ]]; then
    printf '%s\n' 'Lean executable version differs from the pinned toolchain.' >&2
    exit 1
fi
"${lean_root}/bin/lake" --version
printf '%s\n' "$lean_actual"
printf '%s\n' "${lean_root}/bin" >>"$GITHUB_PATH"
rm -- "$lean_archive"
