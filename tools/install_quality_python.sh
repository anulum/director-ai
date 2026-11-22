#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — verified quality interpreter installation

# The one pin of the notebook profile's interpreter, read by every job that
# runs the profile: the python-build-standalone release asset, its size and
# its SHA-256, which tools/quality_ratchet_python_engine.py also names. The
# archive is checked before it is extracted; there is no cache, no second
# source and no fallback interpreter. RUNNER_TEMP owns the installation, and
# GITHUB_ENV receives QUALITY_PYTHON and DIRECTOR_QUALITY_PYTHON, the
# interpreter of a virtual environment holding the hash-locked quality
# requirements.
set -euo pipefail

: "${RUNNER_TEMP:?RUNNER_TEMP must name the job temporary directory}"
: "${GITHUB_ENV:?GITHUB_ENV must name the job environment file}"
for command in curl sha256sum tar mktemp wc uname; do
    command -v "$command" >/dev/null
done
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    printf '%s\n' 'Quality interpreter installer requires Linux x86_64.' >&2
    exit 2
fi

python_url=https://github.com/astral-sh/python-build-standalone/releases/download/20261021/cpython-3.11.17%2B20261021-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz
python_sha256=224ca8be2d28841e7a382a27c51e74f4ed0aabe249a79294684e782363165948
python_bytes=30540500
repository=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_root=$(mktemp -d "${RUNNER_TEMP}/director-quality-python.XXXXXXXX")
python_archive="${python_root}/python.tar.gz"
curl --proto '=https' --tlsv1.2 --location --fail --silent --show-error \
    --retry 3 --retry-connrefused --connect-timeout 20 --max-time 600 \
    --max-filesize "$python_bytes" \
    "$python_url" --output "$python_archive"
if [[ "$(wc -c <"$python_archive")" -ne "$python_bytes" ]]; then
    printf '%s\n' 'Python archive size differs from the pinned release asset.' >&2
    exit 1
fi
printf '%s  %s\n' "$python_sha256" "$python_archive" | sha256sum --check --strict -
tar --gzip --extract --file "$python_archive" --directory "$python_root" \
    --no-same-owner --no-same-permissions
rm -- "$python_archive"
base="${python_root}/python"
sha256sum "${base}/bin/python3.11" "${base}/lib/python3.11/tokenize.py" \
    "${base}/lib/python3.11/token.py"
"${base}/bin/python3.11" -I -m venv --without-pip "${python_root}/venv"
quality_python="${python_root}/venv/bin/python"
"${base}/bin/python3.11" -I -m pip --python "$quality_python" install \
    --no-cache-dir --disable-pip-version-check --require-hashes --no-deps \
    -r "${repository}/requirements/ci-dev.txt" \
    -r "${repository}/requirements/ci-types.txt" \
    -r "${repository}/requirements/ci-lint.txt" \
    -r "${repository}/requirements/quality-protobuf.txt"
"$quality_python" -I -c 'import sys; print(sys.version)'
printf 'QUALITY_PYTHON=%s\n' "$quality_python" >>"$GITHUB_ENV"
printf 'DIRECTOR_QUALITY_PYTHON=%s\n' "$quality_python" >>"$GITHUB_ENV"
