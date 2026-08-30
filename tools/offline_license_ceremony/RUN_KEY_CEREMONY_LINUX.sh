#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Linux Offline Licence Key Ceremony Launcher
set -euo pipefail

ceremony_root="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
cd "$ceremony_root"

printf '%s\n' 'Director-AI SEC-1 offline key ceremony'
printf '%s\n\n' 'This computer must be offline before you continue.'

rm -rf -- .ceremony-venv
python3 verify_bundle.py
python3 -m venv .ceremony-venv
.ceremony-venv/bin/python -m pip install \
  --disable-pip-version-check \
  --no-index \
  --require-hashes \
  --find-links wheelhouse \
  -r requirements-offline.txt

.ceremony-venv/bin/python run_ceremony.py

printf '\n%s\n' 'SUCCESS. Shut down the laptop before removing the PRIVATE vault medium.'
printf '%s\n' 'Only PUBLIC_KEY_ONLY.txt may return to the online workstation.'
