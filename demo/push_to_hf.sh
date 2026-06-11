#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Hugging Face Space publish helper
# Push demo files to HuggingFace Space.
# Prerequisites: run `hf auth login` first.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SPACE_REPO="https://huggingface.co/spaces/anulum/director-ai-guardrail"
TMP_DIR="$(mktemp -d)"
VERSION=$(grep -m1 '^version' "$SCRIPT_DIR/../pyproject.toml" | sed 's/.*"\(.*\)".*/\1/')

echo "Cloning Space repo to $TMP_DIR ..."
git clone "$SPACE_REPO" "$TMP_DIR"

echo "Copying updated files ..."
cp "$SCRIPT_DIR/app.py" "$TMP_DIR/app.py"
cp "$SCRIPT_DIR/requirements.txt" "$TMP_DIR/requirements.txt"
cp "$SCRIPT_DIR/README_HF.md" "$TMP_DIR/README.md"

cd "$TMP_DIR"
git add app.py requirements.txt README.md
git diff --cached --stat

echo ""
read -p "Push to HF Space? [y/N] " confirm
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    git commit -m "Update demo to v${VERSION}"
    git push
    echo "Done. Space will rebuild at: $SPACE_REPO"
else
    echo "Aborted. Files staged in: $TMP_DIR"
fi
