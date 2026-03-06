#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Tag a release and create a local git bundle backup.
# Usage: bash tools/tag-and-backup.sh v2.9.0
set -euo pipefail

TAG="${1:?Usage: $0 <tag>  (e.g. v2.9.0)}"
BACKUP_DIR="$(git rev-parse --show-toplevel)/../../.coordination/backups"
REPO_NAME="$(basename "$(git rev-parse --show-toplevel)")"
DATE="$(date +%Y%m%d)"
BUNDLE="${BACKUP_DIR}/${REPO_NAME}-${TAG}-stable-${DATE}.bundle"

mkdir -p "$BACKUP_DIR"

echo "=== Preflight ==="
python tools/preflight.py --no-tests || { echo "FAIL: preflight failed, aborting tag"; exit 1; }

echo ""
echo "=== Tagging ${TAG} ==="
git tag -a "$TAG" -m "Release ${TAG}"

echo ""
echo "=== Creating backup bundle ==="
git bundle create "$BUNDLE" --all
SIZE=$(du -h "$BUNDLE" | cut -f1)
echo "Backup: $BUNDLE ($SIZE)"

echo ""
echo "=== Next steps ==="
echo "  git push origin main --tags"
echo "  Verify CI passes, then release workflow will create GitHub Release."
