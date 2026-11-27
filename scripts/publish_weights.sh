#!/usr/bin/env bash
# Publishes trained weights to Hugging Face and pins them:  scripts/publish_weights.sh <repo> <weights.npz>
#   scripts/publish_weights.sh hindsight-ai/hs-date-1 text2date/weights/newp.npz
# Writes model.safetensors, config.json and vocab.json next to the repo's model card, uploads them
# (needs `hf auth login`), and rewrites weights.lock. Commit weights.lock; the next release ships them.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$1; FILE=$2
DIR=$(mktemp -d)
hf download "$REPO" README.md assets/banner.png --local-dir "$DIR" >/dev/null 2>&1 || true
uv run python -m training.model.export "$FILE" "$DIR"
REV=$(hf upload "$REPO" "$DIR" . --commit-message "weights from $(basename "$FILE")" | grep -oE 'commit/[0-9a-f]{40}' | cut -d/ -f2)
SHA=$(shasum -a 256 "$DIR/model.safetensors" | cut -d' ' -f1)
mkdir -p text2date/weights && cp "$DIR/model.safetensors" text2date/weights/model.safetensors
rm -r "$DIR"
echo "$REPO $REV $SHA" > weights.lock
echo "pinned $REPO@${REV:0:7}"
