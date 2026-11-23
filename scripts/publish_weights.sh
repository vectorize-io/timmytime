#!/usr/bin/env bash
# Publishes new weights and pins them:  scripts/publish_weights.sh <name> <file.npz>
# Uploads the file as tlm.npz to a GitHub release "weights-<name>", copies it to
# tlm/weights/tlm.npz and rewrites weights.lock. Commit weights.lock; the next package release ships them.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=$1; FILE=$2
SHA=$(shasum -a 256 "$FILE" | cut -d' ' -f1)
TMP=$(mktemp -d)
cp "$FILE" "$TMP/tlm.npz"
gh release create "weights-$NAME" "$TMP/tlm.npz" --repo vectorize-io/tlm \
  --title "weights $NAME" --notes "Model weights $NAME, pinned by weights.lock. Not a package release."
rm -r "$TMP"
cp "$FILE" tlm/weights/tlm.npz
echo "weights-$NAME $SHA" > weights.lock
echo "pinned weights-$NAME ($SHA)"
