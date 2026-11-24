#!/usr/bin/env bash
# Puts the pinned weights (weights.lock: <release tag> <sha256>) at timmytime/weights/timmytime.npz.
# Needs the GitHub CLI logged in (the repo is internal to vectorize-io); in CI it uses GITHUB_TOKEN.
set -euo pipefail
cd "$(dirname "$0")/.."
read -r TAG SHA < weights.lock
OUT=timmytime/weights/timmytime.npz
if [ -f "$OUT" ] && [ "$(shasum -a 256 "$OUT" | cut -d' ' -f1)" = "$SHA" ]; then echo "weights $TAG already here"; exit 0; fi
mkdir -p timmytime/weights
gh release download "$TAG" --repo vectorize-io/timmytime --pattern "*.npz" --output "$OUT.tmp" --clobber
GOT=$(shasum -a 256 "$OUT.tmp" | cut -d' ' -f1)
[ "$GOT" = "$SHA" ] || { echo "checksum mismatch: got $GOT, want $SHA" >&2; rm -f "$OUT.tmp"; exit 1; }
mv "$OUT.tmp" "$OUT"
echo "weights $TAG -> $OUT"
