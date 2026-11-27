#!/usr/bin/env bash
# Puts the pinned model (weights.lock: <Hugging Face repo> <commit> <sha256>) at timmytime/weights/model.safetensors.
set -euo pipefail
cd "$(dirname "$0")/.."
read -r REPO REV SHA < weights.lock
OUT=timmytime/weights/model.safetensors
if [ -f "$OUT" ] && [ "$(shasum -a 256 "$OUT" | cut -d' ' -f1)" = "$SHA" ]; then echo "$REPO already here"; exit 0; fi
mkdir -p timmytime/weights
curl -fsSL "https://huggingface.co/$REPO/resolve/$REV/model.safetensors" -o "$OUT.tmp"
GOT=$(shasum -a 256 "$OUT.tmp" | cut -d' ' -f1)
[ "$GOT" = "$SHA" ] || { echo "checksum mismatch: got $GOT, want $SHA" >&2; rm -f "$OUT.tmp"; exit 1; }
mv "$OUT.tmp" "$OUT"
echo "$REPO@${REV:0:7} -> $OUT"
