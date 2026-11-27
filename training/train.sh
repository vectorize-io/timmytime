#!/usr/bin/env bash
# Trains a model on hs-date-1-data, the way hs-date-1 was made, then prunes it and scores it.
#
#   bash training/train.sh                                   # from scratch: 20 passes, ~6 h on an M-series Mac
#   BASE=text2date/weights/new.pt bash training/train.sh finetune   # 3 short passes on top of a model, ~1 h
#
# Extra training files (new gap data, see TRAINING.md) go in EXTRA="data/train_x.jsonl ...", each
# repeated twice. Output: $OUT.npz / .pt, the pruned ${OUT}p.npz, scores in data/score_<name>_<set>.txt.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-text2date/weights/new}
D=data/hs-date-1-data/train
[ -d $D ] || uvx --from huggingface_hub hf download hindsight-ai/hs-date-1-data --repo-type dataset --local-dir data/hs-date-1-data
rep() { for _ in $(seq "$1"); do printf "%s " "${@:2}"; done; }
TRAIN="$D/base.jsonl $(rep 2 $D/new.jsonl $D/templates.jsonl ${EXTRA:-}) $(rep 3 $D/days_counts.jsonl $D/hard.jsonl)"
G="uv run python -m training.guard --max-gb 6 --gpu-ratio 0.3 --kill-below 12 --"
ARGS="--train $TRAIN --out $OUT.npz --batch 128 --stride 4 --n-dec 1 --n-enc 6 --seed 0 --data-seed ${DATA_SEED:-1}"
case ${1:-scratch} in
  scratch)  $G uv run python -m training.model.model $ARGS --epochs 20 ;;
  finetune) $G uv run python -m training.model.model $ARGS --epochs 3 --lr 3e-4 --resume "${BASE:?set BASE=<model>.pt}" ;;
  *) sed -n 2,8p "$0"; exit 1 ;;
esac
uv run python -m training.model.prune $OUT.npz ${OUT}p.npz $TRAIN
N=$(basename $OUT)
for L in hindsight en es de zh ar ja ko hi id vi th it fr ru pt zh-hant; do
  TEXT2DATE_WEIGHTS=${OUT}p.npz uv run python -m training.eval.score text2date --lang $L -v > data/score_${N}_$L.txt 2>&1
  grep -m1 "exact=" data/score_${N}_$L.txt | sed "s/^== /$L: /" | cut -c1-110
done
