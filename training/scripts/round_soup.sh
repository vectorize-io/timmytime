# The release model, take 2: averaging 3 clean runs from scratch failed (387/451 hindsight tests broke): from the
# same starting weights they still end up far apart. Averaging works between short fine-tunes of ONE trained
# model, so: fine-tune the best clean run ($BASE, default f2) 3 times, 3 passes each, own data order, then average.
#   bash training/scripts/round_soup.sh   (after round_final.sh)
set -e
cd "$(dirname "$0")/../.."
ALL_L="en es de zh ar ja ko hi id vi th it fr ru pt zh-hant"
BASE=${BASE:-f2}
W=timmytime/weights
TPL="data/train_zh_tpl.jsonl data/train_en_tpl.jsonl"
OLD=$(ls data/train_*.jsonl | grep -vE "early|train_hard|_new|_num|_mix2|_hol2|_tpl|_days|_counts" | tr '\n' ' ')
x() { for f in "$@"; do printf "%s " $(yes "$f" | head -$N); done; }
HARD=$(N=3 x data/train_hard*.jsonl)
NEW=$(N=2 x data/train_*_new.jsonl data/train_*_num.jsonl data/train_*_mix2.jsonl data/train_*_hol2.jsonl)
G=$(N=2 x $TPL)
EXTRA=$(N=3 x data/train_*_days.jsonl data/train_*_counts.jsonl)

for s in 1 2 3; do
  [ -f $W/timmytime-b$s.pt ] && grep -qE "^epoch +3 " data/round_soup_b$s.log 2>/dev/null && continue
  uv run python -m training.guard --max-gb 6 --gpu-ratio 0.3 --kill-below 12 -- uv run python -m training.model.model \
    --train $OLD $HARD $NEW $G $EXTRA --out $W/timmytime-b$s.npz --epochs 3 --lr 3e-4 --batch 128 --stride 4 --n-dec 1 --n-enc 6 \
    --seed 0 --data-seed 1$s --resume $W/timmytime-$BASE.pt > data/round_soup_b$s.log 2>&1
done
uv run python -m training.model.soup $W/timmytime-b1.pt $W/timmytime-b2.pt $W/timmytime-b3.pt --out $W/timmytime-bsoup.npz
for n in b1 b2 b3 bsoup; do
  uv run python -m training.guard --threads 2 --max-gb 2 -- uv run python -m training.model.prune $W/timmytime-$n.npz $W/timmytime-${n}p.npz data/train_*.jsonl
  for L in hindsight $ALL_L; do TIMMYTIME_WEIGHTS=$W/timmytime-${n}p.npz uv run python -m training.eval.score timmytime --lang $L -v > data/score_${n}_$L.txt 2>&1; done
  TIMMYTIME_WEIGHTS=$W/timmytime-${n}p.npz uv run python -m training.eval.caps en es de > data/caps_$n.txt 2>&1
done
echo SOUP-DONE
