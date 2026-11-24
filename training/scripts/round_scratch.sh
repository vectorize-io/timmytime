# One clean model from scratch on the final data mix (the g25 mix + junk/no-date templates), instead of 25 fine-tunes stacked on
# each other: no habits left over from early data, and a check that the rounds did not overfit our tests.
# 6 encoder layers, 1 decoder, 20 epochs (~8-9 h on an M-series Mac). If the run dies, it resumes from the
# last saved epoch.   bash training/scripts/round_scratch.sh  (after days_data.sh and round_g25.sh)
set -e
cd "$(dirname "$0")/../.."
ALL_L="en es de zh ar ja ko hi id vi th it fr ru pt zh-hant"
OUT=timmytime/weights/timmytime-s1.npz
LOG=data/round_scratch.log
EPOCHS=20

if [ ! -f $LOG ]; then  # first start: templates with the latest fixes (junk input, bare "wed", a billion years ago)
  uv run python -m training.data.zh_templates --n 40000 --seed 7 --out data/train_zh_tpl.jsonl
  uv run python -m training.data.en_templates --n 20000 --seed 7 --out data/train_en_tpl.jsonl
  uv run python -m training.data.clean data/train_zh_tpl.jsonl data/train_en_tpl.jsonl
fi
TPL="data/train_zh_tpl.jsonl data/train_en_tpl.jsonl"
OLD=$(ls data/train_*.jsonl | grep -vE "early|train_hard|_new|_num|_mix2|_hol2|_tpl|_days" | tr '\n' ' ')
HARD=$(for f in data/train_hard*.jsonl; do echo "$f $f $f"; done | tr '\n' ' ')
NEW=$(for f in data/train_*_new.jsonl data/train_*_num.jsonl data/train_*_mix2.jsonl data/train_*_hol2.jsonl; do echo "$f $f"; done | tr '\n' ' ')
G=$(for f in $TPL; do echo "$f $f"; done | tr '\n' ' ')
DAYS=$(for f in data/train_*_days.jsonl; do echo "$f $f $f"; done | tr '\n' ' ')

for try in 1 2 3 4 5; do
  done_ep=$(grep -oE "^epoch +[0-9]+" $LOG 2>/dev/null | tail -1 | grep -oE "[0-9]+" || echo 0)
  [ "$done_ep" -ge $EPOCHS ] && break
  RES=""
  [ "$done_ep" -gt 0 ] && RES="--resume ${OUT%.npz}.pt --start-epoch $done_ep"
  echo "== try $try from epoch $done_ep" >> $LOG
  uv run python -m training.guard --max-gb 6 --gpu-ratio 0.2 --kill-below 15 -- uv run python -m training.model.model \
    --train $OLD $HARD $NEW $G $DAYS --out $OUT --epochs $EPOCHS --batch 128 --stride 4 --n-dec 1 --n-enc 6 $RES >> $LOG 2>&1 \
    && break || sleep 120
done
uv run python -m training.guard --threads 2 --max-gb 2 -- uv run python -m training.model.prune $OUT ${OUT%.npz}p.npz data/train_*.jsonl
for L in hindsight $ALL_L; do TIMMYTIME_WEIGHTS=${OUT%.npz}p.npz uv run python -m training.guard --threads 1 --max-gb 2 -- uv run python -m training.eval.score timmytime --lang $L -v > data/score_s1_$L.txt 2>&1; done
echo SCRATCH-DONE
