# The release model: 3 clean runs from the same starting weights (--seed 0), each with its own data order
# (--data-seed 1/2/3), two side by side then the third, then their average (training/model/soup.py). Everything is scored:
# each run and the average, on hindsight, the 16 big sets, days-, counts- and ALL CAPS.
# All fixes are in: renumber's number ranges, ALL CAPS, "a month ago" (issues #1-#3), days 29-31, counts.
#   bash training/scripts/days_data.sh; bash training/scripts/counts_data.sh; EPOCHS=40 bash training/scripts/round_final.sh
set -e
cd "$(dirname "$0")/../.."
ALL_L="en es de zh ar ja ko hi id vi th it fr ru pt zh-hant"
EPOCHS=${EPOCHS:-40}
W=timmytime/weights

if [ ! -f data/train_en_tpl_final.done ]; then  # templates with every fix
  uv run python -m training.data.zh_templates --n 40000 --seed 8 --out data/train_zh_tpl.jsonl
  uv run python -m training.data.en_templates --n 20000 --seed 8 --out data/train_en_tpl.jsonl
  uv run python -m training.data.clean data/train_zh_tpl.jsonl data/train_en_tpl.jsonl
  touch data/train_en_tpl_final.done
fi
TPL="data/train_zh_tpl.jsonl data/train_en_tpl.jsonl"
OLD=$(ls data/train_*.jsonl | grep -vE "early|train_hard|_new|_num|_mix2|_hol2|_tpl|_days|_counts" | tr '\n' ' ')
x() { for f in "$@"; do printf "%s " $(yes "$f" | head -$N); done; }
HARD=$(N=3 x data/train_hard*.jsonl)
NEW=$(N=2 x data/train_*_new.jsonl data/train_*_num.jsonl data/train_*_mix2.jsonl data/train_*_hol2.jsonl)
G=$(N=2 x $TPL)
EXTRA=$(N=3 x data/train_*_days.jsonl data/train_*_counts.jsonl)

run() {  # $1 = data seed; resumes from its last saved epoch if killed
  local out=$W/timmytime-f$1.npz log=data/round_final_f$1.log
  for try in 1 2 3 4 5; do
    local ep=$(grep -oE "^epoch +[0-9]+" $log 2>/dev/null | tail -1 | grep -oE "[0-9]+" || echo 0)
    [ "$ep" -ge $EPOCHS ] && return 0
    local res=""; [ "$ep" -gt 0 ] && res="--resume ${out%.npz}.pt --start-epoch $ep"
    echo "== try $try from epoch $ep" >> $log
    uv run python -m training.guard --max-gb 6 --gpu-ratio 0.3 --kill-below 12 -- uv run python -m training.model.model \
      --train $OLD $HARD $NEW $G $EXTRA --out $out --epochs $EPOCHS --batch 128 --stride 4 --n-dec 1 --n-enc 6 \
      --seed 0 --data-seed $1 $res >> $log 2>&1 && return 0 || sleep 120
  done
}
run 1 & sleep 300; run 2 & wait; run 3  # two at once fit in memory next to normal use; three did not

uv run python -m training.model.soup $W/timmytime-f1.pt $W/timmytime-f2.pt $W/timmytime-f3.pt --out $W/timmytime-fsoup.npz
for n in f1 f2 f3 fsoup; do
  uv run python -m training.guard --threads 2 --max-gb 2 -- uv run python -m training.model.prune $W/timmytime-$n.npz $W/timmytime-${n}p.npz data/train_*.jsonl
  for L in hindsight $ALL_L; do TIMMYTIME_WEIGHTS=$W/timmytime-${n}p.npz uv run python -m training.eval.score timmytime --lang $L -v > data/score_${n}_$L.txt 2>&1; done
  TIMMYTIME_WEIGHTS=$W/timmytime-${n}p.npz uv run python -m training.eval.caps en es de > data/caps_$n.txt 2>&1
done
echo FINAL-DONE
