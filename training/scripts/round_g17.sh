# Round g17: hindsight's own tests as the goal (data/test_hindsight.jsonl).
# + it fr ru pt zh-hant (test sets ~3k rows each, full training mix); gap data for every language
# (same day last year, the 3rd of last month, day ranges, since, half a month, week after next, far years,
# brand/idiom/config traps, Chinese numerals and full-width digits); fine-tune g16, prune, score all.
# Needs TIMMYTIME_LLM_KEY in the environment.   bash training/scripts/round_g17.sh
set -e
cd "$(dirname "$0")/../.."
GEN="uv run python -m training.guard --max-gb 1 -- uv run python -m training.data.gen --workers 24"
NUM="price|amount|number|version|port|ticket|age|quantity|product|model|room|street|code|duration|config"
NEWL="it fr ru pt zh-hant"
ALL_L="en es de zh ar ja ko hi id vi th $NEWL"

# two languages at a time
pairs() { local f=$1; shift; local a=($@); local i; for ((i = 0; i < ${#a[@]}; i += 2)); do
  $f ${a[i]} & if [ -n "${a[i+1]:-}" ]; then $f ${a[i+1]} & fi; wait; done; }

t_test() { [ -s data/test_$1_big.jsonl ] && [ ! -e data/test_$1_big.progress ] || $GEN --split test --lang $1 --n 4500 --seed 9100 --out data/test_$1_big.jsonl; }
pairs t_test $NEWL
uv run python -m training.data.audit $(for L in $NEWL; do echo data/test_${L}_big.jsonl; done)
echo NEW-TESTS-DONE

t_lang() {
  L=$1
  $GEN --split train --lang $L --n 24000 --seed 22 --out data/train_${L}.jsonl
  $GEN --split train --lang $L --n 9000 --seed 9200 --none-share 0.1 --kinds holiday,future,modified --out data/train_${L}_new.jsonl
  $GEN --split train --lang $L --n 5000 --seed 9300 --none-share 1 --trap-filter "$NUM" --out data/train_${L}_num.jsonl
  $GEN --split train --lang $L --n 3000 --seed 31 --none-share 0 --kinds date,year_abs,month_name,last_n,day,week,range,modified,multi \
    --style "write every number, year and date fully in words, never digits" --out data/train_${L}_words.jsonl
  $GEN --split train --lang $L --n 2500 --seed 33 --none-share 0 --kinds multi --out data/train_${L}_multi.jsonl
}
pairs t_lang $NEWL
echo NEW-LANGS-DONE

t_gap() {
  L=$1
  $GEN --split train --lang $L --n 3000 --seed 9700 --none-share 0.15 \
    --kinds sameday,half,range,open,day,week,month,year,last_n,month_name,date --out data/train_${L}_gap.jsonl
  $GEN --split train --lang $L --n 1500 --seed 9800 --none-share 1 --trap-filter "config|idiom|brand|周|月|明日|后天|每年|time word" \
    --out data/train_${L}_trap2.jsonl
}
pairs t_gap $ALL_L
for L in zh zh-hant; do
  $GEN --split train --lang $L --n 2500 --seed 9850 --none-share 1 --trap-filter "周|月|明日|后天|每年" --out data/train_${L}_trap3.jsonl
done
t_cjk() {
  $GEN --split train --lang $1 --n 2500 --seed 9900 --none-share 0 --kinds date,year_abs,month_name,range,modified,sameday \
    --style "write years and dates with Chinese numerals (二〇二四年三月五日, 廿一日) or full-width digits (２０２４年３月), mix both" \
    --out data/train_$1_cjknum.jsonl
}
pairs t_cjk zh zh-hant ja
uv run python -m training.data.clean $(for L in $NEWL; do echo data/train_${L}.jsonl data/train_${L}_*.jsonl; done | tr ' ' '\n' | sort -u) \
  data/train_*_gap.jsonl data/train_*_trap2.jsonl data/train_*_trap3.jsonl data/train_*_cjknum.jsonl
echo GAP-GEN-DONE

# The g16 mix, unchanged, plus this round's files twice.
NEWRX="train_(ja|ko|hi|id|vi|th|it|fr|ru|pt|zh-hant)|_gap|_trap2|_trap3|_cjknum"
OLD=$(ls data/train_*.jsonl | grep -vE "early|train_hard|_new|_num|_mix2|_hol2|$NEWRX" | tr '\n' ' ')
ML=$(ls data/train_{es,de,zh,ar}*.jsonl | grep -vE "_new|_num|_mix2|_hol2|$NEWRX" | tr '\n' ' ')
HARD=$(for f in data/train_hard*.jsonl; do echo "$f $f $f"; done | tr '\n' ' ')
NEW=$(for f in data/train_*_new.jsonl data/train_*_num.jsonl data/train_*_mix2.jsonl data/train_*_hol2.jsonl; do echo "$f $f"; done | tr '\n' ' ')
LANGF=$(for L in ja ko hi id vi th $NEWL; do for f in data/train_${L}.jsonl data/train_${L}_words.jsonl data/train_${L}_multi.jsonl; do echo "$f $f"; done; done | tr '\n' ' ')
GAP=$(for f in data/train_*_gap.jsonl data/train_*_trap2.jsonl data/train_*_trap3.jsonl data/train_*_cjknum.jsonl; do echo "$f $f"; done | tr '\n' ' ')
uv run python -m training.guard --max-gb 6 --gpu-ratio 0.2 --kill-below 15 -- uv run python -m training.model.model --train $OLD $ML $HARD $NEW $LANGF $GAP \
  --out timmytime/weights/timmytime-g17.npz --epochs 5 --batch 128 --stride 4 --n-dec 1 --n-enc 6 --resume timmytime/weights/timmytime-g16.pt
uv run python -m training.guard --threads 2 --max-gb 2 -- uv run python -m training.model.prune timmytime/weights/timmytime-g17.npz timmytime/weights/timmytime-g17p.npz data/train_*.jsonl
for L in hindsight $ALL_L; do TIMMYTIME_WEIGHTS=timmytime/weights/timmytime-g17p.npz uv run python -m training.guard --threads 1 --max-gb 2 -- uv run python -m training.eval.score timmytime --lang $L -v > data/score_g17_$L.txt 2>&1; done
echo G17-DONE
