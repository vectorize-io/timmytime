# Round g18: what g17 still missed on data/test_hindsight.jsonl (82.2%): Chinese "since / from X on /
# before / after", rolling windows (三天内, 未来一周, 前5天), day ranges (明后天, 6月5日到8日), 大前天 and
# friends, 一两/两三 fuzzy counts, name-and-word traps (周杰伦, 月饼), far years (1890, 2050), times of day.
# Fine-tune g17, prune, score all.   Needs TIMMYTIME_LLM_KEY.   bash training/scripts/round_g18.sh
set -e
cd "$(dirname "$0")/../.."
GEN="uv run python -m training.guard --max-gb 1 -- uv run python -m training.data.gen --workers 24"
ALL_L="en es de zh ar ja ko hi id vi th it fr ru pt zh-hant"

t_zh() {
  L=$1
  $GEN --split train --lang $L --n 2000 --seed 10100 --none-share 0 --kinds open \
    --style "say since / from then on / before / after with words like 以来, 开始, 起, 迄今, 至今, 之前, 以前, 以后, 之后" --out data/train_${L}_since.jsonl
  $GEN --split train --lang $L --n 2000 --seed 10200 --none-share 0 --kinds last_n,half \
    --style "say rolling windows with words like N天内, 最近N, 过去N, 未来N, 前N天, 两三天, 半个月, 半年, 24小时" --out data/train_${L}_roll.jsonl
  $GEN --split train --lang $L --n 2000 --seed 10300 --none-share 0 --kinds range \
    --style "say day ranges briefly: 明后天, 今明两天, 昨今两天, 6月5日到8日, 5至8日, 5-8日, 周五到周一" --out data/train_${L}_dayrange.jsonl
  $GEN --split train --lang $L --n 2000 --seed 10400 --none-share 0 --kinds day,week,month,half \
    --style "use spoken offsets: 大前天, 大后天, 大大前天, 一周前, 一个星期前, 两週前, 俩月前, 一个月前, 半个月后, 一百天前" --out data/train_${L}_offset.jsonl
  $GEN --split train --lang $L --n 1500 --seed 10500 --none-share 0 --kinds fuzzy \
    --style "say vague counts: 一两天, 两三天, 三两天, 前几天, 数天前, 一两周, 两三个月, 一两年" --out data/train_${L}_fuzzy.jsonl
  $GEN --split train --lang $L --n 2000 --seed 10600 --none-share 1 \
    --style "no date at all, but the text has date characters inside names, foods, titles or habits: singer and show names starting with 周 or 上周, 月饼, 月亮, 年糕, 周末端, 第一季 of a TV show, 每年, 每个星期, 隔周, 后天免疫" \
    --out data/train_${L}_trap4.jsonl
}
t_zh zh & t_zh zh-hant & wait

t_far() {
  $GEN --split train --lang $1 --n 800 --seed 10700 --none-share 0 --kinds date,year_abs,month_name,range \
    --style "use years far from today, between 1800 and 1979 or between 2030 and 2199" --out data/train_$1_far.jsonl
}
for L in $ALL_L; do t_far $L & [ $((++i % 4)) = 0 ] && wait || true; done; wait
$GEN --split train --lang en --n 1500 --seed 10800 --none-share 0 --kinds day,week,month,range \
  --style "mention a time of day (this morning, tonight, this evening, at noon) or short forms (1mon ago, 2wks ago, yday, tmrw)" --out data/train_en_tod.jsonl

G18="data/train_zh*_since.jsonl data/train_zh*_roll.jsonl data/train_zh*_dayrange.jsonl data/train_zh*_offset.jsonl data/train_zh*_fuzzy.jsonl data/train_zh*_trap4.jsonl data/train_*_far.jsonl data/train_en_tod.jsonl"
uv run python -m training.data.clean $G18
echo G18-GEN-DONE

# The g17 mix, unchanged, plus this round's files three times.
G18RX="_since|_roll|_dayrange|_offset|_fuzzy|_trap4|_far|_tod"
OLD=$(ls data/train_*.jsonl | grep -vE "early|train_hard|_new|_num|_mix2|_hol2|$G18RX" | tr '\n' ' ')
HARD=$(for f in data/train_hard*.jsonl; do echo "$f $f $f"; done | tr '\n' ' ')
NEW=$(for f in data/train_*_new.jsonl data/train_*_num.jsonl data/train_*_mix2.jsonl data/train_*_hol2.jsonl; do echo "$f $f"; done | tr '\n' ' ')
G=$(for f in $G18; do echo "$f $f $f"; done | tr '\n' ' ')
uv run python -m training.guard --max-gb 6 --gpu-ratio 0.2 --kill-below 15 -- uv run python -m training.model.model --train $OLD $HARD $NEW $G \
  --out timmytime/weights/timmytime-g18.npz --epochs 3 --batch 128 --stride 4 --n-dec 1 --n-enc 6 --resume timmytime/weights/timmytime-g17.pt
uv run python -m training.guard --threads 2 --max-gb 2 -- uv run python -m training.model.prune timmytime/weights/timmytime-g18.npz timmytime/weights/timmytime-g18p.npz data/train_*.jsonl
for L in hindsight $ALL_L; do TIMMYTIME_WEIGHTS=timmytime/weights/timmytime-g18p.npz uv run python -m training.guard --threads 1 --max-gb 2 -- uv run python -m training.eval.score timmytime --lang $L -v > data/score_g18_$L.txt 2>&1; done
echo G18-DONE
