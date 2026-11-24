# Round g19: what g18 still missed on data/test_hindsight.jsonl (86.6%): very short Chinese queries
# (上上周, 大前年, 当天, 第一季度, 星期一, 三天内, 明天起), far years for real (--far-share; g18's style
# alone kept recent years), times of day (this morning = today). Fine-tune g18, prune, score all.
# Needs TIMMYTIME_LLM_KEY.   bash training/scripts/round_g19.sh
set -e
cd "$(dirname "$0")/../.."
GEN="uv run python -m training.guard --max-gb 1 -- uv run python -m training.data.gen --workers 24"
ALL_L="en es de zh ar ja ko hi id vi th it fr ru pt zh-hant"
SHORT="a very short search-box query, 2 to 10 characters, no full sentence"

t_zh() {
  L=$1
  $GEN --split train --lang $L --n 3000 --seed 11100 --none-share 0.1 --kinds day,week,month,year,quarter,weekday,weekend,modified \
    --style "$SHORT, with words like 上上周, 下下周, 大前年, 大上周, 当天, 当月, 當週, 下一个月, 前一周, 第一季度, 星期一, 礼拜五, 年尾, 上旬, 中旬" --out data/train_${L}_short.jsonl
  $GEN --split train --lang $L --n 2000 --seed 11200 --none-share 0 --kinds open,last_n,half,range,fuzzy \
    --style "$SHORT, with words like 以来, 开始, 起, 之前, 以后, N天内, 一週內, 前N天, 未来N, 半个月, 一两周, 两三个月, 5-8日" --out data/train_${L}_short2.jsonl
  $GEN --split train --lang $L --n 1500 --seed 11300 --none-share 1 \
    --style "$SHORT with no date: a name, food or word that contains date characters (周杰伦, 周星驰, 月饼, 月亮, 年糕, 周末端, 后天免疫, 每年, 每周)" --out data/train_${L}_trap5.jsonl
}
t_zh zh & t_zh zh-hant & wait

t_far() {
  $GEN --split train --lang $1 --n 800 --seed 11400 --far-share 0.9 --none-share 0 --kinds date,year_abs,month_name,range,decade \
    --out data/train_$1_far2.jsonl
}
for L in $ALL_L; do t_far $L & [ $((++i % 4)) = 0 ] && wait || true; done; wait
$GEN --split train --lang en --n 1000 --seed 11500 --none-share 0 --kinds day \
  --style "say the day with a time of day: this morning, tonight, this evening, last night, tomorrow morning" --out data/train_en_tod2.jsonl

G19="data/train_zh*_short.jsonl data/train_zh*_short2.jsonl data/train_zh*_trap5.jsonl data/train_*_far2.jsonl data/train_en_tod2.jsonl"
uv run python -m training.data.clean $G19
echo G19-GEN-DONE

# The g18 mix, unchanged, plus this round's files three times.
G19RX="_short|_short2|_trap5|_far2|_tod2"
OLD=$(ls data/train_*.jsonl | grep -vE "early|train_hard|_new|_num|_mix2|_hol2|$G19RX" | tr '\n' ' ')
HARD=$(for f in data/train_hard*.jsonl; do echo "$f $f $f"; done | tr '\n' ' ')
NEW=$(for f in data/train_*_new.jsonl data/train_*_num.jsonl data/train_*_mix2.jsonl data/train_*_hol2.jsonl; do echo "$f $f"; done | tr '\n' ' ')
G=$(for f in $G19; do echo "$f $f $f"; done | tr '\n' ' ')
uv run python -m training.guard --max-gb 6 --gpu-ratio 0.2 --kill-below 15 -- uv run python -m training.model.model --train $OLD $HARD $NEW $G \
  --out timmytime/weights/timmytime-g19.npz --epochs 3 --batch 128 --stride 4 --n-dec 1 --n-enc 6 --resume timmytime/weights/timmytime-g18.pt
uv run python -m training.guard --threads 2 --max-gb 2 -- uv run python -m training.model.prune timmytime/weights/timmytime-g19.npz timmytime/weights/timmytime-g19p.npz data/train_*.jsonl
for L in hindsight $ALL_L; do TIMMYTIME_WEIGHTS=timmytime/weights/timmytime-g19p.npz uv run python -m training.guard --threads 1 --max-gb 2 -- uv run python -m training.eval.score timmytime --lang $L -v > data/score_g19_$L.txt 2>&1; done
echo G19-DONE
