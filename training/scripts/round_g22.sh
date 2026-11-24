# Round g22: what g21 missed in hindsight's tests: named month + 初/中旬/底 (六月初), 100-365 days ago,
# fuzzy counts with longer tails (两三周前讨论了这个), more far ISO dates (2050-01-15). Templates only, no LLM.
# Fine-tune g21, prune, score all.   bash training/scripts/round_g22.sh
set -e
cd "$(dirname "$0")/../.."
ALL_L="en es de zh ar ja ko hi id vi th it fr ru pt zh-hant"
uv run python -m training.data.zh_templates --n 40000 --seed 3 --out data/train_zh_tpl.jsonl
uv run python -m training.data.en_templates --n 20000 --seed 3 --out data/train_en_tpl.jsonl
uv run python -m training.data.clean data/train_zh_tpl.jsonl data/train_en_tpl.jsonl

# The g21 mix, unchanged, plus the template files twice.
TPL="data/train_zh_tpl.jsonl data/train_en_tpl.jsonl"
OLD=$(ls data/train_*.jsonl | grep -vE "early|train_hard|_new|_num|_mix2|_hol2|_tpl" | tr '\n' ' ')
HARD=$(for f in data/train_hard*.jsonl; do echo "$f $f $f"; done | tr '\n' ' ')
NEW=$(for f in data/train_*_new.jsonl data/train_*_num.jsonl data/train_*_mix2.jsonl data/train_*_hol2.jsonl; do echo "$f $f"; done | tr '\n' ' ')
G=$(for f in $TPL; do echo "$f $f"; done | tr '\n' ' ')
uv run python -m training.guard --max-gb 6 --gpu-ratio 0.2 --kill-below 15 -- uv run python -m training.model.model --train $OLD $HARD $NEW $G \
  --out timmytime/weights/timmytime-g22.npz --epochs 3 --batch 128 --stride 4 --n-dec 1 --n-enc 6 --resume timmytime/weights/timmytime-g21.pt
uv run python -m training.guard --threads 2 --max-gb 2 -- uv run python -m training.model.prune timmytime/weights/timmytime-g22.npz timmytime/weights/timmytime-g22p.npz data/train_*.jsonl
for L in hindsight $ALL_L; do TIMMYTIME_WEIGHTS=timmytime/weights/timmytime-g22p.npz uv run python -m training.guard --threads 1 --max-gb 2 -- uv run python -m training.eval.score timmytime --lang $L -v > data/score_g22_$L.txt 2>&1; done
echo G22-DONE
