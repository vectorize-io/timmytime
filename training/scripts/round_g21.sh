# Round g21: the g20 templates, fixed (这两天 = last 2 days; 中 after a month is "won", not "mid"; 曾经; 年糕/末端 traps;
# 100+ days ago; more far years). Short Chinese queries from training/data/zh_templates.py,
# and English ones (en_templates.py: "port 9077 and also last Tuesday", June 0000, far ISO dates).
# No LLM calls. Fine-tune g20, prune, score all.   bash training/scripts/round_g21.sh
set -e
cd "$(dirname "$0")/../.."
ALL_L="en es de zh ar ja ko hi id vi th it fr ru pt zh-hant"
uv run python -m training.data.zh_templates --n 40000 --seed 2 --out data/train_zh_tpl.jsonl
uv run python -m training.data.en_templates --n 6000 --seed 2 --out data/train_en_tpl.jsonl
uv run python -m training.data.clean data/train_zh_tpl.jsonl data/train_en_tpl.jsonl

# The g20 mix, unchanged, plus the template files twice.
TPL="data/train_zh_tpl.jsonl data/train_en_tpl.jsonl"
OLD=$(ls data/train_*.jsonl | grep -vE "early|train_hard|_new|_num|_mix2|_hol2|_tpl" | tr '\n' ' ')
HARD=$(for f in data/train_hard*.jsonl; do echo "$f $f $f"; done | tr '\n' ' ')
NEW=$(for f in data/train_*_new.jsonl data/train_*_num.jsonl data/train_*_mix2.jsonl data/train_*_hol2.jsonl; do echo "$f $f"; done | tr '\n' ' ')
G=$(for f in $TPL; do echo "$f $f"; done | tr '\n' ' ')
uv run python -m training.guard --max-gb 6 --gpu-ratio 0.2 --kill-below 15 -- uv run python -m training.model.model --train $OLD $HARD $NEW $G \
  --out timmytime/weights/timmytime-g21.npz --epochs 3 --batch 128 --stride 4 --n-dec 1 --n-enc 6 --resume timmytime/weights/timmytime-g20.pt
uv run python -m training.guard --threads 2 --max-gb 2 -- uv run python -m training.model.prune timmytime/weights/timmytime-g21.npz timmytime/weights/timmytime-g21p.npz data/train_*.jsonl
for L in hindsight $ALL_L; do TIMMYTIME_WEIGHTS=timmytime/weights/timmytime-g21p.npz uv run python -m training.guard --threads 1 --max-gb 2 -- uv run python -m training.eval.score timmytime --lang $L -v > data/score_g21_$L.txt 2>&1; done
echo G20-DONE
