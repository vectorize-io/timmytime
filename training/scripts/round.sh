# One hard-example mining round, unattended:  bash training/scripts/round.sh <round> <seed> <base model, e.g. g7>
# fresh pool (training prompts) -> mine with the base model -> verify mined no-date rows ->
# fine-tune base -> score every language. Each step goes through training.guard.
set -e
R=$1; SEED=$2; BASE=$3
set -a; source ~/nicolo-secrets.env >/dev/null 2>&1; set +a
cd ~/dev/tlm
GEN="uv run python -m training.guard --max-gb 1 -- uv run python -m training.data.gen --split train --workers 24"
$GEN --lang en --n 8000 --seed $SEED --none-share 0 --kinds month_name --out data/pool${R}_en_month.jsonl
$GEN --lang en --n 8000 --seed $((SEED+1)) --none-share 0 --kinds multi --out data/pool${R}_en_multi.jsonl
$GEN --lang en --n 8000 --seed $((SEED+2)) --none-share 1 --out data/pool${R}_en_none.jsonl
$GEN --lang ar --n 8000 --seed $((SEED+3)) --out data/pool${R}_ar.jsonl
uv run python -m training.data.clean data/pool${R}_*.jsonl
export TLM_WEIGHTS=tlm/weights/tlm-ml-$BASE.npz
uv run python -m training.guard --threads 1 --max-gb 2 -- uv run python -m training.eval.mine data/pool${R}_*.jsonl > data/hard${R}.jsonl 2> data/mine${R}.log
python3 - $R <<'PY'
import json, sys
r = sys.argv[1]
rows = [json.loads(l) for l in open(f"data/hard{r}.jsonl")]
open(f"data/hard{r}_none.jsonl", "w").writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rows if x["code"] == "none")
open(f"data/hard{r}_dated.jsonl", "w").writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rows if x["code"] != "none")
PY
uv run python -m training.data.verify data/hard${R}_none.jsonl data/hard${R}_none.verified.jsonl
python3 - $R <<'PY'
import glob, json, sys
r = sys.argv[1]
hard_none = {json.loads(l)["q"] for l in open(f"data/hard{r}_none.jsonl")}
rows = [l for f in sorted(glob.glob(f"data/pool{r}_*.jsonl")) for l in open(f)
        if not (json.loads(l)["code"] == "none" and json.loads(l)["q"] in hard_none)]
open(f"data/train_pool{r}.jsonl", "w").writelines(rows)
open(f"data/train_hard{r}.jsonl", "w").writelines(list(open(f"data/hard{r}_dated.jsonl")) + list(open(f"data/hard{r}_none.verified.jsonl")))
PY
unset TLM_WEIGHTS
ALL=$(ls data/train_*.jsonl | grep -vE "early|train_hard" | tr '\n' ' ')
ML=$(ls data/train_{es,de,zh,ar}*.jsonl | tr '\n' ' ')
HARD=$(for f in data/train_hard*.jsonl; do if [ "$f" = "data/train_hard${R}.jsonl" ]; then echo "$f $f $f $f $f"; else echo "$f $f $f"; fi; done | tr '\n' ' ')
NEXT=g$R
uv run python -m training.guard --max-gb 6 --gpu-ratio 0.2 --kill-below 15 -- uv run python -m training.model.model --train $ALL $ML $HARD \
  --out tlm/weights/tlm-ml-$NEXT.npz --epochs 4 --batch 128 --stride 4 --n-dec 1 --n-enc ${N_ENC:-4} --resume tlm/weights/tlm-ml-$BASE.pt
for L in en es de zh ar; do TLM_WEIGHTS=tlm/weights/tlm-ml-$NEXT.npz uv run python -m training.guard --threads 1 --max-gb 2 -- uv run python -m training.eval.score tlm --lang $L -v > data/score_${NEXT}_$L.txt 2>&1; done
echo ROUND-$R-DONE
