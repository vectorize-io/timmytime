# "N units ago" / "the last N units" up to 365, "a month ago", far years (issues #1 and #2: the old rows were
# written before the samplers drew these). A test set per language (data/test_<lang>_counts.jsonl, scored as
# counts-<lang>) and training rows. Needs TIMMYTIME_LLM_KEY.   bash training/scripts/counts_data.sh
set -e
cd "$(dirname "$0")/../.."
GEN="uv run python -m training.guard --max-gb 1 -- uv run python -m training.data.gen --workers 16"
K="day,week,month,year,last_n,year_abs,date"
c() {
  $GEN --split test --lang $1 --n 250 --seed 30100 --none-share 0 --kinds $K --far-share 0.3 --far-count-share 0.5 --one-ago-share 0.6 --out data/test_$1_counts.jsonl
  $GEN --split train --lang $1 --n 1500 --seed 30200 --none-share 0 --kinds $K --far-share 0.3 --far-count-share 0.4 --one-ago-share 0.5 --out data/train_$1_counts.jsonl
}
for L in en es de zh ar ja ko hi id vi th it fr ru pt zh-hant; do c $L & [ $((++i % 4)) = 0 ] && wait || true; done; wait
uv run python -m training.data.clean data/train_*_counts.jsonl
echo COUNTS-DONE
