# Dates on the 29th-31st: every sampler stopped at day 28 until g25, so the model barely saw them and the
# test sets never asked. A test set per language (data/test_<lang>_days.jsonl, scored as days-<lang>) and
# training rows. Needs TIMMYTIME_LLM_KEY.   bash training/scripts/days_data.sh
set -e
cd "$(dirname "$0")/../.."
GEN="uv run python -m training.guard --max-gb 1 -- uv run python -m training.data.gen --workers 16"
K="date,sameday,range,modified,open"
d() {
  $GEN --split test --lang $1 --n 250 --seed 20100 --none-share 0 --kinds $K --month-end-share 1 --out data/test_$1_days.jsonl
  $GEN --split train --lang $1 --n 1500 --seed 20200 --none-share 0 --kinds $K --month-end-share 0.6 --out data/train_$1_days.jsonl
}
for L in en es de zh ar ja ko hi id vi th it fr ru pt zh-hant; do d $L & [ $((++i % 4)) = 0 ] && wait || true; done; wait
uv run python -m training.data.clean data/train_*_days.jsonl
echo DAYS-DONE
