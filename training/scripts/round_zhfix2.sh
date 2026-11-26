# csoup (round_zhfix.sh) fixed the 3 Chinese misses and passes 450/451 hindsight tests; the last one, 周一的会议,
# turned into next Monday: bare weekday templates had future tails (的计划) but never 的会议 in the past.
# New templates, then 3 fine-tunes of csoup, averaged -> dsoup.   bash training/scripts/round_zhfix2.sh
set -e
cd "$(dirname "$0")/../.."
uv run python -m training.data.zh_templates --n 40000 --seed 10 --out data/train_zh_tpl.jsonl
uv run python -m training.data.clean data/train_zh_tpl.jsonl
BASE=csoup TAG=d DS=3 bash training/scripts/round_soup.sh
