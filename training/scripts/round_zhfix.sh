# Fix round on the release candidate bsoup for the 3 Chinese misses in hindsight's tests: 前两个月 read as "two
# months ago" (50 LLM rows said so; clean.py now relabels 前N个月/前N周, N >= 2, to "the last N"), 这两三天 rare in
# training (new template rows), and the tail 计划 / 計劃 without 的. Then 3 fine-tunes of bsoup, averaged -> csoup.
#   bash training/scripts/round_zhfix.sh
set -e
cd "$(dirname "$0")/../.."
uv run python -m training.data.zh_templates --n 40000 --seed 9 --out data/train_zh_tpl.jsonl
uv run python -m training.data.clean data/train_*.jsonl | grep -v " 0 relabelled" || true
BASE=bsoup TAG=c DS=2 bash training/scripts/round_soup.sh
