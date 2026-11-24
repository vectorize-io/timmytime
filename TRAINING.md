# Training and evaluating timmytime

Everything here is for people changing the model. Users only need README.md.

## Status (2026-11-24, shipped weights `weights-g16p`, next: g22/g23, not released)

6.4M weights before pruning, 9 MB after. Bytes in, shortened 4x by a conv, 6 encoder layers,
1 decoder layer, a 1271-word output vocabulary (years 1800-2199, numbers -365..365). 16 languages from g17 on;
per-language results are in README.md. Test sets: `data/test_<lang>_big.jsonl` (~3k rows each,
every kind) plus the older, smaller sets, and `data/test_hindsight.jsonl`: hindsight's own
query-analyzer tests and golden corpus as timmytime codes (`score timmytime --lang hindsight`).

The goal is to replace dateparser in hindsight with the model alone. Gaps are fixed with training
data, not with rules in the code. Where hindsight's tests and timmytime read a phrase differently,
timmytime's reading wins: "N units ago" is that whole unit, "before X" is open-ended, early / mid /
late are thirds.

How we got here (exact match on the English sets; each step fine-tunes the one before):

| Weights | What changed | Result |
|---|---|---|
| r1-r10 | English only, 4 layers; LLM-written data, round-trip checked | 98% on the first English set |
| g2-g7 | + es, de, zh, ar; hard-example mining; second opinion on no-date labels | 93-97.5% in the new languages |
| g10, g10p | grown to 6 layers by copying layers; dead feed-forward units pruned | same answers, 15% faster |
| g14 | holidays, "in N weeks", part-of-month, numbers that are not dates; 3k test rows per language | 92-93% -> 95.5-98.1% on the big sets |
| g15 | mining round on holidays and the new kinds | holidays 89-97% |
| g16, g16p | + ja, ko, hi, id, vi, th; plain "Christmas" fixed | new languages 94-96%, others unchanged or better |
| g17 | + it, fr, ru, pt, zh-hant; gap data for all 16 (same day last year, the 3rd of last month, day ranges, since, half a month, far years, name/idiom/config traps, Chinese numerals) | hindsight set 63.9% -> 82.2%; new languages 94-96% |
| g18, g19 | Chinese since / within / spoken offsets; short search-box queries; far years for real (`gen --far-share`) | hindsight set 86.6%, then 88.8% |
| g20 | template data, no LLM (`training/data/zh_templates.py`, `en_templates.py`) | hindsight set 97.2%; hindsight's unit tests 438/452 |
| g21 | fixed templates (这两天, 中了奖, 前两天) | hindsight set 97.7%; unit tests 447/451 |
| g22 | month name + 初/中旬/底, longer tails, more far ISO dates | hindsight set 98.3%; unit tests 448/451; 15 of 16 languages up |
| g23 | vocabulary grows to 100..365 ("100 days ago" could not be written); round far years | training |

Tried and dropped: averaging two models' weights, more rounds of blind data, 5 layers, folding the
first conv into a lookup table, a lower confidence cut-off per language picked on test data.

## Layout

One repo, two parts. `timmytime/` is the package and needs numpy only; `training/` may import `timmytime`,
never the reverse (`tests/test_runtime.py` checks it).

| Path | What it does |
|---|---|
| `timmytime/runtime.py` | runs the model in numpy; `Parser.parse()` |
| `timmytime/resolve.py` | meaning code + anchor + time zone -> date ranges (the grammar is at the top) |
| `timmytime/hindsight.py` | drop-in for hindsight's `extract_temporal_constraint` |
| `timmytime/holidays.json` | lunar holiday dates 1950-2076 (`training/scripts/make_holidays.py`) |
| `timmytime/weights/timmytime.npz` | the model; not in git, pinned by `weights.lock`, fetched by `scripts/get_weights.sh` |
| `training/data/meanings.py` | random meanings with plain-English descriptions, for data generation |
| `training/data/gen.py`, `llm.py` | writes labelled questions with an LLM (Gemma on hindsight dev inference), checks each |
| `training/data/zh_templates.py`, `en_templates.py` | short queries from templates, with their codes (no LLM) |
| `training/data/clean.py` | drops training rows that are in a test set; applies English label rules |
| `training/data/verify.py`, `audit.py` | second opinion on no-date labels; marks disputed test rows |
| `training/model/model.py` | the torch model, training, export to `.npz` |
| `training/model/prune.py`, `tune.py` | drops feed-forward units that never switch on; picks the confidence cut-off |
| `training/eval/score.py`, `mine.py` | scores `baseline`, `gemma` or `timmytime` on the test sets; keeps the rows a model gets wrong |
| `training/eval/baseline/` | hindsight's old dateparser path, copied from main `f6038a742`, for comparison |
| `training/guard.py` | runs a job inside memory/CPU/GPU limits; every heavy job goes through it |
| `training/scripts/round.sh` | one full mining round, unattended |
| `training/scripts/round_g17.sh` ... `round_g21.sh` | the rounds above, each one script, unattended |
| `training/tests/` | numpy/torch parity, code round trip, old-rule equivalence, hindsight's golden file |
| `data/test_*.jsonl`, `data/hand_en.tsv`, `data/tune_*.jsonl` | test, hand-written and tuning sets (in git) |

## Commands

```sh
uv sync                                          # dev + training groups (torch, dateparser)
scripts/get_weights.sh                           # the pinned weights, for scoring and tests
set -a; source ~/nicolo-secrets.env; set +a     # HINDSIGHT_INFERENCE_DEV_API_KEY (or TIMMYTIME_LLM_KEY) for Gemma
G="uv run python -m training.guard --max-gb 6 --gpu-ratio 0.2 --"
$G uv run python -m training.data.gen --split train --lang en --n 40000 --seed 5 --out data/train_en_b.jsonl
uv run python -m training.data.clean data/train_*.jsonl
$G uv run python -m training.model.model --train data/train_*.jsonl --out timmytime/weights/new.npz --epochs 30 --batch 128 --stride 4 --n-dec 1 --n-enc 6
uv run python -m training.model.prune timmytime/weights/new.npz timmytime/weights/newp.npz data/train_*.jsonl
TIMMYTIME_WEIGHTS=timmytime/weights/newp.npz uv run python -m training.eval.score timmytime --lang en -v
PYTEST_ADDOPTS= uv run pytest -q
```

Training data (`data/train_*.jsonl`) is not in git; regenerate it with the commands above.

## Shipping new weights

1. `scripts/publish_weights.sh <name> timmytime/weights/newp.npz`: uploads them to a GitHub release
   `weights-<name>`, installs them locally and rewrites `weights.lock`.
2. Run the tests (the README table must still pass), commit `weights.lock`.
3. Bump `version` in `pyproject.toml`, tag `v<version>` and push the tag: the release workflow
   builds the package with those weights inside and attaches it to a GitHub release.
