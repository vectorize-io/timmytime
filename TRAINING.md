# Training and evaluating tlm

Everything here is for people changing the model. Users only need README.md.

## Status (2026-11-23, weights `weights-g16p`)

6.4M weights before pruning, 9 MB after. Bytes in, shortened 4x by a conv, 6 encoder layers,
1 decoder layer, a 528-word output vocabulary. 11 languages; per-language results are in README.md.
Test sets: `data/test_<lang>_big.jsonl` (~3k rows each, every kind) plus the older, smaller sets.

How we got here (exact match on the English sets; each step fine-tunes the one before):

| Weights | What changed | Result |
|---|---|---|
| r1-r10 | English only, 4 layers; LLM-written data, round-trip checked | 98% on the first English set |
| g2-g7 | + es, de, zh, ar; hard-example mining; second opinion on no-date labels | 93-97.5% in the new languages |
| g10, g10p | grown to 6 layers by copying layers; dead feed-forward units pruned | same answers, 15% faster |
| g14 | holidays, "in N weeks", part-of-month, numbers that are not dates; 3k test rows per language | 92-93% -> 95.5-98.1% on the big sets |
| g15 | mining round on holidays and the new kinds | holidays 89-97% |
| g16, g16p | + ja, ko, hi, id, vi, th; plain "Christmas" fixed | new languages 94-96%, others unchanged or better |

Tried and dropped: averaging two models' weights, more rounds of blind data, 5 layers, folding the
first conv into a lookup table, a lower confidence cut-off per language picked on test data.

## Layout

One repo, two parts. `tlm/` is the package and needs numpy only; `training/` may import `tlm`,
never the reverse (`tests/test_runtime.py` checks it).

| Path | What it does |
|---|---|
| `tlm/runtime.py` | runs the model in numpy; `Parser.parse()` |
| `tlm/resolve.py` | meaning code + anchor + time zone -> date ranges (the grammar is at the top) |
| `tlm/hindsight.py` | drop-in for hindsight's `extract_temporal_constraint` |
| `tlm/holidays.json` | lunar holiday dates 1950-2076 (`training/scripts/make_holidays.py`) |
| `tlm/weights/tlm.npz` | the model; not in git, pinned by `weights.lock`, fetched by `scripts/get_weights.sh` |
| `training/data/meanings.py` | random meanings with plain-English descriptions, for data generation |
| `training/data/gen.py`, `llm.py` | writes labelled questions with an LLM (Gemma on hindsight dev inference), checks each |
| `training/data/clean.py` | drops training rows that are in a test set; applies English label rules |
| `training/data/verify.py`, `audit.py` | second opinion on no-date labels; marks disputed test rows |
| `training/model/model.py` | the torch model, training, export to `.npz` |
| `training/model/prune.py`, `tune.py` | drops feed-forward units that never switch on; picks the confidence cut-off |
| `training/eval/score.py`, `mine.py` | scores `baseline`, `gemma` or `tlm` on the test sets; keeps the rows a model gets wrong |
| `training/eval/baseline/` | hindsight's old dateparser path, copied from main `f6038a742`, for comparison |
| `training/guard.py` | runs a job inside memory/CPU/GPU limits; every heavy job goes through it |
| `training/scripts/round.sh` | one full mining round, unattended |
| `training/tests/` | numpy/torch parity, code round trip, old-rule equivalence, hindsight's golden file |
| `data/test_*.jsonl`, `data/hand_en.tsv`, `data/tune_*.jsonl` | test, hand-written and tuning sets (in git) |
| `integration/` | patch that adds `TlmQueryAnalyzer` to hindsight behind `HINDSIGHT_API_QUERY_ANALYZER=tlm` |

## Commands

```sh
uv sync                                          # dev + training groups (torch, dateparser)
scripts/get_weights.sh                           # the pinned weights, for scoring and tests
set -a; source ~/nicolo-secrets.env; set +a     # HINDSIGHT_INFERENCE_DEV_API_KEY for Gemma
G="uv run python -m training.guard --max-gb 6 --gpu-ratio 0.2 --"
$G uv run python -m training.data.gen --split train --lang en --n 40000 --seed 5 --out data/train_en_b.jsonl
uv run python -m training.data.clean data/train_*.jsonl
$G uv run python -m training.model.model --train data/train_*.jsonl --out tlm/weights/new.npz --epochs 30 --batch 128 --stride 4 --n-dec 1 --n-enc 6
uv run python -m training.model.prune tlm/weights/new.npz tlm/weights/newp.npz data/train_*.jsonl
TLM_WEIGHTS=tlm/weights/newp.npz uv run python -m training.eval.score tlm --lang en -v
PYTEST_ADDOPTS= uv run pytest -q
```

Training data (`data/train_*.jsonl`) is not in git; regenerate it with the commands above.

## Shipping new weights

1. `scripts/publish_weights.sh <name> tlm/weights/newp.npz`: uploads them to a GitHub release
   `weights-<name>`, installs them locally and rewrites `weights.lock`.
2. Run the tests (the README table must still pass), commit `weights.lock`.
3. Bump `version` in `pyproject.toml`, tag `v<version>` and push the tag: the release workflow
   builds the package with those weights inside and attaches it to a GitHub release.
