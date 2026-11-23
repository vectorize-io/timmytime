# Training and evaluating tlm

Everything here is for people changing the model. Users only need README.md.

## Status (2026-11-20, model `tlm/weights/tlm.npz` = multilingual g10p: en, es, de, zh, ar)

numpy only. Bytes in, shortened 4x by a conv, 6 encoder layers, 1 decoder layer. g10p is g10 with
the feed-forward units that never switch on removed (`tlm/prune.py`, 38% of them): same answers on
every test set, ~15% faster. The older 4-layer g7 is kept as `tlm-ml-g7.npz`.
One model for all languages: fine-tuned from the English model, then on ~330k rows including
hard examples mined from fresh pools (`tlm/mine.py`) and re-checked no-date labels (`tlm/verify.py`).

| Test set | Rows | tlm exact | tlm false dates | old code exact | old code false dates |
|---|---|---|---|---|---|
| English, Gemma-written (`data/test_en.jsonl` + `_2`) | 2,787 | 98.6% | 0.7% (0.5%*) | 54% | 28% |
| English, small kinds (`data/test_en_3.jsonl`) | 755 | 97.0% | - | - | - |
| English, hand-written traps (`data/hand_en.tsv`) | 126 | 99.2% | 0.0% | 53% | 32% |
| German (`data/test_de.jsonl`) | 642 | 97.8% | 1.1% (0.8%*) | 52% | 14% |
| Spanish (`data/test_es.jsonl`) | 643 | 95.5% | 1.1% (0.4%*) | 54% | 13% |
| Chinese (`data/test_zh.jsonl`) | 662 | 96.2% | 1.1% (0.4%*) | 70% | 3.3% |
| Arabic (`data/test_ar.jsonl`) | 656 | 94.5% | 0.3% (0.0%*) | 47% | 4.0% |

\* leaving out no-date test rows the LLM itself reads a date in (`tlm/audit.py`, model-independent).
English per kind (3,542 rows): every kind at 95% or more; lowest are month names 95% (426) and years 96% (96).
Confidence cut-off 0.5 (`MIN_CONFIDENCE`), picked on `data/tune_*.jsonl` by `tlm/tune.py`.
Speed (numpy, one thread, busy machine, English): p50 1.2 ms, p95 1.9 ms.
hindsight: `integration/` patch, tested on a local branch (614 hindsight tests pass), not applied upstream.

Speed vs accuracy (2026-11-20, English test labels after the "this <month>" rule):

| Model | Encoder layers | English exact / false | Month names / two-date | Arabic | p95 (quiet) |
|---|---|---|---|---|---|
| `tlm-ml-g7.npz` | 4 | 98.4% / 0.6% | 92% / 94% | 93.3% | 1.6-1.8 ms |
| `tlm-ml-g10.npz` | 6 | 98.6% / 0.7% | 95% / 96% | 94.5% | 2.0-2.2 ms |
| `tlm-ml-g10p.npz` (default) | 6, pruned | 98.6% / 0.7% | 95% / 96% | 94.5% | 1.9 ms (busy machine) |

Pruning makes g10 meet the 2 ms p95 target with identical answers.

## Files

| Path | What it does |
|---|---|
| `tlm/resolve.py` | meaning code + anchor + timezone -> date ranges (the grammar is at the top) |
| `tlm/runtime.py` | runs the model in numpy; `Parser.parse()` |
| `tlm/mine.py`, `tlm/verify.py`, `tlm/audit.py` | hard-example mining; second opinion on no-date labels; marks disputed test rows |
| `tlm/prune.py` | drops feed-forward units that never switch on (same answers, faster) |
| `tlm/tune.py` | picks the confidence cut-off on the tuning set |
| `integration/` | patch that adds `TlmQueryAnalyzer` to hindsight behind `HINDSIGHT_API_QUERY_ANALYZER=tlm` |
| `tlm/hindsight.py` | drop-in for hindsight's `extract_temporal_constraint`: first window as datetimes, never raises |
| `tlm/model.py` | the torch model, training, export to `tlm/weights/*.npz` |
| `tlm/meanings.py` | random meanings with plain-English descriptions, for data generation |
| `tlm/gen.py` | writes labelled questions with an LLM (Gemma on hindsight dev inference), checks each |
| `tlm/clean.py` | drops training rows that are in a test set; applies English label rules |
| `tlm/score.py` | scores `baseline` (hindsight's current code), `gemma` or `tlm` on the test sets |
| `tlm/guard.py` | runs a job inside memory/CPU/GPU limits; every heavy job goes through it |
| `tlm/baseline/` | hindsight's current temporal code, copied from main `f6038a742`, for comparison |
| `data/test_en.jsonl`, `data/hand_en.tsv` | the English test sets (frozen) |

## Commands

```sh
set -a; source ~/nicolo-secrets.env; set +a     # HINDSIGHT_INFERENCE_DEV_API_KEY for Gemma
G="uv run python -m tlm.guard --max-gb 6 --gpu-ratio 0.2 --"
$G uv run python -m tlm.gen --split train --n 40000 --seed 5 --out data/train_en_b.jsonl
uv run python -m tlm.clean data/train_*.jsonl
$G uv run python -m tlm.model --train data/train_*.jsonl --out tlm/weights/tlm.npz --epochs 30 --batch 128 --stride 4 --n-dec 1
uv run python -m tlm.score tlm -v
PYTEST_ADDOPTS= uv run pytest -q
```

Training data (`data/train_*.jsonl`) is not in git; regenerate it with the commands above.
