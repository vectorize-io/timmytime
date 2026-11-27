# Training and evaluating the model

Everything here is for people changing the model. Users only need README.md.

## The model

[hs-date-1](https://huggingface.co/hindsight-ai/hs-date-1): 4.8M weights (6.4M before pruning), a 9.7 MB
file. The question's UTF-8 bytes, shortened 4x by two convolutions, 6 encoder layers, 1 decoder layer,
256 wide. It writes a meaning code from a 1,271-word vocabulary (years 1800-2199, numbers -365..365);
`text2date/resolve.py` turns the code into dates.

Fixes come from training data, never from rules in the code. Where Hindsight's old tests and the model
read a phrase differently, the model's conventions win: "N units ago" is that whole unit, "before X" is
open-ended, early / mid / late are thirds.

## Data

[hs-date-1-data](https://huggingface.co/datasets/hindsight-ai/hs-date-1-data) holds the 940k training
questions (`train/`) and a copy of the test sets. The test sets also live here, in `data/`:

| Set | What it is |
|---|---|
| `data/test_<lang>_big.jsonl` | the main held-out set, ~3,000 rows per language, every kind of date |
| `data/test_<lang>_days.jsonl`, `_counts.jsonl` | dates on the 29th-31st; "N units ago" up to 365 |
| `data/test_hindsight.jsonl` | Hindsight's query-analyzer tests and golden corpus, as codes |
| `data/test_en*.jsonl`, `data/hand_en.tsv` | older English sets and hand-written traps |
| `data/tune_<lang>.jsonl` | for picking the confidence cut-off (`training/model/tune.py`) |

How the training data was made:

- `training/data/gen.py` picks random meanings (`meanings.py`), asks an LLM (Gemma 4, through
  `llm.py`) to write questions for them, and keeps a row only if its code turns back into the same dates.
  `verify.py` gets a second opinion on no-date labels.
- `zh_templates.py` and `en_templates.py` write short search-box queries with their codes, no LLM.
- `training/eval/mine.py` keeps the test-like rows a model gets wrong, to relabel and train on.
- `clean.py` drops training rows that are in a test set and applies the label conventions.

## Train

```sh
uv sync                                   # dev + training groups (torch, dateparser)
scripts/get_weights.sh                    # the released model, for tests and comparisons
bash training/train.sh                    # from scratch: downloads the data, 20 passes, prunes, scores
```

`train.sh` writes `text2date/weights/new.npz` (and `.pt`), the pruned `newp.npz`, and a score file per
set in `data/`. Every heavy job runs through `training/guard.py`, which kills it before it starves the
machine. One run from scratch takes about 6 hours on an M-series Mac. Runs with a different
`DATA_SEED` land 1-3 points apart: the data order matters, so train 2-3 and keep the best.

hs-date-1 was trained this way, then fine-tuned three times to fix Chinese gaps.

## Fix a gap

1. Write training rows for it: `gen.py`, the templates, or relabelled rows from `mine.py`. Never rules
   in the code. Clean them: `uv run python -m training.data.clean data/train_fix.jsonl`.
2. Fine-tune the released model (its unpruned checkpoint is in the Hugging Face repo) on everything
   plus the new rows, 3 passes at a lower learning rate:

   ```sh
   uvx --from huggingface_hub hf download hindsight-ai/hs-date-1 training/checkpoint.safetensors --local-dir data/hs-date-1
   EXTRA=data/train_fix.jsonl BASE=data/hs-date-1/training/checkpoint.safetensors OUT=text2date/weights/fix1 \
     DATA_SEED=1 bash training/train.sh finetune
   ```

   Do it with 2-3 `DATA_SEED`s. `training/model/soup.py` averages short fine-tunes of one model
   (never runs trained from scratch: they don't average). Keep the run or average that scores best.
3. Ship it only if it is no worse on every set: the `data/score_*.txt` files, ALL CAPS
   (`training/eval/caps.py`) and the tests.
4. After a few fixes, train from scratch again on everything, so fine-tunes don't pile up.

## Ship new weights

```sh
scripts/publish_weights.sh hindsight-ai/hs-date-1 text2date/weights/fix1p.npz
uv run pytest -q    # the README examples must still pass
```

`publish_weights.sh` writes `model.safetensors`, `config.json` and `vocab.json`, uploads them to the
Hugging Face repo (upload the new unpruned checkpoint as `training/checkpoint.safetensors` too), and pins that commit in `weights.lock`. Update the model card's numbers, commit
`weights.lock`, add a CHANGELOG.md entry and run `./release.sh patch`. A new vocabulary or a model
trained from scratch gets a new repo (`hs-date-2`), so old text2date versions never load it.

## Layout

`text2date/` is the package and needs numpy only; `training/` may import `text2date`, never the reverse
(`tests/test_runtime.py` checks it).

| Path | What it does |
|---|---|
| `text2date/runtime.py` | runs the model in numpy; `Parser.parse()` |
| `text2date/resolve.py` | meaning code + anchor + time zone -> date ranges (the grammar is at the top) |
| `text2date/hindsight.py` | drop-in for Hindsight's `extract_temporal_constraint` |
| `text2date/holidays.json` | lunar holiday dates 1950-2076 (`training/scripts/make_holidays.py`) |
| `training/train.sh` | the training recipe: from scratch or a fine-tune, then prune and score |
| `training/data/` | data generation, templates, cleaning |
| `training/model/model.py` | the torch model and its training loop |
| `training/model/prune.py`, `soup.py`, `tune.py`, `export.py` | drop unused units; average fine-tunes; pick the cut-off; write the Hugging Face files |
| `training/eval/score.py`, `caps.py` | scores `text2date`, `baseline` (dateparser) or `gemma` on the test sets |
| `training/eval/baseline/` | Hindsight's old dateparser path, copied from its main branch, for comparison |
| `training/tests/` | numpy/torch parity, code round trip, old-rule equivalence, Hindsight's golden file |
