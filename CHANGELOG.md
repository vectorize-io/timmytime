# Changelog

All notable changes to text2date are documented in this file.

## [0.1.1] - 2026-11-27

### Fixed

- Fast when several processes share the CPU (test workers, a busy server). On Linux, numpy's math library split each of the model's tiny matrix products across all cores, and the threads fought: with 6 processes on 6 cores a parse took ~350 ms instead of ~5 ms. text2date now runs its math on one thread and restores the setting afterwards. One thread is also faster alone (2.5 ms instead of 3.9 ms). New dependency: `threadpoolctl`.

## [0.1.0] - 2026-11-27

First release.

- `text2date.parse()` finds the date ranges in a question in 16 languages, with numpy only, in about 1 ms.
- Ships the [hs-date-1](https://huggingface.co/hindsight-ai/hs-date-1) model (4.8M parameters) inside the package.
- `text2date.hindsight.extract_temporal_constraint`: the shape Hindsight's recall expects.
