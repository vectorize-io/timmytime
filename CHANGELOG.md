# Changelog

All notable changes to timmytime are documented in this file.

## [0.1.0] - 2026-11-27

First release.

- `timmytime.parse()` finds the date ranges in a question in 16 languages, with numpy only, in about 1 ms.
- Ships the [hs-date-1](https://huggingface.co/hindsight-ai/hs-date-1) model (4.8M parameters) inside the package.
- `timmytime.hindsight.extract_temporal_constraint`: the shape Hindsight's recall expects.
