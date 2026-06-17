# BugHunt Training Rules

Use this file as the project rule sheet for local and Kaggle training.

## Vision

Build one robust advanced trainer for vulnerability discovery, bug bounty report classification, HTTP attack-pattern learning, chain-impact prediction, live-data refresh, and reinforcement training.

## Operating Loop

Analyze the code and data first. Plan the smallest useful change. Debug with real commands. Make the change. Train on a bounded data slice. Correct failures. Increase coverage only after the previous run is stable. Repeat: analyze, iterate, plan, debug, make, train, correct.

## Supported Entry Points

- `advanced_train.py` is the single supported trainer.
- `training_workflow.py` is the supported wrapper for fresh, reinforcement, and cycle runs.
- `inference.py` is the supported local inference entry point.
- `scale_dataset.py` is the supported synthetic raw-data scaler.
- `live_data_collector.py` is the supported public live-data collector.

## Data Rules

- Curated raw files are loaded by default when present.
- Large generated bulk caches require an explicit percentage.
- Use `--data-percent N` for both bulk text and bulk HTTP caches.
- Use `--bulk-text-percent N` and `--bulk-http-percent N` when text and HTTP cache coverage should differ.
- Start local validation at `0` to `1` percent before increasing to `5`, `10`, or higher.
- Do not use `100` percent of the 21 GiB text cache unless the machine has enough runtime and memory headroom.

## Training Rules

- Run a dry run before a long training job.
- Use `--skip-collect` when training only from uploaded or cached raw data.
- Use live collection only when refreshing the dataset.
- Use `--resume-latest-best` for reinforcement.
- Use `--update-default-best` only when the run should promote the new checkpoint.
- Keep run names explicit for comparable experiments.

## Local Environment

Activate the requested local environment with:

```bash
source /home/rafal/miniforge3/etc/profile.d/conda.sh
conda activate ml
```

Then run:

```bash
python -m py_compile bugHunt/advanced_train.py bugHunt/training_workflow.py bugHunt/inference.py bugHunt/live_data_collector.py bugHunt/scale_dataset.py
python bugHunt/advanced_train.py --skip-collect --dry-run --print-options --data-percent 0
```

## Kaggle Rule

Follow `bugHunt/KAGGLE.md`. Upload raw data as a Kaggle dataset, point `BUGHUNT_RAW_DIR` at the Kaggle input path, select a conservative `--data-percent`, and reinforce only after a stable base checkpoint exists.
