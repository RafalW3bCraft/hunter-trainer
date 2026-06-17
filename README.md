# BugHunt Advanced Trainer

This project trains one advanced bug bounty model from local cached data, live public vulnerability data, and generated HTTP samples. The supported training entry point is:

```bash
python bugHunt/advanced_train.py
```

Older trainer entry points were removed so all fresh training, cached training, live-data training, dashboard training, and reinforcement training use the same setup.

## Current Rules

- Use `advanced_train.py` for every training run.
- Use `training_workflow.py` only as a convenience wrapper for cleanup, fresh runs, and reinforcement cycles.
- Curated raw files in `data/raw/` are always loaded when present.
- Large generated bulk caches are never loaded accidentally. Select them with `--data-percent`, `--bulk-text-percent`, or `--bulk-http-percent`.
- Use a small percentage locally first, then increase only after the run is stable.
- Use `--resume-latest-best` or workflow `--mode reinforce` for reinforcement rounds.
- Keep raw data outside code changes. Regenerate or upload raw data through `scale_dataset.py`, `live_data_collector.py`, or Kaggle datasets.

## Data Percentage

The local raw directory currently supports curated files plus large generated caches such as:

- `data/raw/bulk_text_records.json`
- `data/raw/bulk_http_samples.json`

`bulk_text_records.json` can be tens of GiB. The trainer now reads only the selected percentage and stops after a valid JSON record boundary.

```bash
# Load curated files only, skip bulk caches.
python bugHunt/advanced_train.py --skip-collect --data-percent 0

# Load curated files plus 1 percent of both bulk caches.
python bugHunt/advanced_train.py --skip-collect --data-percent 1

# Load 5 percent of bulk text and all bulk HTTP samples.
python bugHunt/advanced_train.py --skip-collect --bulk-text-percent 5 --bulk-http-percent 100
```

For local CPU/GPU testing, start with `0` to `1`. For a stronger local GPU run, try `5` to `10`. Use `100` only when the machine has enough time, disk throughput, RAM, and GPU capacity.

## Local Setup

The requested local environment is `ml`. On this machine, activate it through Miniforge:

```bash
source /home/rafal/miniforge3/etc/profile.d/conda.sh
conda activate ml
```

Install or refresh dependencies:

```bash
pip install -r bugHunt/requirements.txt
```

Smoke check:

```bash
python -m py_compile bugHunt/advanced_train.py bugHunt/training_workflow.py bugHunt/inference.py bugHunt/live_data_collector.py bugHunt/scale_dataset.py
python bugHunt/advanced_train.py --skip-collect --dry-run --print-options --data-percent 0
```

## Training

Cached local training:

```bash
python bugHunt/advanced_train.py \
  --skip-collect \
  --training-profile min \
  --epochs 1 \
  --batch-size 8 \
  --data-percent 0.1 \
  --run-name local_smoke \
  --no-periodic-checkpoints
```

Fuller local or workstation training:

```bash
python bugHunt/advanced_train.py \
  --skip-collect \
  --training-profile default \
  --epochs 10 \
  --batch-size 16 \
  --data-percent 5 \
  --run-name local_cached_5pct \
  --update-default-best
```

Live collection plus training:

```bash
python bugHunt/advanced_train.py \
  --training-profile default \
  --nvd-per-kw 300 \
  --gh-max 2000 \
  --osv-per-eco 200 \
  --h1-pages 10 \
  --synth-n 15000 \
  --data-percent 1 \
  --run-name live_plus_cache
```

Reinforcement from the current best checkpoint:

```bash
python bugHunt/advanced_train.py \
  --skip-collect \
  --resume-latest-best \
  --epochs 5 \
  --data-percent 5 \
  --run-name reinforce_5pct \
  --update-default-best
```

Workflow wrapper for repeated rounds:

```bash
python bugHunt/training_workflow.py \
  --mode cycle \
  --collect cached \
  --rounds 3 \
  --data-percent 5 \
  --run-name-prefix cycle_cached
```

## Outputs

Training writes:

- `models/advanced_model_best_<run>.pt`
- `models/advanced_model_final.pt`
- `models/metadata.json`
- `logs/advanced_eval_<run>.json`
- `logs/dataset_manifest_<run>.json`
- `logs/advanced_metrics.jsonl`

Inference uses `models/advanced_model_best.pt` or `models/advanced_model_final.pt`:

```bash
python bugHunt/inference.py "SQL injection in login allows account takeover"
```

## Kaggle

Use [KAGGLE.md](KAGGLE.md) for Kaggle dataset upload, notebook setup, cached training, live data collection, and reinforcement training.
