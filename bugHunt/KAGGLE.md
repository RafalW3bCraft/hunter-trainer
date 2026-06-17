# Kaggle Training Guide

This guide runs the consolidated BugHunt trainer on Kaggle with uploaded raw data, optional live data, and reinforcement rounds.

## 1. Prepare Data Locally

The trainer expects raw JSON files like:

- `bulk_text_records.json`
- `bulk_http_samples.json`
- `advanced_http_samples.json`
- `nvd_cves.json`
- `github_advisories.json`
- `cisa_kev.json`
- `osv_vulns.json`
- `patt_payloads.json`
- `unified_dataset.json`
- `scale_summary.json`

If you need to generate a raw bulk dataset before upload:

```bash
source /home/rafal/miniforge3/etc/profile.d/conda.sh
conda activate ml
python bugHunt/scale_dataset.py --target-gb 20.5 --text-ratio 0.98
```

Upload the raw data directory as a Kaggle Dataset. Keep the JSON files at the dataset root or inside one `raw/` folder.

## 2. Create Kaggle Notebook

1. Create a new Kaggle notebook.
2. Add the uploaded raw-data dataset as notebook input.
3. Enable a GPU accelerator in notebook settings.
4. Upload or clone this project into the notebook working directory.

Expected Kaggle paths:

```bash
/kaggle/input/<your-dataset-name>/
/kaggle/working/
```

If the dataset files are inside a `raw/` folder, use that folder as `BUGHUNT_RAW_DIR`.

## 3. Install Dependencies

In a Kaggle notebook cell:

```bash
cd /kaggle/working/hunter-trainer
pip install -r bugHunt/requirements.txt
```

If Kaggle already has compatible packages, this may finish quickly.

## 4. Configure Paths

Set output paths to Kaggle working storage:

```bash
export BUGHUNT_BASE_DIR=/kaggle/working/hunter-trainer
export BUGHUNT_RAW_DIR=/kaggle/input/<your-dataset-name>
export BUGHUNT_MODELS_DIR=/kaggle/working/hunter-trainer/models
export BUGHUNT_LOGS_DIR=/kaggle/working/hunter-trainer/logs
export BUGHUNT_CHECKPOINTS_DIR=/kaggle/working/hunter-trainer/logs/checkpoints
```

If your uploaded dataset has a nested raw folder:

```bash
export BUGHUNT_RAW_DIR=/kaggle/input/<your-dataset-name>/raw
```

## 5. Dry Run

Always verify options and paths first:

```bash
python bugHunt/advanced_train.py \
  --skip-collect \
  --dry-run \
  --print-options \
  --data-percent 1
```

The output should show the Kaggle raw-data path and the selected bulk text/HTTP percentages.

## 6. Train From Uploaded Raw Data

Start with a bounded percentage. The 21 GiB text cache should not be streamed in full unless you intentionally choose `100`.

```bash
python bugHunt/advanced_train.py \
  --skip-collect \
  --training-profile default \
  --epochs 10 \
  --batch-size 16 \
  --data-percent 5 \
  --run-name kaggle_cached_5pct \
  --update-default-best \
  --no-periodic-checkpoints
```

For a stronger run after the notebook is stable:

```bash
python bugHunt/advanced_train.py \
  --skip-collect \
  --training-profile max \
  --epochs 30 \
  --batch-size 32 \
  --bulk-text-percent 10 \
  --bulk-http-percent 100 \
  --run-name kaggle_max_10pct \
  --update-default-best \
  --no-periodic-checkpoints
```

## 7. Add Live Data

Live collection can refresh NVD, GitHub Advisory, OSV, CISA KEV, HackerOne public reports, and generated HTTP samples before training. Use smaller fetch counts on Kaggle first:

```bash
python bugHunt/advanced_train.py \
  --training-profile default \
  --nvd-per-kw 100 \
  --gh-max 1000 \
  --osv-per-eco 100 \
  --h1-pages 3 \
  --synth-n 5000 \
  --data-percent 1 \
  --run-name kaggle_live_refresh \
  --update-default-best
```

Optional API keys can reduce rate-limit friction:

```bash
python bugHunt/advanced_train.py \
  --github-token "$GITHUB_TOKEN" \
  --nvd-api-key "$NVD_API_KEY" \
  --training-profile default \
  --data-percent 1 \
  --run-name kaggle_live_with_keys
```

## 8. Reinforcement Training

After one successful promoted run creates `models/advanced_model_best.pt`, reinforce from it:

```bash
python bugHunt/advanced_train.py \
  --skip-collect \
  --resume-latest-best \
  --epochs 5 \
  --data-percent 5 \
  --run-name kaggle_reinforce_5pct \
  --update-default-best \
  --no-periodic-checkpoints
```

For repeated fresh-plus-reinforcement cycles:

```bash
python bugHunt/training_workflow.py \
  --mode cycle \
  --collect cached \
  --rounds 3 \
  --data-percent 5 \
  --run-name-prefix kaggle_cycle \
  --no-periodic-checkpoints
```

## 9. Save Artifacts

Download or persist these outputs after training:

- `models/advanced_model_best.pt`
- `models/advanced_model_final.pt`
- `models/metadata.json`
- `models/tokenizer_config.json`
- `logs/advanced_eval_<run>.json`
- `logs/dataset_manifest_<run>.json`
- `logs/advanced_metrics.jsonl`

Kaggle working storage is temporary. Save the model files as notebook outputs or upload them to a Kaggle Dataset before ending the session.

## 10. Inference Check

Run a quick prediction before saving artifacts:

```bash
python bugHunt/inference.py "SSRF reaches cloud metadata and exposes credentials"
```

If inference loads the checkpoint and returns labels, the training artifacts are usable outside Kaggle.
