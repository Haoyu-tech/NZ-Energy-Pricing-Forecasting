# Reproduction guide

## Environment and input modes

Install [requirements.txt](../requirements.txt), then edit [config.yaml](../config.yaml). `data.use_synthetic: true` selects the synthetic path without downloading EMI files; `false` selects the configured real-data source and requires network access on the first uncached run.

```powershell
python -m pip install -r requirements.txt
python run_pipeline.py --no-cv
```

Run from the root. For the full walk-forward workflow, run `python run_pipeline.py` without `--no-cv`. The quick run and full validation serve different purposes.

## Outputs

Processed data and monthly raw caches go under `02_data/`. Reports and charts go under `outputs/`, as described in [README](../README.md). Preserve a copy of historical report artifacts before rerunning if you need to compare against them.

## Comparison record

Record node, date range, target frequency, hold-out period, feature lags, exogenous inputs, CV windows and software versions. Compare the same input mode and period: synthetic output does not reproduce the real BEN2201 result table.

Timestamp and daylight-saving treatment should be checked against the raw source before drawing new empirical conclusions. No new EMI download or forecasting run was performed during cleanup.

## Run record

For every result you intend to compare or share, retain the repository commit, exact command/configuration, input-data identity, environment versions and output location. Save a dependency snapshot after a successful reference run; newly added requirements lists are intentionally unpinned until that validation exists.

## Validation status

Cleanup review: 14 October 2026. File paths and documented entry points were checked against the checkout. Full empirical results were not regenerated. Project-specific external prerequisites and known blockers are listed above.
