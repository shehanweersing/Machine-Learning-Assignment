# Preprocessing and feature-engineering strategy

The reusable implementation is in `preprocessing.py` and is imported by
`student_performance_pipeline.py`. Preprocessing is fitted inside each
cross-validation fold through a scikit-learn `Pipeline`; therefore validation
statistics and category levels cannot leak from validation or test data.

## Data preparation

- The semicolon-delimited Math dataset is loaded with an explicit schema
  check for the `G3` target.
- Missing values, duplicate rows, numeric ranges, target distribution and
  zero-grade count are written to `outputs/*/results.json`.
- The dataset is split before any transformer is fitted (`80%` train and
  `20%` untouched test set, `random_state=42`).

## Feature policy

Two scenarios are supported:

- **End-of-term:** retains `G1` and `G2` and adds `grade_progress = G2 - G1`.
- **Early intervention:** excludes `G1`, `G2`, and `grade_progress`, avoiding
  an unrealistic claim that a prediction is available before those grades.

Ordinal survey responses are treated as categories rather than equally spaced
continuous measurements. `age`, `absences`, `G1`, `G2`, and the engineered
progress feature are scaled numerically when available. Nominal and ordinal
features are one-hot encoded with `handle_unknown="ignore"`.

## Model workflow

The pipeline compares a mean baseline, Ridge regression, Random Forest and
Extra Trees. Five-fold shuffled cross-validation is performed on training data
only. Grid search tunes the non-baseline models using the same training
partition; the untouched test set is used only for final RMSE, MAE and R2.
