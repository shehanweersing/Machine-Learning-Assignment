[IT3091_Assignment.pdf](https://github.com/user-attachments/files/31911280/IT3091_Assignment_Descriptor_V1.pdf)

# Student Performance Prediction (Regression Lens)

## Overview
This project predicts a student's final mathematics grade (`G3`) from
demographic, social, lifestyle and academic attributes. The primary decision
is whether a teacher should review a student's expected final trajectory and
consider support; the model is not an automated grading or disciplinary tool.

## Dataset
* **Source:** Student Performance Dataset (`student-mat.csv`).
* **Scope:** Only the Math dataset is used: 395 students and 33 variables.
* **Target Variable:** `G3` (Final grade ranging from 0 to 20).
* **Features:** Includes weekly study time, past failures, family background, alcohol consumption, and term progression grades (`G1`, `G2`).







## Performance prediction (regression lens)

What it is: Predicting a student's numerical final grade after the second
period when `G1` and `G2` are available.

ML Task & Models: A mean baseline, Ridge regression, Random Forest, and Extra Trees are compared in a leakage-safe pipeline.

Evaluation Metrics: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and $R^2$ score.

Business Value: Provides a screening signal for teacher/counsellor review and
academic support planning. It does not establish causation or replace human
judgement.


| Model | CV RMSE mean | CV RMSE std | CV MAE mean | CV R² mean |
|---|---:|---:|---:|---:|
| Random Forest | 1.507 | 0.469 | 0.949 | 0.888 |
| Extra Trees | 1.637 | 0.426 | 1.085 | 0.869 |
| Ridge | 2.074 | 0.235 | 1.543 | 0.782 |
| Baseline (mean) | 4.563 | 0.634 | 3.403 | -0.035 |

After training-only grid search, Random Forest was selected with
`max_depth=10`, `min_samples_leaf=1`, and `n_estimators=300`. On the untouched
79-student test set it achieved RMSE **2.008**, MAE **1.208**, and R²
**0.803**. Thus, the average absolute error was about 1.21 grade points in
this split; it is not a guarantee for future cohorts.

The final model diagnostics are available in
[`feature_importance.csv`](student-performance-regression/outputs/with_grades/feature_importance.csv),
[`test_predictions_and_residuals.csv`](student-performance-regression/outputs/with_grades/test_predictions_and_residuals.csv),
and [`residual_plot.png`](student-performance-regression/outputs/with_grades/residual_plot.png).
The largest model-derived importance is assigned to `G2`, which is consistent
with the EDA correlation but is not evidence that the feature causes the
final grade. Feature importance is used for interpretation, not causal
decision-making.

The stricter early-intervention run, which excludes `G1` and `G2`, achieved
RMSE **4.021**, MAE **3.253**, and R² **0.211**. This large difference is why
the main claim is explicitly end-of-term rather than early prediction.

## Recommendation and limitations

Use the tuned Random Forest as a **screening aid after G2 is available**.
Students with predicted `G3 < 10` may be referred for teacher or counsellor
review, but the flag must be combined with context and a human conversation.
The model must not be used as the sole basis for grading, exclusion or
disciplinary action.

Limitations include the small historical sample, the 2005-2006 Portuguese
school context, lack of external validation, sensitivity to the train/test
split, and possible bias across demographic or family-background groups.
The observational data supports prediction, not causal claims such as
“more study time causes a higher grade”. `G1` and `G2` also make the model
unsuitable for a genuinely pre-assessment intervention.

## Evidence and reproducibility

See the [data understanding and EDA evidence](student-performance-regression/EDA/Data%20Understanding%20Table.md),
[problem framing canvas](Problem%20Framing%20Canvas.md),
[EDA insight log](student-performance-regression/EDA/EDA%20Insight%20Log.md),
[preprocessing strategy](student-performance-regression/src/preprocessing.md),
and [decision log](Decision%20Log%20Table.md). The generated model outputs
are in `student-performance-regression/outputs/`. Add the completed
[AI-use transparency statement](AI%20Use%20Transparency.md) and one completed
[personal learning journey](Personal%20Learning%20Journey%20Template.md) per
group member before submission.

The interactive lecturer-facing dashboard is in
[website/index.html](website/index.html). It includes the grade distribution,
correlation analysis, academic progression (`G1` to `G3`), missing-value audit,
before/after scaling comparison, outlier review, model comparison, prediction
quality and the responsible recommendation workflow.





### Requirements

Install the dependency ranges from [`requirements.txt`](requirements.txt):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Reproducible code workflow

The reusable end-to-end implementation is
[`student_performance_pipeline.py`](student-performance-regression/src/student_performance_pipeline.py).
Run it from the repository root:

```powershell
.\.venv\Scripts\python.exe student-performance-regression\src\student_performance_pipeline.py
```

This runs the end-of-term scenario, where `G1` and `G2` are available. To
evaluate the stricter early-intervention scenario without `G1` and `G2`:

```powershell
.\.venv\Scripts\python.exe student-performance-regression\src\student_performance_pipeline.py --exclude-grades --output-dir student-performance-regression\outputs\early_intervention
```

Each run writes the data audit and final hold-out metrics to `results.json`,
the cross-validated model comparison to `model_comparison.csv`, and tuning
results to `hyperparameter_tuning.csv`. The final recommendation is a
human-review screening flag, not an automatic academic decision.
