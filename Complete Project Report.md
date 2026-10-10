# Student Performance Machine Learning Project Report

## EXECUTIVE SUMMARY

This project uses the UCI Student Performance dataset to support two related
but deliberately separate analytical lenses:

1. **Primary lens - supervised regression:** estimate a student's final
   Mathematics grade (`G3`) after the second grading period, when `G1` and `G2`
   are available.
2. **Secondary lens - unsupervised intervention profiling:** discover broad
   similarity groups and cross-subject support patterns using matched
   Mathematics and Portuguese records.

The primary dataset for the regression model is `student-mat.csv`, containing
395 students and 33 variables. The target is `G3`, a final Mathematics grade
from 0 to 20. The selected Random Forest model achieved test RMSE **2.008**,
MAE **1.208**, and R2 **0.803** on the untouched 79-student test set.

The secondary lens uses 382 matched records produced from the Mathematics and
Portuguese files using the key documented in `student-merge.R`. It combines
transparent rule-based profiles with a clustering comparison. K-Means with
`k=2` was the best tested configuration by silhouette score (**0.215**).
Because the merge key is not a unique student identifier, all merged outputs
are treated as matched records for exploratory human review, not verified
student identities.

Neither lens is an automated grading, exclusion, disciplinary or diagnostic
system. Both are decision-support evidence for teacher/counsellor review.

---

## Primary Lens

### Objective

Answer the question:

> Given information available after the second grading period, what final
> Mathematics grade is plausible for a student?

The output is a numerical prediction of `G3`. This is explicitly an
**after-second-period prediction** task, not a pre-assessment prediction,
because `G1` and `G2` are used in the main model.

### Dataset scope

| Item | Project evidence |
|---|---|
| Source | UCI Student Performance dataset |
| Main file | `student-performance-regression/data/student-mat.csv` |
| Rows / columns | 395 / 33 |
| Unit of analysis | One student enrolled in Mathematics |
| Target | `G3`, final Mathematics grade, 0-20 |
| Missing cells | 0 |
| Duplicate rows | 0 |
| Zero final grades | 38 |
| SHA-256 | `E47F9EE225E1EE6E69B7564E6DAC7123E80B8486677FE111F351964CEF5DEC80` |

### Primary business question

Can a school use a model as a screening signal to decide which students may
benefit from a contextual academic-support conversation after the second
grading period?

The model does not determine a final grade, prove why a student performs in a
certain way, or decide whether a student should be punished or excluded.

---

## Secondary Lens

### Objective

The secondary lens asks:

> What broad cross-subject outcome and behaviour patterns can help educators
> prioritise human-led support review?

It is not a second grade-prediction model. It contains:

- transparent rule-based intervention profiles;
- K-Means clustering;
- Ward hierarchical clustering;
- DBSCAN density clustering;
- cluster interpretation and stability checks.

### Merge evidence and limitation

`student-merge.R` reads `student-mat.csv` and `student-por.csv`, then performs
an inner merge using shared demographic and background fields:

`school`, `sex`, `age`, `address`, `famsize`, `Pstatus`, `Medu`, `Fedu`,
`Mjob`, `Fjob`, `reason`, `nursery`, and `internet`.

The merge produces **382 matched records** from 395 Mathematics records and
649 Portuguese records. However, the key is not an explicit anonymised
student ID. Four Mathematics records and twelve Portuguese records contain
duplicate merge keys. Therefore, a matched record must not be described as a
verified unique student.

### Rule-based intervention profiles

Using the transparent screening threshold `G3 < 10`, the current run produced:

| Profile | Records | Interpretation |
|---|---:|---|
| Priority: low outcomes in both subjects | 23 | Both Mathematics and Portuguese final grades are below 10 |
| Priority: Mathematics-specific support | 104 | Mathematics is below 10 while Portuguese is not |
| Priority: Portuguese-specific support | 9 | Portuguese is below 10 while Mathematics is not |
| Monitor: emerging risk signals | 107 | No low final outcome, but weak progression, failures or high absence signal |
| Monitor: currently stable | 139 | No configured screening signal is present |

These are support-review queues, not student labels or diagnoses.

### Clustering analysis

Clustering uses standardised age, study time, travel time, absences, previous
failures and `G1`/`G2` progression features from both subjects. `G3` is
excluded from cluster formation and is used only afterwards to interpret the
groups.

| Algorithm | Configurations tested | Purpose |
|---|---|---|
| K-Means | `k=2` through `k=6` | Centroid-based similarity groups |
| Hierarchical | Ward linkage, `k=2` through `k=6` | Nested similarity structure |
| DBSCAN | Multiple `eps` values, `min_samples=8` | Density groups and noise records |

The selected result is **K-Means, `k=2`**, with silhouette score **0.215**:

| Cluster | Records | Mean Math G3 | Mean Portuguese G3 | Interpretation evidence |
|---:|---:|---:|---:|---|
| 0 | 241 | 12.18 | 13.82 | Lower average absence/failure signals |
| 1 | 141 | 7.32 | 10.28 | Higher average absence/failure signals and lower outcomes |

Cluster labels are arbitrary identifiers. Cluster 1 is not a “bad student”
category; it is an exploratory similarity group that should trigger context
checking, not an automatic decision.

The selected K-Means solution was repeated with 10 random seeds. The mean
Adjusted Rand Index was approximately **0.989** and the minimum was
approximately **0.979**, indicating stable assignments for this dataset and
feature policy. This stability does not prove causal validity or future
generalisability.

---

## PROBLEM FRAMING CANVAS

| Item | Primary regression lens | Secondary intervention/clustering lens |
|---|---|---|
| Stakeholder | Mathematics teachers and school counsellors | Teachers, counsellors and academic-support planners |
| Decision need | Which students may need contextual review after G2? | Which matched records show similar support-relevant patterns? |
| Unit of analysis | One Mathematics-course student | One matched Mathematics/Portuguese record |
| Inputs | Demographic, family, social, lifestyle, failure, absence, G1 and G2 data | Matched subject records, attendance, failures, study/travel time, G1/G2 |
| Output | Predicted numerical Mathematics `G3` | Rule-based profile and exploratory cluster membership |
| Success evidence | Low hold-out RMSE/MAE, reasonable R2, training-only CV | Silhouette comparison, cluster balance, feature interpretation, stability |
| Operational use | Human-review screening after G2 | Human-led support prioritisation and programme planning |
| Main trade-off | G1/G2 improve prediction but prevent a genuinely early claim | Cross-subject context is useful, but the merge key is not a unique ID |

### Scope boundaries

The primary lens does not prove causes or intervention effects. The secondary
lens does not diagnose students, rank student quality, or prove that a cluster
will respond to a particular intervention. Both outputs require current
context and human judgement.

---

## Business Context & Stakeholder

The project is designed for a school-support context where educators need
evidence to decide where to start a conversation. A prediction or cluster
profile can help organise attention, but the final action should involve:

1. reviewing the evidence and its limitations;
2. speaking with the student and relevant staff;
3. checking current attendance, wellbeing and learning context;
4. offering proportionate support such as tutoring, study planning, attendance
   support or counselling;
5. recording support and reviewing progress later.

Sensitive demographic variables must not be used as automatic exclusion,
discipline or opportunity-denial rules.

---

## WORKFLOW DIAGRAM

```text
UCI source files
  |
  +--> student-mat.csv ------------------------------+
  |                                                   |
  |   audit -> split -> preprocessing pipeline         |
  |                         |                          |
  |       CV model comparison -> training-only tuning  |
  |                         |                          |
  |             untouched test evaluation              |
  |                         |                          |
  |        predicted G3 -> human-review recommendation |
  |
  +--> student-por.csv + student-merge.R
                              |
                    matched-record quality audit
                              |
             +----------------+----------------+
             |                                 |
   rule-based support profiles       standardise pre-outcome features
                                               |
                         K-Means / Hierarchical / DBSCAN
                                               |
               silhouette + balance + heatmap + stability validation
                                               |
                    exploratory human-led intervention planning
```

---

## DATA DICTIONARY

### Primary dataset variables

| Variables | Meaning and treatment |
|---|---|
| `school`, `sex`, `address`, `famsize`, `Pstatus` | School, sex, urban/rural address, family size and parental status |
| `Mjob`, `Fjob`, `reason`, `guardian` | Family and school-choice context |
| `schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`, `romantic` | Binary categorical support, lifestyle and intention variables |
| `age` | Numeric age |
| `Medu`, `Fedu` | Ordinal parental education levels |
| `traveltime`, `studytime` | Ordinal travel and weekly study-time categories |
| `failures` | Previous class failures; 3 represents three or more |
| `famrel`, `freetime`, `goout`, `Dalc`, `Walc`, `health` | Ordinal family, lifestyle and health variables |
| `absences` | Absence count |
| `G1`, `G2` | First- and second-period grades |
| `G3` | Final Mathematics grade and primary target |

### Secondary lens variables

The merged file contains subject-suffixed versions such as `G1_math`,
`G1_portuguese`, `G2_math`, `G2_portuguese`, `G3_math`, `G3_portuguese`,
`failures_math`, `failures_portuguese`, `absences_math` and
`absences_portuguese`.

The clustering feature policy excludes `G3_math` and `G3_portuguese` from
cluster formation. They are used only for post-clustering interpretation.

---

## EDA INSIGHT LOG

### Primary lens findings

| Finding | Evidence | Modelling impact |
|---|---:|---|
| Target has useful variation | Mean 10.415, SD 4.581, range 0-20 | Regression is appropriate |
| `G2` is strongest | Correlation with `G3` = 0.905 | Main model is explicitly after G2 |
| `G1` is strong | Correlation with `G3` = 0.801 | Retain when available at prediction time |
| Previous failures are negatively associated | Correlation = -0.360 | Include as a signal, not a causal rule |
| Study-time association is weak/non-monotonic | Group means 10.048 to 11.400 | Avoid claiming study time alone determines results |
| Absence association is weak | Correlation = 0.034 | Do not use absence as a standalone rule |
| Zero grades exist | 38 observations | Retain and investigate context; do not silently delete |

### Secondary lens findings

- 23 matched records fall below 10 in both final-subject outcomes.
- 104 records show a Mathematics-specific low-outcome pattern.
- The selected lower-outcome cluster has higher average failures and absences,
  but this is an association within historical data, not a cause.
- Rule-based profiles are easier to explain operationally than arbitrary
  cluster labels.
- Clustering adds exploratory similarity evidence rather than replacing the
  transparent support rules.

---

## Data Quality Issues

### Primary dataset

- 0 missing cells;
- 0 duplicate rows;
- 38 zero final grades;
- absence outliers were retained because they are plausible observations;
- data is historical and from a 2005-2006 Portuguese school context.

### Secondary merged records

- 382 inner-join matches;
- merge key is not a unique student ID;
- 4 Mathematics rows and 12 Portuguese rows have duplicate merge keys;
- matched counts may therefore include key collisions;
- a future operational dataset should use a genuine anonymised identifier.

---

## Temporal Patterns

The dataset is cross-sectional, so it does not support a calendar time-series
claim. The valid ordered academic view is:

`G1 -> G2 -> G3`

This progression is used in EDA and feature engineering. The primary model
predicts after G2. The early-intervention sensitivity run excludes `G1`, `G2`
and `grade_progress`, and achieved RMSE **4.021**, MAE **3.253** and R2
**0.211**, showing why it must be reported as a materially weaker separate
scenario.

---

## Outlier Observations

Absence outliers were audited using the IQR rule and retained. Removing
plausible students without domain evidence could bias the model and profiles.
The 38 zero final grades were also retained and reported rather than silently
treated as data errors.

---

## EDA Visualizations

Primary EDA outputs are generated by
`student-performance-regression/EDA/eda_analysis.py`:

- `g3_distribution.png`
- `correlation_heatmap.png`
- `g3_by_failures.png`
- `g2_vs_g3.png`
- `academic_progression.png`
- `missing_values_analysis.png`
- `feature_scaling_comparison.png`
- `top_correlations.png`
- `absences_outlier_analysis.png`

Secondary lens and clustering outputs are generated by
`intervention-group-profiling/intervention_group_profiling.py`:

- `intervention_profile_counts.png`
- `cross_subject_risk_quadrants.png`
- `profile_grade_distributions.png`
- `cluster_pca_visualization.png`
- `clustering_algorithm_comparison.png`
- `cluster_feature_heatmap.png`
- `cluster_sizes.png`

---

## PREPROCESSING & FEATURE LOG

### Primary lens

1. Load the semicolon-delimited file with an explicit `G3` check.
2. Audit missing values, duplicates, numeric ranges and target distribution.
3. Split before fitting transformations: 80% train and 20% untouched test,
   `random_state=42`.
4. Fit preprocessing inside each cross-validation fold through a
   scikit-learn `Pipeline`.
5. One-hot encode nominal and ordinal categorical variables with
   `handle_unknown="ignore"`.
6. Scale numeric features inside the pipeline.
7. Compare models and tune only on the training partition.

### Secondary lens

1. Read both subject files with the semicolon delimiter.
2. Apply the documented inner merge and audit duplicate merge keys.
3. Create transparent low-outcome, failure and absence signals.
4. Standardise clustering features.
5. Fit and compare K-Means, hierarchical and DBSCAN configurations.
6. Keep `G3` out of cluster formation.
7. Interpret clusters with grades and behaviour only after assignments exist.

### Data Cleaning Steps

No imputation or duplicate-row deletion was needed for the primary Math file
because it contains no missing cells and no duplicate rows. No plausible
absence observations were deleted. For the secondary lens, merge-key
non-uniqueness is reported rather than hidden or incorrectly “fixed.”

### Feature Engineering

For the primary end-of-term model:

`grade_progress = G2 - G1`

The early-intervention sensitivity run excludes `G1`, `G2` and
`grade_progress`. For the secondary lens, subject-specific suffixes, grade
gaps, low-outcome flags, failure signals and high-absence signals are used for
interpretation and profiling.

### Feature Scaling

Numeric features are scaled inside the primary model pipeline and before
clustering. This is important for distance-based algorithms because absences,
grades and ordinal-coded variables have different raw scales. The raw source
data is not overwritten.

---

## MODEL/METHOD COMPARISON

### Primary supervised models

- Mean baseline;
- Ridge regression;
- Random Forest;
- Extra Trees.

All candidates use the same training-only cross-validation design.

### Evaluation Metrics

- **RMSE:** penalises larger grade errors more strongly.
- **MAE:** average absolute error in interpretable grade points.
- **R2:** proportion of target variance explained relative to the mean
  baseline.

### Model Comparison Results

| Model | CV RMSE mean | CV RMSE SD | CV MAE mean | CV R2 mean |
|---|---:|---:|---:|---:|
| Random Forest | 1.507 | 0.469 | 0.949 | 0.888 |
| Extra Trees | 1.637 | 0.426 | 1.085 | 0.869 |
| Ridge | 2.074 | 0.235 | 1.542 | 0.782 |
| Mean baseline | 4.563 | 0.634 | 3.403 | -0.035 |

Random Forest was selected because it achieved the lowest cross-validated
RMSE. After training-only tuning, the selected parameters were
`max_depth=10`, `min_samples_leaf=1`, and `n_estimators=300`.

### Secondary unsupervised comparison

The secondary lens uses silhouette score for model-selection evidence. The
comparison is exploratory, because unsupervised clusters do not have a
ground-truth target. K-Means `k=2` achieved the strongest valid score in the
current run.

---

## EVALUATION, VALIDATION & CRITICAL JUDGEMENT

### Primary hold-out evaluation

The final Random Forest was evaluated once on the untouched 79-student test
set:

| Metric | Result |
|---|---:|
| RMSE | 2.008 |
| MAE | 1.208 |
| R2 | 0.803 |

The MAE means that the average absolute error in this split was about 1.21
grade points. It is not a guarantee for future cohorts.

Residual plots, actual-versus-predicted evidence and feature importance are
used for diagnosis. Feature importance describes model behaviour; it does not
prove that changing a feature will cause a grade change.

### Validation decisions

- final hold-out data is not used for tuning;
- five-fold shuffled CV is performed only on training data;
- preprocessing is fitted within folds;
- the early-intervention scenario is reported separately;
- the clustering lens reports separation, balance, feature patterns and
  repeated-seed stability;
- merge-key uncertainty is explicitly disclosed.

### Critical judgement

The strongest model result depends heavily on `G1` and `G2`, especially `G2`.
Therefore, the headline model is useful after the second period but unsuitable
for a genuinely pre-assessment intervention. The early-intervention sensitivity
run is materially weaker. The clustering silhouette score is modest, so the
clusters are useful as exploratory context rather than a definitive student
taxonomy. Historical single-school data, merge-key collisions and lack of
external validation limit deployment claims.

### Notebook Files

The repository includes:

- `student-performance-regression/EDA/exploratory_data_analysis.ipynb`
- `student-performance-regression/src/train_models.ipynb`
- `student-performance-regression/src/evaluate_models.ipynb`

The reusable scripts are the canonical reproducible implementation:

- `student-performance-regression/EDA/eda_analysis.py`
- `student-performance-regression/src/preprocessing.py`
- `student-performance-regression/src/student_performance_pipeline.py`
- `intervention-group-profiling/intervention_group_profiling.py`

---

## RECOMMENDATION, LIMITATIONS & STAKEHOLDER VALUE

### Recommendations from the primary lens

After G2 is available, use predicted `G3 < 10` as a possible teacher/counsellor
review trigger. Combine the signal with current context and conversation. Do
not use it as the sole basis for grading, exclusion, discipline or access to
opportunities.

### Recommendations from the secondary lens

Use the rule-based profiles and exploratory clusters to organise support
conversations:

- both-subject low outcomes may justify broader coordinated review;
- Mathematics-specific patterns may suggest Mathematics tutoring;
- Portuguese-specific patterns may suggest Portuguese support;
- emerging signals may justify checking attendance, failures, progression and
  wellbeing;
- stable profiles should not be assumed to need no support without context.

These are possible support pathways, not automatic interventions.

### Stakeholder value

The project provides:

- an interpretable numerical prediction after G2;
- a transparent cross-subject support-profiling lens;
- reproducible EDA and data-quality evidence;
- baseline and alternative model comparison;
- clustering algorithm comparison;
- hold-out evaluation and stability diagnostics;
- a lecturer-facing dashboard with separate prediction and intervention pages;
- explicit ethical, timing and data-quality boundaries.

### Limitations

- The Math dataset is small and historical.
- The source context is one Portuguese school setting from 2005-2006.
- There is no external cohort validation.
- Results are sensitive to the sample and split.
- `G1` and `G2` make the primary model an after-G2 model.
- Observational associations are not causal effects.
- The secondary merge key is not a unique student ID.
- The clustering silhouette score is modest.
- Cluster labels are arbitrary and not rankings.
- Thresholds such as `G3 < 10` and the 75th-percentile absence signal are
  project screening choices, not official policy.
- Fairness, current-cohort performance and intervention effectiveness would
  need further validation before operational deployment.

### Reproducibility commands

From the repository root:

```powershell
.\.venv\Scripts\python.exe student-performance-regression\src\student_performance_pipeline.py
.\.venv\Scripts\python.exe student-performance-regression\src\student_performance_pipeline.py --exclude-grades --output-dir student-performance-regression\outputs\early_intervention
.\.venv\Scripts\python.exe intervention-group-profiling\intervention_group_profiling.py
```

The corresponding lecturer-facing pages are:

- `website/index.html` - Prediction Lens
- `website/intervention.html` - Intervention Group Profiling and Clustering Lens

