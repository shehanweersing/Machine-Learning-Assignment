# Intervention Group Profiling Lens

## Purpose

The project now includes a second, descriptive lens alongside the regression
model. It uses the Mathematics and Portuguese files to profile matched records
into transparent support-priority groups. This answers:

- Which records show low outcomes in both subjects?
- Which records show a subject-specific support need?
- Which records have emerging signals such as previous failures or unusually
  high absences?
- Which records currently show stable outcomes?

This is **not** a second grading model. The profiles support teacher/counsellor
review and programme planning only. The lens now combines transparent
rule-based profiling with an unsupervised clustering comparison.

## Reproducible command

From the repository root:

```powershell
.\.venv\Scripts\python.exe intervention-group-profiling\intervention_group_profiling.py
```

The script writes its evidence to
`intervention-group-profiling/outputs/`:

- `student_intervention_profiles.csv` — row-level profile evidence
- `intervention_profile_summary.csv` — group counts and rates
- `intervention_profile_summary.json` — rules, quality audit, and limitations
- `intervention_profile_counts.png` — profile sizes
- `cross_subject_risk_quadrants.png` — Mathematics versus Portuguese outcomes
- `profile_grade_distributions.png` — grade distributions by profile
- `clustering_comparison.csv` — K-Means, hierarchical and DBSCAN scores
- `cluster_profile_summary.csv` — outcome and behaviour summaries by cluster
- `student_cluster_assignments.csv` — row-level cluster assignments
- `cluster_pca_visualization.png` — selected clustering solution in two PCA dimensions
- `clustering_algorithm_comparison.png` — silhouette scores for tested configurations
- `cluster_feature_heatmap.png` — relative feature patterns by cluster
- `cluster_sizes.png` — balance of the selected clusters
- `cluster_feature_profile.csv` — standardised cluster feature means
- `clustering_stability.json` — repeated-seed stability audit

## Merge and quality evidence

The workflow follows [`student-merge.R`](student-merge.R) and uses these shared attributes:
school, sex, age, address, family size, parental status and education,
parental jobs, reason, nursery and internet access. It produces 382 matched
records from 395 Mathematics records and 649 Portuguese records.

The key is not an explicit student ID. Four Mathematics records and twelve
Portuguese records have duplicate merge keys, so a matched record is **not
guaranteed to be a verified unique student**. This is reported as a limitation,
not hidden. A future study should use a genuine anonymised student identifier.

## Transparent profiling rules

The threshold `G3 < 10` is used only as a clearly labelled screening
threshold. The absence signal is subject-specific absences at or above that
subject's 75th percentile. Profiles are assigned in this order:

1. **Priority: low outcomes in both subjects** — both Mathematics and
   Portuguese `G3` are below 10.
2. **Priority: Mathematics-specific support** — Mathematics `G3` is below 10,
   but Portuguese `G3` is not.
3. **Priority: Portuguese-specific support** — Portuguese `G3` is below 10,
   but Mathematics `G3` is not.
4. **Monitor: emerging risk signals** — no low final outcome, but at least one
   of weak `G2`, previous failures, or the high-absence signal is present.
5. **Monitor: currently stable** — none of the above screening signals is
   present.

These rules describe patterns; they do not diagnose a learner or prove that
absence, failure history or any demographic attribute causes a grade.

## Clustering analysis

The profiling lens compares three unsupervised algorithms:

1. **K-Means** — tests `k=2` through `k=6` and groups records around centroids.
2. **Hierarchical clustering** — tests the same group counts using Ward linkage.
3. **DBSCAN** — tests density-based groups and can mark unusual records as
   noise (`-1`).

Clustering uses standardised behaviour, attendance, previous-failure and
academic-progress features from both subjects. Final grades `G3` are excluded
from clustering; they are used only afterwards to interpret what each cluster
looks like. The configuration with the highest valid silhouette score is
selected. In the current run, **K-Means with `k=2`** was selected with a
silhouette score of approximately **0.215**:

- Cluster 0: 241 records; mean Math `G3` 12.18; mean Portuguese `G3` 13.82
- Cluster 1: 141 records; mean Math `G3` 7.32; mean Portuguese `G3` 10.28

This result suggests a broad higher-support-needs cluster, but it is not a
student quality ranking. Cluster labels are arbitrary, the silhouette score is
modest, and clustering does not prove causes or intervention effects. The
rule-based profiles remain the clearer operational explanation; clusters are
additional exploratory evidence for planning.

### Additional validation

- **Cluster separation:** silhouette score is reported for every tested
  configuration, not only the selected result.
- **Balance:** selected cluster sizes are reported and visualised; very small
  groups would be treated cautiously.
- **Interpretability:** cluster feature means are standardised and shown in a
  heatmap. This explains behaviour patterns without ranking students.
- **Stability:** the selected K-Means solution was repeated with 10 random
  seeds. The current mean Adjusted Rand Index is approximately **0.989** and
  the minimum is approximately **0.979**, indicating stable assignments for
  this dataset and feature policy.
- **Reproducibility:** the seed, tested parameter ranges, feature list and
  selected configuration are saved in the generated JSON/CSV evidence.

## Responsible recommendation workflow

1. Use the profile as a queue for human review.
2. Confirm current context with the student, teacher and counsellor.
3. Offer proportionate support such as subject tutoring, attendance support,
   study planning or wellbeing referral.
4. Record the support offered and review progress later.
5. Never use a profile as the sole basis for grading, exclusion, discipline or
   access to opportunities.

## Limitations

The data is historical, cross-sectional and from one school context. The
matched records may include merge-key collisions. The threshold and percentile
rules are project screening choices, not official policy. Small group sizes can
make averages unstable. Sensitive attributes should be used to audit fairness,
not to deny support. Before operational use, the lens would need a validated
student identifier, current data, stakeholder approval, fairness checks and
external validation.
