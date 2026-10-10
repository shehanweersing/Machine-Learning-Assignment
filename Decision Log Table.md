# Decision log

| Decision | Options considered | Selected choice | Evidence / trade-off |
|---|---|---|---|
| Primary lens | Classification, regression, descriptive analysis | Regression | `G3` is a continuous 0-20 grade and RMSE/MAE have direct grade-point meaning |
| Dataset | Math, Portuguese, merged data | Math only | Keeps the decision focused on mathematics; selection should not be justified by unverified “higher variance” claims |
| Prediction timing | Before G1/G2, after G2 | Report both scenarios | End-of-term model uses G1/G2; early-intervention model excludes them to avoid an unrealistic timing claim |
| Target | G1, G2, G3 | G3 | G3 is the final outcome; G1/G2 are predictors only when available at prediction time |
| Feature engineering | Raw features only, add progression | Add `grade_progress = G2 - G1` in end-of-term scenario | Captures change between periods while remaining available after G2 |
| Feature typing | Treat all integer codes as continuous, encode ordinal levels | One-hot encode ordinal survey codes | Prevents assuming equal spacing between categories; the choice may reduce linear efficiency but is safer |
| Missing/duplicates | Impute/delete, audit only | Audit and report; no imputation needed | Dataset has 0 missing cells and 0 duplicate rows |
| Preprocessing | Fit before split, fit inside pipeline | Fit inside each CV fold | Prevents validation/test leakage |
| Validation | One split only, cross-validation | 80/20 hold-out plus shuffled 5-fold CV on training data | Gives both final generalisation estimate and model-selection evidence |
| Models | One model, several candidates | Mean baseline, Ridge, Random Forest, Extra Trees | Baseline establishes usefulness; linear and tree ensembles provide contrasting inductive biases |
| Tuning | Defaults, grid search | GridSearchCV on training data | Selected parameters are not influenced by the final test set |
| Metrics | Accuracy, RMSE only, multiple regression metrics | RMSE, MAE, R² | RMSE penalises large errors; MAE is interpretable in grade points; R² gives explained variance |
| Recommendation | Automatic intervention, human review | Human review screening only | Predictions must not be the sole basis for student decisions |
| Intervention profiling lens | Use only Math, merge both subjects, descriptive groups | Use matched Math/Portuguese records with transparent rules | Adds cross-subject support context; merge key is not a unique student ID, so results are records for human review rather than verified identities |
| Unsupervised clustering | K-Means only, hierarchical only, DBSCAN only, compare methods | Compare K-Means (`k=2..6`), Ward hierarchical (`k=2..6`) and DBSCAN; select highest valid silhouette | Adds an exploratory similarity lens; excludes G3 from clustering and keeps cluster labels descriptive rather than prescriptive |
| Cluster validation | Report one score, inspect only a PCA plot | Report silhouette scores, cluster sizes, feature heatmap, PCA projection and repeated-seed ARI | Makes separation, balance, interpretation and stability auditable rather than relying on one visual |
