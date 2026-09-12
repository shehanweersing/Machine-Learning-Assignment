# Decision Points

| Decision Point | Options Considered | Selected Choice | Justification & Evidence |
|---|---|---|---|
| Dataset Selection | Math dataset vs. Portuguese dataset | **Math Dataset** | Higher variance and stronger behavioral correlations. |
| Term Grades Handling | Include vs. Drop G1, G2 | **Included** | Supports academic progression tracking. |
| Feature Encoding | Label Encoding vs. One-Hot Encoding | **One-Hot Encoding** | Prevents false ordinal relationships. |
| Model Selection | Linear Regression vs. Multi-Model | **Multi-Model Pipeline** | Compares linear and non-linear models. |
| Evaluation Metrics | Accuracy vs. RMSE, MAE, R² | **RMSE, MAE, R²** | Suitable for continuous grade prediction. |