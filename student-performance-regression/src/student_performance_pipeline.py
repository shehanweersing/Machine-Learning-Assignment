"""End-to-end, reproducible student-grade regression workflow.

The workflow deliberately separates:
1. data preparation and audit,
2. train/test splitting,
3. fold-safe preprocessing and feature engineering,
4. model comparison,
5. nested hyperparameter selection on training data,
6. final hold-out evaluation, and
7. a cautious intervention recommendation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_validate,
    train_test_split,
)

from preprocessing import build_preprocessing_pipeline, prepare_features


RANDOM_STATE = 42
TEST_SIZE = 0.2
N_JOBS = -1


def load_and_audit(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the semicolon-delimited UCI file and produce a quality audit."""

    frame = pd.read_csv(path, sep=";")
    if frame.empty:
        raise ValueError(f"Dataset is empty: {path}")

    numeric_ranges = {
        column: {
            "min": float(frame[column].min()),
            "max": float(frame[column].max()),
        }
        for column in frame.select_dtypes(include="number").columns
    }
    audit = {
        "source": str(path),
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "missing_values": int(frame.isna().sum().sum()),
        "missing_by_column": frame.isna().sum().astype(int).to_dict(),
        "duplicate_rows": int(frame.duplicated().sum()),
        "numeric_ranges": numeric_ranges,
        "target_distribution": frame["G3"].describe().round(4).to_dict(),
        "zero_final_grades": int((frame["G3"] == 0).sum()),
    }
    return frame, audit


def get_models() -> dict[str, Any]:
    """Return a baseline and three distinct model families."""

    return {
        "Baseline (mean)": DummyRegressor(strategy="mean"),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),
    }


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    include_grades: bool,
) -> pd.DataFrame:
    """Compare models using identical training-only five-fold CV."""

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in get_models().items():
        pipeline = build_preprocessing_pipeline(
            X_train, model, include_grades=include_grades
        )
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
            },
            n_jobs=1,
            return_train_score=False,
        )
        rows.append(
            {
                "Model": name,
                "CV RMSE mean": -scores["test_rmse"].mean(),
                "CV RMSE std": scores["test_rmse"].std(),
                "CV MAE mean": -scores["test_mae"].mean(),
                "CV R2 mean": scores["test_r2"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("CV RMSE mean").reset_index(drop=True)


def tune_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    include_grades: bool,
) -> tuple[str, Any, pd.DataFrame]:
    """Tune non-baseline models with an inner CV on the training partition."""

    candidates = {
        "Ridge": (
            Ridge(),
            {"model__alpha": [0.1, 1.0, 10.0, 50.0]},
        ),
        "Random Forest": (
            RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
            ),
            {
                "model__n_estimators": [300, 500],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_leaf": [1, 2],
            },
        ),
        "Extra Trees": (
            ExtraTreesRegressor(
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
            ),
            {
                "model__n_estimators": [300, 500],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_leaf": [1, 2],
            },
        ),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    searches: list[dict[str, Any]] = []
    best_name = ""
    best_search = None
    for name, (model, grid) in candidates.items():
        pipeline = build_preprocessing_pipeline(
            X_train, model, include_grades=include_grades
        )
        search = GridSearchCV(
            pipeline,
            grid,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=1,
            refit=True,
            return_train_score=False,
        )
        search.fit(X_train, y_train)
        searches.append(
            {
                "Model": name,
                "Best CV RMSE": -search.best_score_,
                "Best parameters": json.dumps(search.best_params_, default=str),
            }
        )
        if best_search is None or search.best_score_ > best_search.best_score_:
            best_name = name
            best_search = search

    if best_search is None:
        raise RuntimeError("Hyperparameter tuning did not produce a model.")
    return best_name, best_search, pd.DataFrame(searches).sort_values(
        "Best CV RMSE"
    )


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Evaluate only once on the untouched hold-out test set."""

    predictions = np.asarray(model.predict(X_test), dtype=float)
    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "R2": float(r2_score(y_test, predictions)),
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
    }
    return metrics


def save_model_diagnostics(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> None:
    """Persist residual and feature-importance evidence for the final model."""

    predictions = np.asarray(model.predict(X_test), dtype=float)
    residuals = np.asarray(y_test, dtype=float) - predictions
    pd.DataFrame(
        {
            "actual_G3": np.asarray(y_test, dtype=float),
            "predicted_G3": predictions,
            "residual": residuals,
        }
    ).to_csv(output_dir / "test_predictions_and_residuals.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.scatter(predictions, residuals, alpha=0.7, color="purple")
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted G3")
    plt.ylabel("Residual (actual - predicted)")
    plt.title("Final-model residuals on the untouched test set")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_plot.png", dpi=160)
    plt.close()

    fitted_model = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    if hasattr(fitted_model, "feature_importances_"):
        importance = fitted_model.feature_importances_
    elif hasattr(fitted_model, "coef_"):
        importance = np.abs(np.asarray(fitted_model.coef_))
    else:
        return
    pd.DataFrame(
        {"feature": feature_names, "importance": importance}
    ).sort_values("importance", ascending=False).to_csv(
        output_dir / "feature_importance.csv", index=False
    )


def make_recommendation(
    model,
    X_test: pd.DataFrame,
    *,
    intervention_grade: float = 10.0,
) -> dict[str, Any]:
    """Create a screening recommendation, not an automatic student decision."""

    predictions = np.asarray(model.predict(X_test), dtype=float)
    flagged = predictions < intervention_grade
    return {
        "intervention_grade": intervention_grade,
        "students_screened": int(len(predictions)),
        "students_flagged_for_human_review": int(flagged.sum()),
        "flag_rate": float(flagged.mean()),
        "action": (
            "Refer flagged students to a teacher or counsellor for human review; "
            "do not use predictions as the sole basis for academic decisions."
        ),
    }


def predict_student(
    model,
    student: dict[str, Any],
    *,
    include_grades: bool,
) -> float:
    """Predict one student's G3 after applying the same feature policy."""

    row = pd.DataFrame([student])
    if include_grades:
        if not {"G1", "G2"}.issubset(row.columns):
            raise ValueError("G1 and G2 are required when include_grades=True.")
        row["grade_progress"] = row["G2"] - row["G1"]
    return float(model.predict(row)[0])


def run_workflow(
    data_path: Path,
    output_dir: Path,
    *,
    include_grades: bool,
    student: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the complete workflow and persist auditable result tables."""

    output_dir.mkdir(parents=True, exist_ok=True)
    frame, audit = load_and_audit(data_path)
    X, y = prepare_features(frame, include_grades=include_grades)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    comparison = compare_models(
        X_train, y_train, include_grades=include_grades
    )
    best_name, best_search, tuning = tune_models(
        X_train, y_train, include_grades=include_grades
    )
    evaluation = evaluate_model(best_search.best_estimator_, X_test, y_test)
    save_model_diagnostics(
        best_search.best_estimator_, X_test, y_test, output_dir
    )
    recommendation = make_recommendation(
        best_search.best_estimator_, X_test
    )
    individual_prediction = None
    if student is not None:
        individual_prediction = {
            "predicted_G3": predict_student(
                best_search.best_estimator_,
                student,
                include_grades=include_grades,
            )
        }

    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    tuning.to_csv(output_dir / "hyperparameter_tuning.csv", index=False)
    results = {
        "configuration": {
            "include_grades": include_grades,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "cv_folds": 5,
            "selected_model": best_name,
            "best_parameters": best_search.best_params_,
        },
        "audit": audit,
        "split": {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "evaluation": evaluation,
        "recommendation": recommendation,
        "individual_prediction": individual_prediction,
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=str),
        encoding="utf-8",
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parents[1] / "data" / "student-mat.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parents[1] / "outputs",
    )
    parser.add_argument(
        "--exclude-grades",
        action="store_true",
        help="Run the early-intervention scenario without G1 and G2.",
    )
    parser.add_argument(
        "--student-json",
        type=Path,
        help="Optional JSON file containing one student's feature values.",
    )
    args = parser.parse_args()
    student = None
    if args.student_json is not None:
        student = json.loads(args.student_json.read_text(encoding="utf-8"))
    results = run_workflow(
        args.data,
        args.output_dir,
        include_grades=not args.exclude_grades,
        student=student,
    )
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
