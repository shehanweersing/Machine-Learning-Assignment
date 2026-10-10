"""Leakage-safe preprocessing for the student performance project."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ORDINAL_COLUMNS = (
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
)

def get_preprocessor(
    X: pd.DataFrame,
    *,
    include_grades: bool = True,
    extra_categorical_columns: Iterable[str] = (),
) -> ColumnTransformer:
    """Build a transformer using only training-fold information.

    G1 and G2 are retained only for the end-of-term prediction scenario.
    Ordinal survey responses are one-hot encoded rather than treated as
    equally spaced continuous values.
    """

    excluded = {"G3"}
    if not include_grades:
        excluded.update({"G1", "G2", "grade_progress"})

    available = [column for column in X.columns if column not in excluded]
    categorical_columns = [
        column
        for column in available
        if X[column].dtype == "object" or str(X[column].dtype) == "string"
    ]
    categorical_columns.extend(
        column
        for column in ORDINAL_COLUMNS
        if column in available and column not in categorical_columns
    )
    categorical_columns.extend(
        column
        for column in extra_categorical_columns
        if column in available and column not in categorical_columns
    )
    categorical_columns = list(dict.fromkeys(categorical_columns))

    numeric_columns = [
        column
        for column in available
        if column not in categorical_columns
    ]

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", StandardScaler(), numeric_columns))
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_features(
    frame: pd.DataFrame,
    *,
    include_grades: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create the model matrix and target without fitting any transformer."""

    if "G3" not in frame.columns:
        raise ValueError("The input data must contain the G3 target column.")

    data = frame.copy()
    if include_grades:
        data["grade_progress"] = data["G2"] - data["G1"]

    return data.drop(columns=["G3"]), data["G3"].astype(float)


def build_preprocessing_pipeline(
    X_train: pd.DataFrame,
    model,
    *,
    include_grades: bool = True,
) -> Pipeline:
    """Create a model pipeline so preprocessing is fitted inside each fold."""

    return Pipeline(
        steps=[
            (
                "preprocessor",
                get_preprocessor(X_train, include_grades=include_grades),
            ),
            ("model", model),
        ]
    )
