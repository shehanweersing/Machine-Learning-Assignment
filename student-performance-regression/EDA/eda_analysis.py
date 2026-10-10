"""Generate reproducible EDA tables and figures for the Math dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def analyse(data_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(data_path, sep=";")

    numeric = frame.select_dtypes(include="number")
    correlations = numeric.corr(numeric_only=True)["G3"].drop("G3")
    correlation_table = (
        correlations.abs()
        .sort_values(ascending=False)
        .to_frame("absolute_correlation_with_G3")
        .assign(correlation_with_G3=correlations)
    )
    correlation_table.to_csv(output_dir / "target_correlations.csv")

    quality = pd.DataFrame(
        {
            "metric": [
                "rows",
                "columns",
                "missing_cells",
                "duplicate_rows",
                "zero_final_grades",
            ],
            "value": [
                len(frame),
                len(frame.columns),
                int(frame.isna().sum().sum()),
                int(frame.duplicated().sum()),
                int((frame["G3"] == 0).sum()),
            ],
        }
    )
    quality.to_csv(output_dir / "data_quality_summary.csv", index=False)

    grouped = pd.concat(
        [
            frame.groupby("failures")["G3"].agg(["count", "mean", "median"]).assign(
                grouping="failures"
            ).reset_index(names="level"),
            frame.groupby("studytime")["G3"].agg(["count", "mean", "median"]).assign(
                grouping="studytime"
            ).reset_index(names="level"),
        ],
        ignore_index=True,
    )
    grouped.to_csv(output_dir / "grouped_target_summary.csv", index=False)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.histplot(frame["G3"], bins=21, discrete=True, color="royalblue")
    plt.title("Distribution of final mathematics grade (G3)")
    plt.xlabel("Final grade (0-20)")
    plt.ylabel("Students")
    plt.tight_layout()
    plt.savefig(output_dir / "g3_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, linewidths=0.3)
    plt.title("Correlation heatmap for numeric variables")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=frame, x="failures", y="G3", color="skyblue")
    sns.stripplot(data=frame, x="failures", y="G3", color="black", alpha=0.25, size=3)
    plt.title("Final grade by number of previous failures")
    plt.xlabel("Previous failures")
    plt.ylabel("Final grade (G3)")
    plt.tight_layout()
    plt.savefig(output_dir / "g3_by_failures.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.regplot(
        data=frame,
        x="G2",
        y="G3",
        scatter_kws={"alpha": 0.45},
        line_kws={"color": "red"},
    )
    plt.title("Final grade versus second-period grade")
    plt.xlabel("Second-period grade (G2)")
    plt.ylabel("Final grade (G3)")
    plt.tight_layout()
    plt.savefig(output_dir / "g2_vs_g3.png", dpi=160)
    plt.close()

    # This is a cross-sectional dataset, so progression across G1 -> G2 -> G3
    # is the appropriate time-ordered view rather than a calendar time series.
    progression = frame[["G1", "G2", "G3"]].mean()
    progression_std = frame[["G1", "G2", "G3"]].std()
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        ["G1", "G2", "G3"],
        progression.values,
        yerr=progression_std.values,
        marker="o",
        linewidth=3,
        color="#13b5a4",
        capsize=6,
    )
    plt.ylim(0, 20)
    plt.title("Academic progression across grading periods")
    plt.xlabel("Grading period")
    plt.ylabel("Mean grade (with standard deviation)")
    plt.tight_layout()
    plt.savefig(output_dir / "academic_progression.png", dpi=160)
    plt.close()

    # A zero-missingness result is still evidence and should be visible.
    missing = frame.isna().sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(8, 5))
    plt.bar(missing.index, missing.values, color="#8178df")
    plt.axhline(0, color="#17343b", linewidth=0.8)
    plt.xticks(rotation=45, ha="right")
    plt.title("Missing-value audit (top 10 columns)")
    plt.ylabel("Missing cells")
    plt.tight_layout()
    plt.savefig(output_dir / "missing_values_analysis.png", dpi=160)
    plt.close()

    # Show why scaling is useful for numeric estimators without changing the
    # source data: all transformed columns have comparable standard deviation.
    scaling_columns = ["absences", "G1", "G2", "G3"]
    raw_std = frame[scaling_columns].std()
    scaled = StandardScaler().fit_transform(frame[scaling_columns])
    scaled_std = pd.Series(scaled.std(axis=0), index=scaling_columns)
    x = np.arange(len(scaling_columns))
    width = 0.36
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, raw_std.values, width, label="Before scaling", color="#ef6f61")
    plt.bar(x + width / 2, scaled_std.values, width, label="After scaling", color="#13b5a4")
    plt.xticks(x, scaling_columns)
    plt.ylabel("Standard deviation")
    plt.title("Feature scaling comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "feature_scaling_comparison.png", dpi=160)
    plt.close()

    top_correlations = correlation_table.head(8).sort_values(
        "correlation_with_G3"
    )
    plt.figure(figsize=(8, 5))
    colors = ["#ef6f61" if value < 0 else "#13b5a4" for value in top_correlations["correlation_with_G3"]]
    plt.barh(top_correlations.index, top_correlations["correlation_with_G3"], color=colors)
    plt.axvline(0, color="#17343b", linewidth=0.8)
    plt.xlabel("Pearson correlation with G3")
    plt.title("Top numeric correlations with final grade")
    plt.tight_layout()
    plt.savefig(output_dir / "top_correlations.png", dpi=160)
    plt.close()

    q1, q3 = frame["absences"].quantile([0.25, 0.75])
    iqr = q3 - q1
    outlier_count = int(((frame["absences"] < q1 - 1.5 * iqr) | (frame["absences"] > q3 + 1.5 * iqr)).sum())
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=frame["absences"], color="#f4c95d")
    plt.title(f"Absence outlier review ({outlier_count} IQR outliers; retained)")
    plt.xlabel("Absences")
    plt.tight_layout()
    plt.savefig(output_dir / "absences_outlier_analysis.png", dpi=160)
    plt.close()

    summary = {
        "rows": len(frame),
        "columns": len(frame.columns),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "target": frame["G3"].describe().round(4).to_dict(),
        "target_correlations": correlations.round(4).to_dict(),
        "failure_group_means": frame.groupby("failures")["G3"].mean().round(4).to_dict(),
        "studytime_group_means": frame.groupby("studytime")["G3"].mean().round(4).to_dict(),
        "missing_by_column": frame.isna().sum().astype(int).to_dict(),
        "absences_iqr_outliers": outlier_count,
    }
    (output_dir / "eda_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).parents[2]
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "student-performance-regression" / "data" / "student-mat.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "outputs",
    )
    args = parser.parse_args()
    print(json.dumps(analyse(args.data, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
