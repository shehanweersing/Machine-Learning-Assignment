"""Generate reproducible EDA tables and figures for the Math dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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

    summary = {
        "rows": len(frame),
        "columns": len(frame.columns),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "target": frame["G3"].describe().round(4).to_dict(),
        "target_correlations": correlations.round(4).to_dict(),
        "failure_group_means": frame.groupby("failures")["G3"].mean().round(4).to_dict(),
        "studytime_group_means": frame.groupby("studytime")["G3"].mean().round(4).to_dict(),
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
