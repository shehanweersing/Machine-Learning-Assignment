"""Build a reproducible intervention-group profiling lens from both subjects.

This is a descriptive profiling workflow, not a second grading model.  It
uses the 382 records matched by the key in ``student-merge.R`` to
identify broad support-priority groups across Mathematics and Portuguese.
The groups are intended for human review and programme planning only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


MERGE_KEYS = [
    "school",
    "sex",
    "age",
    "address",
    "famsize",
    "Pstatus",
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "reason",
    "nursery",
    "internet",
]

CLUSTER_FEATURES = [
    "age",
    "traveltime_math",
    "studytime_math",
    "failures_math",
    "absences_math",
    "G1_math",
    "G2_math",
    "traveltime_portuguese",
    "studytime_portuguese",
    "failures_portuguese",
    "absences_portuguese",
    "G1_portuguese",
    "G2_portuguese",
]


def load_matched_data(math_path: Path, portuguese_path: Path) -> pd.DataFrame:
    """Return the inner merge used by the supplied R script."""

    math = pd.read_csv(math_path, sep=";")
    portuguese = pd.read_csv(portuguese_path, sep=";")
    merged = math.merge(
        portuguese,
        on=MERGE_KEYS,
        suffixes=("_math", "_portuguese"),
        how="inner",
    )
    if merged.empty:
        raise ValueError("The subject datasets produced no matched records.")
    return merged


def assign_profile(frame: pd.DataFrame) -> pd.Series:
    """Assign mutually exclusive, transparent support-priority profiles."""

    math_low = frame["G3_math"] < 10
    portuguese_low = frame["G3_portuguese"] < 10
    math_weak_progress = frame["G2_math"] < 10
    portuguese_weak_progress = frame["G2_portuguese"] < 10
    repeated_failures = (frame["failures_math"] > 0) | (
        frame["failures_portuguese"] > 0
    )
    high_absence = (frame["absences_math"] >= frame["absences_math"].quantile(0.75)) | (
        frame["absences_portuguese"]
        >= frame["absences_portuguese"].quantile(0.75)
    )

    both_subjects = math_low & portuguese_low
    math_only = math_low & ~portuguese_low
    portuguese_only = portuguese_low & ~math_low
    emerging = (
        ~both_subjects
        & ~math_only
        & ~portuguese_only
        & (math_weak_progress | portuguese_weak_progress | repeated_failures | high_absence)
    )
    profile = pd.Series(
        pd.NA,
        index=frame.index,
        dtype="string",
    )
    profile = profile.mask(
        both_subjects, "Priority: low outcomes in both subjects"
    )
    profile = profile.mask(math_only, "Priority: Mathematics-specific support")
    profile = profile.mask(
        portuguese_only, "Priority: Portuguese-specific support"
    )
    profile = profile.mask(emerging, "Monitor: emerging risk signals")
    return profile.fillna("Monitor: currently stable")


def build_profiles(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create student-level evidence and an aggregate group table."""

    profiles = merged.copy()
    profiles["math_portuguese_grade_gap"] = (
        profiles["G3_math"] - profiles["G3_portuguese"]
    )
    profiles["math_low_outcome"] = profiles["G3_math"] < 10
    profiles["portuguese_low_outcome"] = profiles["G3_portuguese"] < 10
    profiles["previous_failure_signal"] = (
        (profiles["failures_math"] > 0) | (profiles["failures_portuguese"] > 0)
    )
    profiles["high_absence_signal"] = (
        (profiles["absences_math"] >= profiles["absences_math"].quantile(0.75))
        | (
            profiles["absences_portuguese"]
            >= profiles["absences_portuguese"].quantile(0.75)
        )
    )
    profiles["profile"] = assign_profile(profiles)

    profile_summary = (
        profiles.groupby("profile", as_index=False)
        .agg(
            students=("profile", "size"),
            mean_math_G3=("G3_math", "mean"),
            mean_portuguese_G3=("G3_portuguese", "mean"),
            low_math_rate=("math_low_outcome", "mean"),
            low_portuguese_rate=("portuguese_low_outcome", "mean"),
            failure_signal_rate=("previous_failure_signal", "mean"),
            high_absence_signal_rate=("high_absence_signal", "mean"),
        )
        .sort_values(["students", "profile"], ascending=[False, True])
    )
    return profiles, profile_summary


def _score_labels(values: np.ndarray, labels: np.ndarray) -> float | None:
    """Score non-degenerate cluster labels, excluding DBSCAN noise."""

    valid = labels != -1
    unique = np.unique(labels[valid])
    if valid.sum() < 3 or len(unique) < 2:
        return None
    return float(silhouette_score(values[valid], labels[valid]))


def build_clusters(
    profiles: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str, int]:
    """Compare K-Means, hierarchical and DBSCAN on pre-outcome features."""

    values = profiles[CLUSTER_FEATURES].astype(float).to_numpy()
    scaled = StandardScaler().fit_transform(values)
    candidates: list[dict[str, Any]] = []
    fitted: dict[str, np.ndarray] = {}

    for k in range(2, 7):
        labels = KMeans(
            n_clusters=k, random_state=42, n_init=20
        ).fit_predict(scaled)
        score = _score_labels(scaled, labels)
        candidates.append(
            {"algorithm": "K-Means", "parameters": f"k={k}", "clusters": k, "noise": 0, "silhouette": score}
        )
        fitted[f"K-Means|k={k}"] = labels

        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(scaled)
        score = _score_labels(scaled, labels)
        candidates.append(
            {"algorithm": "Hierarchical", "parameters": f"k={k}", "clusters": k, "noise": 0, "silhouette": score}
        )
        fitted[f"Hierarchical|k={k}"] = labels

    for eps in (0.8, 1.0, 1.2, 1.5, 1.8):
        labels = DBSCAN(eps=eps, min_samples=8).fit_predict(scaled)
        score = _score_labels(scaled, labels)
        cluster_count = len(set(labels)) - (1 if -1 in labels else 0)
        candidates.append(
            {
                "algorithm": "DBSCAN",
                "parameters": f"eps={eps}, min_samples=8",
                "clusters": cluster_count,
                "noise": int((labels == -1).sum()),
                "silhouette": score,
            }
        )
        fitted[f"DBSCAN|eps={eps}, min_samples=8"] = labels

    comparison = pd.DataFrame(candidates)
    valid = comparison.dropna(subset=["silhouette"])
    if valid.empty:
        raise RuntimeError("No clustering configuration produced valid groups.")
    best_row = valid.sort_values(
        ["silhouette", "noise"], ascending=[False, True]
    ).iloc[0]
    best_key = f"{best_row['algorithm']}|{best_row['parameters']}"
    best_labels = fitted[best_key]
    profiles = profiles.copy()
    profiles["cluster"] = best_labels
    return profiles, comparison.sort_values("silhouette", ascending=False), str(best_row["algorithm"]), int(best_row["clusters"])


def save_cluster_outputs(
    profiles: pd.DataFrame,
    comparison: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    """Write cluster assignments, comparison evidence and visual diagnostics."""

    profiles.to_csv(output_dir / "student_cluster_assignments.csv", index=False)
    comparison.to_csv(output_dir / "clustering_comparison.csv", index=False)
    cluster_summary = (
        profiles.groupby("cluster", dropna=False, as_index=False)
        .agg(
            records=("cluster", "size"),
            mean_math_G3=("G3_math", "mean"),
            mean_portuguese_G3=("G3_portuguese", "mean"),
            mean_math_absences=("absences_math", "mean"),
            mean_portuguese_absences=("absences_portuguese", "mean"),
            mean_math_failures=("failures_math", "mean"),
            mean_portuguese_failures=("failures_portuguese", "mean"),
        )
        .sort_values("cluster")
    )
    cluster_summary.to_csv(output_dir / "cluster_profile_summary.csv", index=False)
    cluster_sizes = (
        profiles["cluster"].value_counts().sort_index().rename_axis("cluster").reset_index(name="records")
    )
    cluster_sizes.to_csv(output_dir / "cluster_sizes.csv", index=False)

    scaled = StandardScaler().fit_transform(
        profiles[CLUSTER_FEATURES].astype(float)
    )
    projection = PCA(n_components=2, random_state=42).fit_transform(scaled)
    plot_data = pd.DataFrame(
        {"pc1": projection[:, 0], "pc2": projection[:, 1], "cluster": profiles["cluster"].astype(str)}
    )
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=plot_data, x="pc1", y="pc2", hue="cluster", palette="tab10", alpha=0.75)
    plt.title("Best clustering solution projected into two PCA dimensions")
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_pca_visualization.png", dpi=160)
    plt.close()

    feature_means = profiles.groupby("cluster")[CLUSTER_FEATURES].mean()
    feature_z = (feature_means - feature_means.mean()) / feature_means.std(ddof=0).replace(0, 1)
    feature_z.to_csv(output_dir / "cluster_feature_profile.csv")
    plt.figure(figsize=(12, 5))
    sns.heatmap(feature_z.T, cmap="vlag", center=0, annot=False)
    plt.title("Relative feature profile by cluster (standardised group means)")
    plt.xlabel("Cluster")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_feature_heatmap.png", dpi=160)
    plt.close()

    valid_comparison = comparison.dropna(subset=["silhouette"]).copy()
    valid_comparison["configuration"] = (
        valid_comparison["algorithm"] + " · " + valid_comparison["parameters"]
    )
    plt.figure(figsize=(11, 6))
    plot_data = valid_comparison.head(15).sort_values("silhouette")
    sns.barplot(data=plot_data, x="silhouette", y="configuration", hue="algorithm", dodge=False)
    plt.xlim(0, max(0.5, float(plot_data["silhouette"].max()) + 0.05))
    plt.title("Clustering configurations ranked by silhouette score")
    plt.xlabel("Silhouette score (higher is better)")
    plt.ylabel("")
    plt.legend(title="Algorithm", loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "clustering_algorithm_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.barplot(data=cluster_sizes, x="cluster", y="records", color="#13b5a4")
    plt.title("Records assigned to each selected cluster")
    plt.xlabel("Cluster label (arbitrary)")
    plt.ylabel("Matched records")
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_sizes.png", dpi=160)
    plt.close()

    best = comparison.dropna(subset=["silhouette"]).iloc[0]
    selected_labels = profiles["cluster"].to_numpy()
    stability_scores = []
    selected_scaled = StandardScaler().fit_transform(
        profiles[CLUSTER_FEATURES].astype(float)
    )
    for seed in range(10):
        repeat_labels = KMeans(
            n_clusters=int(best["clusters"]),
            random_state=seed,
            n_init=20,
        ).fit_predict(selected_scaled)
        stability_scores.append(
            adjusted_rand_score(selected_labels, repeat_labels)
        )
    stability = {
        "repeats": 10,
        "mean_adjusted_rand_index": float(np.mean(stability_scores)),
        "minimum_adjusted_rand_index": float(np.min(stability_scores)),
        "scores": stability_scores,
    }
    (output_dir / "clustering_stability.json").write_text(
        json.dumps(stability, indent=2), encoding="utf-8"
    )
    return {
        "features_used": CLUSTER_FEATURES,
        "selection_metric": "silhouette score (higher is better)",
        "selected_algorithm": str(best["algorithm"]),
        "selected_parameters": str(best["parameters"]),
        "selected_clusters": int(best["clusters"]),
        "selected_silhouette": float(best["silhouette"]),
        "cluster_summary": cluster_summary.to_dict(orient="records"),
        "cluster_sizes": cluster_sizes.to_dict(orient="records"),
        "stability": stability,
        "limitations": [
            "Clustering finds similarity patterns; it does not prove causes or intervention effects.",
            "G3 was excluded from clustering and used only to interpret the resulting groups.",
            "The matched merge key is not a unique student ID and can create duplicate-key records.",
            "Cluster labels are arbitrary identifiers and should not be treated as student quality rankings.",
        ],
    }


def save_outputs(
    profiles: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    cluster_result: dict[str, Any],
) -> dict[str, Any]:
    """Persist tables, charts, and a machine-readable audit summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    export_columns = [
        "school",
        "sex",
        "age",
        "G3_math",
        "G3_portuguese",
        "G2_math",
        "G2_portuguese",
        "failures_math",
        "failures_portuguese",
        "absences_math",
        "absences_portuguese",
        "math_portuguese_grade_gap",
        "math_low_outcome",
        "portuguese_low_outcome",
        "previous_failure_signal",
        "high_absence_signal",
        "profile",
    ]
    profiles[export_columns].to_csv(output_dir / "student_intervention_profiles.csv", index=False)
    summary.to_csv(output_dir / "intervention_profile_summary.csv", index=False)

    sns.set_theme(style="whitegrid")
    order = summary["profile"].tolist()
    plt.figure(figsize=(10, 5))
    sns.barplot(data=summary, y="profile", x="students", order=order, color="#13b5a4")
    plt.title("Matched students by intervention profile")
    plt.xlabel("Students")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "intervention_profile_counts.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=profiles,
        x="G3_math",
        y="G3_portuguese",
        hue="profile",
        hue_order=order,
        palette="Set2",
        alpha=0.75,
    )
    plt.axvline(10, color="#ef6f61", linestyle="--", linewidth=1)
    plt.axhline(10, color="#ef6f61", linestyle="--", linewidth=1)
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.xlabel("Mathematics final grade (G3)")
    plt.ylabel("Portuguese final grade (G3)")
    plt.title("Cross-subject outcome quadrants")
    plt.tight_layout()
    plt.savefig(output_dir / "cross_subject_risk_quadrants.png", dpi=160)
    plt.close()

    long_grades = profiles.melt(
        id_vars="profile",
        value_vars=["G3_math", "G3_portuguese"],
        var_name="subject",
        value_name="G3",
    )
    long_grades["subject"] = long_grades["subject"].map(
        {"G3_math": "Mathematics", "G3_portuguese": "Portuguese"}
    )
    plt.figure(figsize=(11, 5))
    sns.boxplot(data=long_grades, x="profile", y="G3", hue="subject")
    plt.axhline(10, color="#ef6f61", linestyle="--", linewidth=1)
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 20)
    plt.title("Grade distributions within intervention profiles")
    plt.xlabel("")
    plt.ylabel("Final grade (G3)")
    plt.tight_layout()
    plt.savefig(output_dir / "profile_grade_distributions.png", dpi=160)
    plt.close()

    summary_payload = {
        "matched_records": int(len(profiles)),
        "merge_keys": MERGE_KEYS,
        "low_grade_threshold": 10,
        "high_absence_rule": "subject absences at or above that subject's 75th percentile",
        "profiles": summary.to_dict(orient="records"),
        "clustering": cluster_result,
        "interpretation": (
            "Profiles are descriptive support-prioritisation signals. They do not "
            "prove causes, diagnose students, or replace teacher/counsellor review."
        ),
        "limitations": [
            "The merge key has no explicit student ID, so matched records are not guaranteed identity matches.",
            "The data is historical and cross-sectional; profile membership is not a causal intervention effect.",
            "The grade threshold and absence rule are transparent screening choices, not official school policy.",
            "Sensitive attributes must not be used as automatic exclusion or discipline rules.",
        ],
    }
    (output_dir / "intervention_profile_summary.json").write_text(
        json.dumps(summary_payload, indent=2, default=str), encoding="utf-8"
    )
    return summary_payload


def run(
    math_path: Path,
    portuguese_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run matching, profiling, quality checks, and evidence generation."""

    merged = load_matched_data(math_path, portuguese_path)
    profiles, summary = build_profiles(merged)
    profiles, comparison, _, _ = build_clusters(profiles)
    cluster_result = save_cluster_outputs(profiles, comparison, output_dir)
    result = save_outputs(profiles, summary, output_dir, cluster_result)
    math = pd.read_csv(math_path, sep=";")
    portuguese = pd.read_csv(portuguese_path, sep=";")
    result["source_rows"] = {
        "mathematics": int(math.shape[0]),
        "portuguese": int(portuguese.shape[0]),
    }
    result["merge_quality"] = {
        "math_duplicate_merge_keys": int(math.duplicated(MERGE_KEYS).sum()),
        "portuguese_duplicate_merge_keys": int(
            portuguese.duplicated(MERGE_KEYS).sum()
        ),
        "warning": (
            "The supplied merge key is not a unique student identifier. "
            "Counts describe matched records, not verified student identities."
        ),
    }
    return result


def main() -> None:
    root = Path(__file__).parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--math-data",
        type=Path,
        default=root / "student-performance-regression" / "data" / "student-mat.csv",
    )
    parser.add_argument(
        "--portuguese-data",
        type=Path,
        default=root / "student-performance-regression" / "data" / "student-por.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "intervention-group-profiling" / "outputs",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.math_data, args.portuguese_data, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
