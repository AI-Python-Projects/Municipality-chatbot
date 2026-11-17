import os
import joblib
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from config import (
    TRAIN_CSV, TEST_CSV, MODEL_PATH, SLA_CSV, ISSUE_STATS_CSV,
    OUTPUT_DIR, SLA_BUFFER, DEFAULT_SLA_HOURS, MIN_SAMPLES_PER_CLASS,
)
from preprocessing import load_and_prepare

sns.set_style("whitegrid")


def main():
    print("=" * 60)
    print("  TRAINING MUNICIPAL SERVICES CLASSIFIER")
    print("=" * 60)

    df = load_and_prepare(TRAIN_CSV)
    print(f"Loaded {len(df)} rows across {df['department'].nunique()} departments\n")
    print(df["department"].value_counts().to_string())

    counts = df["department"].value_counts()
    rare = counts[counts < MIN_SAMPLES_PER_CLASS].index.tolist()
    if rare:
        print(f"\nDropping rare departments (< {MIN_SAMPLES_PER_CLASS}): {rare}")
        df = df[~df["department"].isin(rare)].reset_index(drop=True)
        print(f"Kept {len(df)} rows across {df['department'].nunique()} departments.")

    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["department"]
    )
    test_df.to_csv(TEST_CSV, index=False)
    print(f"\nTrain: {len(train_df)}   Test: {len(test_df)}")
    print(f"Test split saved -> {TEST_CSV}")

    numeric_cols = ["severity", "area", "reports", "is_weekend"]
    cat_cols     = ["issue_type", "is_anonymous"]

    preprocess = ColumnTransformer([
        ("text", TfidfVectorizer(ngram_range=(1, 2), max_features=5000,
                                 min_df=1, sublinear_tf=True), "text"),
        ("num",  StandardScaler(), numeric_cols),
        ("cat",  OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    model = Pipeline([
        ("preprocess", preprocess),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=None,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )),
    ])

    X_train = train_df.drop(columns=["department"])
    y_train = train_df["department"]

    print("\nFitting RandomForestClassifier...")
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved -> {MODEL_PATH}")

    _save_class_chart(df)
    _save_sla(df)
    _save_issue_stats(df)

    print("\n[DONE] Training complete. Now run:  python test.py\n")


def _save_class_chart(df):
    counts = df["department"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=counts.values, y=counts.index, palette="viridis", ax=ax)
    ax.set_title("Reports per Department", fontweight="bold")
    ax.set_xlabel("Number of reports")
    ax.set_ylabel("")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "class_distribution.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Chart saved -> {path}")


def _save_sla(df):
    print("\nDeriving SLA targets...")
    resolved = df[df["resolution_hours"].notna() & (df["resolution_hours"] > 0)]

    grouped = (resolved.groupby(["department", "priority_level"])
                        ["resolution_hours"].median().reset_index())
    grouped["sla_hours"] = (grouped["resolution_hours"] * SLA_BUFFER).round(0).astype(int)

    rows = []
    for dept in sorted(df["department"].unique()):
        for level in ["low", "medium", "high", "critical"]:
            m = grouped[(grouped.department == dept) &
                        (grouped.priority_level == level)]
            sla = int(m["sla_hours"].iloc[0]) if not m.empty else DEFAULT_SLA_HOURS[level]
            rows.append({"department": dept, "priority_level": level, "sla_hours": sla})

    pd.DataFrame(rows).to_csv(SLA_CSV, index=False)
    print(f"SLA targets saved -> {SLA_CSV}")


def _save_issue_stats(df):
    stats = (df.groupby("issue_type")
               .agg(median_severity=("severity", "median"),
                    median_area=("area", "median"),
                    median_reports=("reports", "median"),
                    count=("severity", "count"))
               .reset_index())
    stats["median_severity"] = stats["median_severity"].round(1)
    stats["median_area"]     = stats["median_area"].round(1)
    stats["median_reports"]  = stats["median_reports"].round(1)
    stats.to_csv(ISSUE_STATS_CSV, index=False)
    print(f"Issue stats saved -> {ISSUE_STATS_CSV}")


if __name__ == "__main__":
    main()