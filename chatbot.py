"""
chatbot.py - Fully automated batch analysis of the municipal dataset.
No user input. Reads the CSV, runs ML on every row, prints a text report,
and saves outputs/all_predictions.csv.

Usage:
    python chatbot.py
    python chatbot.py data/other.csv
"""
import sys
import os
import pandas as pd

from predict import analyse_auto
from config import TRAIN_CSV, PREDICTIONS_CSV


def _find_desc_col(df):
    for col in ["issue_description", "description", "Description"]:
        if col in df.columns:
            return col
    raise ValueError("No description column found in CSV.")


def analyse_dataset(csv_path):
    print("=" * 62)
    print("  MUNICIPAL SERVICES - AUTOMATED BATCH ANALYSIS")
    print("=" * 62)
    print(f"\nDataset: {csv_path}")

    df = pd.read_csv(csv_path)
    desc_col = _find_desc_col(df)
    print(f"Loaded {len(df)} rows\n")

    rows = []
    for _, r in df.iterrows():
        desc = str(r.get(desc_col, "")).strip()
        if not desc:
            continue
        res = analyse_auto(desc)
        rows.append({
            "description":         desc[:80],
            "detected_issue_type": res["detected_issue_type"],
            "department":          res["department"],
            "priority":            res["priority_level"],
            "score":               res["priority_score"],
            "sla_hours":           res["sla_hours"],
        })

    out = pd.DataFrame(rows)
    out.to_csv(PREDICTIONS_CSV, index=False)

    print(f"Processed {len(out)} reports\n")

    print("--- Department breakdown ---")
    print(out["department"].value_counts().to_string())
    print()

    print("--- Priority breakdown ---")
    print(out["priority"].value_counts().to_string())
    print()

    print("--- Detected issue-type breakdown ---")
    print(out["detected_issue_type"].value_counts().to_string())
    print()

    print("--- Average SLA by department (hours) ---")
    sla_summary = (out.groupby("department")["sla_hours"]
                      .mean().round(1).astype(int).reset_index()
                      .sort_values("sla_hours"))
    print(sla_summary.to_string(index=False))
    print()

    print("--- Priority distribution by department ---")
    cross = pd.crosstab(out["department"], out["priority"])
    for col in ["low", "medium", "high", "critical"]:
        if col not in cross.columns:
            cross[col] = 0
    cross = cross[["low", "medium", "high", "critical"]]
    print(cross.to_string())
    print()

    print(f"Full results saved -> {PREDICTIONS_CSV}")
    print("=" * 62)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else TRAIN_CSV
    analyse_dataset(path)