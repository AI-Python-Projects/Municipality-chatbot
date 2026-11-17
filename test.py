import os
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score,
)

from config import TEST_CSV, MODEL_PATH, OUTPUT_DIR

sns.set_style("whitegrid")

def main():
    print("=" * 60)
    print("  TESTING MUNICIPAL SERVICES CLASSIFIER")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run python train.py first.")
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"Test set not found at {TEST_CSV}. Run python train.py first.")

    model = joblib.load(MODEL_PATH)
    df    = pd.read_csv(TEST_CSV)
    print(f"Model : {MODEL_PATH}")
    print(f"Data  : {TEST_CSV}  ({len(df)} rows)\n")

    X_test = df.drop(columns=["department"])
    y_true = df["department"]
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("-- Overall performance --")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}   (weighted)")
    print(f"  Recall    : {rec:.4f}   (weighted)")
    print(f"  F1-score  : {f1:.4f}   (weighted)")

     report_txt = classification_report(y_true, y_pred, zero_division=0)
    print("\n" + report_txt)

    _save_confusion_matrix(y_true, y_pred)
    _save_per_class(y_true, y_pred)
    _save_confidence(model, X_test)
    _save_feature_importance(model)

    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Accuracy  : {acc:.4f}\n")
        f.write(f"Precision : {prec:.4f}  (weighted)\n")
        f.write(f"Recall    : {rec:.4f}  (weighted)\n")
        f.write(f"F1-score  : {f1:.4f}  (weighted)\n\n")
        f.write(report_txt)
    print(f"\nReport saved -> {report_path}")
    print("\n[DONE] Testing complete.")

    def _save_confusion_matrix(y_true, y_pred):
    labels = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Confusion Matrix - Test Set", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Chart saved -> {path}")

    def _save_per_class(y_true, y_pred):
    labels = sorted(y_true.unique())
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rows = [(k, v["precision"], v["recall"], v["f1-score"])
            for k, v in report.items() if k in labels]
    per_df = pd.DataFrame(rows, columns=["class", "precision", "recall", "f1"])
    per_m  = per_df.melt(id_vars="class", var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=per_m, x="score", y="class",
                hue="metric", palette="Set2", ax=ax)
    ax.set_title("Precision / Recall / F1 per Department", fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_ylabel("")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "per_class_scores.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Chart saved -> {path}")