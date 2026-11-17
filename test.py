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