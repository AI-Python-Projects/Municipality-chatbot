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

