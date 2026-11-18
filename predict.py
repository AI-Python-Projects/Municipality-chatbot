import os
import pandas as pd
import joblib

from config import (
    MODEL_PATH, SLA_CSV, ISSUE_STATS_CSV, DEFAULT_SLA_HOURS,
)
from preprocessing import compute_priority

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run python train.py first.")

model = joblib.load(MODEL_PATH)

SLA_TABLE = {}
if os.path.exists(SLA_CSV):
    _df = pd.read_csv(SLA_CSV)
    SLA_TABLE = {(r["department"], r["priority_level"]): int(r["sla_hours"])
                 for _, r in _df.iterrows()}

ISSUE_STATS = {}
if os.path.exists(ISSUE_STATS_CSV):
    _s = pd.read_csv(ISSUE_STATS_CSV)
    ISSUE_STATS = {
        r["issue_type"]: {
            "severity": float(r["median_severity"]),
            "area":     float(r["median_area"]),
            "reports":  float(r["median_reports"]),
        }
        for _, r in _s.iterrows()
    }