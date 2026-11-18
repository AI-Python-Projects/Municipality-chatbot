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

    _KEYWORDS = {
    "pothole":         ["pothole", "asphalt", "crater", "road damage",
                        "road surface", "pavement", "cracked road"],
    "noise":           ["noise", "loud", "music", "humming", "barking",
                        "party", "construction", "fan", "hvac", "machinery",
                        "bass", "alarm"],
    "water_leakage":   ["water leak", "leaking", "hydrant", "pipe",
                        "burst", "water main", "bubbling", "puddle"],
    "streetlight":     ["streetlight", "street light", "lamp", "flickering",
                        "bulb", "light pole", "lights out"],
    "drainage":        ["drain", "sewer", "storm drain", "manhole",
                        "flooding", "standing water", "water pools", "grate"],
    "garbage":         ["garbage", "trash", "litter", "dumpster", "bin",
                        "rubbish", "waste", "recycling", "debris"],
    "illegal_parking": ["illegal parking", "double-parked", "double parked",
                        "parked on", "fire hydrant", "no-parking",
                        "bike lane", "driveway", "abandoned vehicle"],
    "illegal_dumping": ["illegal dumping", "dumping", "dumped", "mattress",
                        "furniture", "tires", "appliances", "electronics"],
    "graffiti":        ["graffiti", "vandalism", "spray paint", "tags"],
}

def _infer_issue_type(text):
    t = text.lower()
    best, best_hits = "unknown", 0
    for itype, kws in _KEYWORDS.items():
        hits = sum(1 for kw in kws if kw in t)
        if hits > best_hits:
            best_hits = hits
            best = itype
    return best
