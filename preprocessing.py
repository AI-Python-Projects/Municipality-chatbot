import pandas as pd

from config import (
    COL_DESC, COL_TYPE, COL_DEPT, COL_PRIORITY,
    COL_SEVERITY, COL_AREA, COL_REPORTS, COL_ANON,
    COL_REPORT_DT, COL_RESOLVE_DT, COL_STATUS,
    COL_CITY, COL_STATE,
    W_SEVERITY, W_AREA, W_REPORTS, PRIORITY_BANDS,
)
_DEPT_ALIASES = {
    "road maintenance":      "Roads",
    "roads & highways":      "Roads",
    "roads and highways":    "Roads",
    "roads":                 "Roads",

    "electrical department": "Electrical",
    "electrical services":   "Electrical",
    "electrical":            "Electrical",

    "water department":      "Water",
    "water services":        "Water",
    "water management":      "Water",
    "waterworks":            "Water",

    "sanitation":            "Sanitation",
    "waste management":      "Sanitation",

    "drainage authority":    "Drainage",
    "drainage":              "Drainage",

    "public safety":         "Public Safety",
    "noise control":         "Public Safety",
    "event management":      "Public Safety",

    "traffic department":    "Traffic",
    "traffic":               "Traffic",

    "parks department":      "Parks",
    "parks & recreation":    "Parks",
    "parks and recreation":  "Parks",

    "public works":          "Public Works",
    "community services":    "Public Works",
    "code enforcement":      "Public Works",
    "postal services":       "Public Works",
}
_TYPE_ALIASES = {
    "water-leak": "water_leakage",
    "water_leak": "water_leakage",
    "illegal-parking": "illegal_parking",
    "illegal-dumping": "illegal_dumping",
}
def _norm_dept(s):
    if pd.isna(s) or not str(s).strip():
        return None
    k = str(s).strip().lower()
    return _DEPT_ALIASES.get(k, str(s).strip())


def _norm_type(s):
    if pd.isna(s) or not str(s).strip():
        return "unknown"
    k = str(s).strip().lower()
    return _TYPE_ALIASES.get(k, k)

def _to_bool(s):
    if pd.isna(s):
        return 0
    if isinstance(s, bool):
        return int(s)
    return int(str(s).strip().lower() in ("true", "1", "yes"))


def _norm_str(s):
    if pd.isna(s) or not str(s).strip():
        return "unknown"
    return str(s).strip().lower()

def compute_priority(severity, area, reports):
    score = round(
        W_SEVERITY * float(severity)
        + W_AREA   * float(area)
        + W_REPORTS * (float(reports) / 10.0),
        2,
    )
    if score >= PRIORITY_BANDS["critical"]:
        return score, "critical"
    if score >= PRIORITY_BANDS["high"]:
        return score, "high"
    if score >= PRIORITY_BANDS["medium"]:
        return score, "medium"
    return score, "low"
    def load_and_prepare(path):
    df = pd.read_csv(path)

    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = [c.strip() for c in df.columns]
    # Target
df["department"] = df[COL_DEPT].apply(_norm_dept)

# Text
df["issue_type"]  = df[COL_TYPE].apply(_norm_type)
df["description"] = df[COL_DESC].fillna("").astype(str)
df["text"] = ...