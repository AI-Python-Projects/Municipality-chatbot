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