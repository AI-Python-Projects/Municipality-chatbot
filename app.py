"""
app.py - Municipal Services ML Dashboard
Fully automated. Loads dataset, runs ML, renders tables + charts.
No user input.
"""
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from sklearn.model_selection import train_test_split

from config import TRAIN_CSV, MODEL_PATH, SLA_CSV
from preprocessing import load_and_prepare

st.set_page_config(
    page_title="Municipal Services | ML Dashboard",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.main-title { font-size: 2.2rem; font-weight: 700; color: #0f4c81; margin-bottom: 0.2rem; }
.subtitle   { color: #6c757d; font-size: 1rem; margin-bottom: 1.5rem; }
.kpi-card {
    padding: 1.2rem; border-radius: 12px; color: white; text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
.kpi-val { font-size: 1.9rem; font-weight: 700; }
.kpi-lbl { font-size: 0.85rem; opacity: 0.92; margin-top: 0.3rem; }
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

PALETTE = [
    "#0f4c81", "#2e86ab", "#a23b72", "#f18f01",
    "#c73e1d", "#6a994e", "#8e5572", "#457b9d", "#bc4749",
]
PRIORITY_COLORS = {
    "critical": "#c73e1d",
    "high":     "#f18f01",
    "medium":   "#2e86ab",
    "low":      "#6a994e",
}


@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    return load_and_prepare(TRAIN_CSV)


@st.cache_resource(show_spinner="Loading ML model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Loading SLA targets...")
def load_sla():
    if not os.path.exists(SLA_CSV):
        return pd.DataFrame()
    return pd.read_csv(SLA_CSV)


@st.cache_data(show_spinner="Running model...")
def predict_all(_df):
    model = joblib.load(MODEL_PATH)
    X = _df.drop(columns=["department"])
    out = _df.copy()
    out["predicted_department"] = model.predict(X)
    if hasattr(model, "predict_proba"):
        out["confidence"] = model.predict_proba(X).max(axis=1).round(3)
    else:
        out["confidence"] = np.nan
    return out


def kpi(col, value, label, color):
    col.markdown(
        f'<div class="kpi-card" style="background:linear-gradient(135deg,{color} 0%,{color}cc 100%);">'
        f'<div class="kpi-val">{value}</div>'
        f'<div class="kpi-lbl">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


    def style(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#0f4c81")),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        margin=dict(l=40, r=20, t=60, b=40),
        height=420,
    )
    return fig


# =====================================================================
st.markdown('<div class="main-title">Municipal Services ML Dashboard</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Automated routing, prioritisation and SLA analytics '
    'powered by machine learning.</div>',
    unsafe_allow_html=True,
)

df     = load_data()
model  = load_model()
sla_df = load_sla()

if model is None:
    st.error("Model not found. Run `python train.py` first, then refresh.")
    st.stop()

# defensive: ensure optional columns exist
for col, default in [("city", "unknown"), ("state", "unknown"),
                    ("status", "unknown")]:
    if col not in df.columns:
        df[col] = default

df_pred = predict_all(df)

_, test_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["department"]
)
X_test = test_df.drop(columns=["department"])
y_true = test_df["department"]
y_pred = model.predict(X_test)

acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

c1, c2, c3, c4 = st.columns(4)
kpi(c1, f"{len(df):,}",  "Total Reports",  PALETTE[0])
kpi(c2, f"{df['department'].nunique()}", "Departments", PALETTE[1])
kpi(c3, f"{df['issue_type'].nunique()}",  "Issue Types", PALETTE[2])
kpi(c4, f"{acc*100:.1f}%", "Model Accuracy", PALETTE[5])

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Distributions", "ML Performance",
    "SLA Analysis", "Full Predictions",
])