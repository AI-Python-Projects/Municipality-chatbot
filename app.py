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