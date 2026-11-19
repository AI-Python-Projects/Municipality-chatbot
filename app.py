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