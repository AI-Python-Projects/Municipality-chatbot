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
