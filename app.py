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

# ---- TAB 1 ----
with tab1:
    st.subheader("Dataset Summary")
    summary = pd.DataFrame({
        "Metric": [
            "Total reports", "Departments", "Issue types",
            "Cities", "States", "Anonymous reports",
            "Date range (start)", "Date range (end)",
        ],
        "Value": [
            f"{len(df):,}",
            f"{df['department'].nunique()}",
            f"{df['issue_type'].nunique()}",
            f"{df['city'].nunique()}",
            f"{df['state'].nunique()}",
            f"{int(df['is_anonymous'].sum()):,} "
            f"({df['is_anonymous'].mean()*100:.1f}%)",
            str(df["report_datetime"].min().date())
            if df["report_datetime"].notna().any() else "n/a",
            str(df["report_datetime"].max().date())
            if df["report_datetime"].notna().any() else "n/a",
        ],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)
    
    st.subheader("First 10 Rows")
        cols = ["issue_type", "description", "department",
                "priority_level", "severity", "area", "reports", "city"]
        st.dataframe(df[cols].head(10), use_container_width=True, hide_index=True)
    
        c1, c2, c3 = st.columns(3)
    
        status_counts = df["status"].value_counts()
        fig = px.pie(values=status_counts.values, names=status_counts.index,
                     color_discrete_sequence=PALETTE, hole=0.45)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        c1.plotly_chart(style(fig, "Status Breakdown"), use_container_width=True)
    
        anon_counts = df["is_anonymous"].map(
            {0: "Identified", 1: "Anonymous"}).value_counts()
        fig = px.pie(values=anon_counts.values, names=anon_counts.index,
                     color_discrete_sequence=[PALETTE[1], PALETTE[4]], hole=0.45)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        c2.plotly_chart(style(fig, "Reporter Type"), use_container_width=True)

    priority_counts = df["priority_level"].value_counts()
        fig = px.pie(values=priority_counts.values, names=priority_counts.index,
                     color=priority_counts.index,
                     color_discrete_map=PRIORITY_COLORS, hole=0.45)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        c3.plotly_chart(style(fig, "Priority Distribution"), use_container_width=True)
    
        st.subheader("Top Cities by Report Volume")
        city_counts = df["city"].value_counts().head(10).reset_index()
        city_counts.columns = ["City", "Reports"]
        fig = px.bar(city_counts, x="Reports", y="City", orientation="h",
                     color="Reports", color_continuous_scale="Blues")
        st.plotly_chart(style(fig, ""), use_container_width=True)

    # ---- TAB 2 ----
with tab2:
    c1, c2 = st.columns(2)

    dept_counts = df["department"].value_counts().reset_index()
    dept_counts.columns = ["Department", "Reports"]
    fig = px.bar(dept_counts, x="Reports", y="Department",
                 orientation="h", color="Reports",
                 color_continuous_scale="Blues", text="Reports")
    fig.update_traces(textposition="outside")
    c1.plotly_chart(style(fig, "Reports per Department"), use_container_width=True)

    type_counts = df["issue_type"].value_counts().reset_index()
    type_counts.columns = ["Issue Type", "Reports"]
    fig = px.bar(type_counts, x="Reports", y="Issue Type",
                 orientation="h", color="Reports",
                 color_continuous_scale="Teal", text="Reports")
    fig.update_traces(textposition="outside")
    c2.plotly_chart(style(fig, "Reports per Issue Type"), use_container_width=True)

    c3, c4 = st.columns(2)

    fig = px.histogram(df, x="severity", nbins=10,
                           color_discrete_sequence=[PALETTE[3]])
        fig.update_layout(xaxis_title="Severity Score", yaxis_title="Frequency")
        c3.plotly_chart(style(fig, "Severity Distribution"), use_container_width=True)
    
        fig = px.histogram(df, x="area", nbins=10,
                           color_discrete_sequence=[PALETTE[1]])
        fig.update_layout(xaxis_title="Area Importance", yaxis_title="Frequency")
        c4.plotly_chart(style(fig, "Area Importance Distribution"),
                        use_container_width=True)
    
        fig = px.histogram(df, x="reports", nbins=20,
                           color_discrete_sequence=[PALETTE[2]])
        fig.update_layout(xaxis_title="Citizen Reports Count",
                          yaxis_title="Frequency")
        st.plotly_chart(style(fig, "Citizen Reports Volume"), use_container_width=True)

    st.subheader("Department Statistics")
        stats = (
            df.groupby("department")
              .agg(Reports=("department", "count"),
                   Avg_Severity=("severity", "mean"),
                   Avg_Area=("area", "mean"),
                   Avg_Citizen_Reports=("reports", "mean"),
                   Avg_Resolution_Hours=("resolution_hours", "mean"))
              .round(2).reset_index()
              .rename(columns={"department": "Department"})
              .sort_values("Reports", ascending=False)
        )
        st.dataframe(stats, use_container_width=True, hide_index=True)

    # ---- TAB 3 ----
with tab3:
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, f"{acc:.3f}",  "Accuracy",  PALETTE[0])
    kpi(c2, f"{prec:.3f}", "Precision", PALETTE[1])
    kpi(c3, f"{rec:.3f}",  "Recall",    PALETTE[2])
    kpi(c4, f"{f1:.3f}",   "F1 Score",  PALETTE[5])

    st.markdown("### Confusion Matrix")
    labels = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels, colorscale="Blues",
        text=cm, texttemplate="%{text}", textfont={"size": 12},
        colorbar=dict(title="Count"),
    ))
    fig.update_layout(title="Confusion Matrix (Test Set)",
                      xaxis_title="Predicted", yaxis_title="Actual",
                      height=500, plot_bgcolor="white")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

