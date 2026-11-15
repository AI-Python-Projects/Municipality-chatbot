# Municipality-chatbot

An **automated machine-learning dashboard** that analyses municipal service
reports — routing each complaint to the correct department, scoring priority,
and projecting SLA response times. **No user input required** — the dashboard
loads the dataset, runs ML, and renders tables and charts.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.25%2B-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Municipal call centres receive thousands of complaints daily. Manual routing is
slow and inconsistent. This project replaces it with a **Random Forest
classifier** trained on **1,100 historical reports**:

- **Classification** — routes each report to one of 9 departments
- **Prioritisation** — deterministic scoring from severity, area, reports
- **SLA projection** — derived from historical resolution times
- **Analytics** — 5 dashboard tabs with 15+ tables and charts

Everything is **fully automated** — no prompts, no forms, just data in, insights out.

---

## 📊 What you see

The Streamlit dashboard has **5 tabs**:

| Tab                  | Contents                                                                                                                    |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Overview**         | Dataset summary, sample rows, status/reporter/priority breakdown, top cities                                                |
| **Distributions**    | Department, issue type, severity, area, reports histogram, department stats table                                           |
| **ML Performance**   | Accuracy / precision / recall / F1 metrics, confusion matrix, per-class scores, feature importance, confidence distribution |
| **SLA Analysis**     | SLA heatmap (dept × priority), resolution boxplot, avg resolution by department                                             |
| **Full Predictions** | Complete ML predictions table + download button + misrouted rows                                                            |

---

## 🚀 Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Add dataset
#    place municipal_training_set_1100.csv in data/

# 3. Train
python train.py

# 4. Evaluate
python test.py

# 5. Launch dashboard
streamlit run app.py
```
