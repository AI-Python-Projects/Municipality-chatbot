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