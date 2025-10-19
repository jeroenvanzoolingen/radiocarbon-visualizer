import streamlit as st
import pandas as pd
import numpy as np
import re
import pdfplumber
import plotly.graph_objects as go
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
import tempfile
import os

st.set_page_config(page_title="Radiocarbon Visualizer – verbeterde versie", layout="wide")

st.title("📆 Radiocarbon Dating Visualizer (gekalibreerd, BC/AD-schaal)")

st.markdown("""
Upload laboratoriumdata of voer handmatig waarden in.  
De app toont de **gekalibreerde posterioren** van elk monster samen met de **IntCal20-band (2σ)**,  
automatisch geschaald op het relevante tijdsinterval.
""")

# ---------- Helpers ----------
def parse_bp_value(value):
    if not value:
        return None, None
    match = re.match(r"(\d+)\s*[±+\-/]*\s*(\d+)", str(value))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def parse_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    pattern = re.compile(r"(\S+)\s+(\S+)\s+(\d+\s*[±\+-]\s*\d+)\s*BP", re.IGNORECASE)
    rows = []
    for m in pattern.finditer(text):
        rows.append({
            "Sample name": m.group(1),
            "Lab. no.": m.group(2),
            "14C date (BP)": m.group(3) + " BP"
        })
    return pd.DataFrame(rows)

@st.cache_data
def load_intcal_curve(path="intcal20.csv"):
    df = pd.read_csv(path)
    df = df.sort_values("calBP").reset_index(drop=True)
    # Smooth IntCal over 50 jaar voor visuele rust
    df["mu14C_smooth"] = df["mu14C"].rolling(window=50, min_periods=1, center=True).mean()
    df["sigma_curve_smooth"] = df["sigma_curve"].rolling(window=50, min_periods=1, center=True).mean()
    return df

intcal_df = load_intcal_curve()

def calibrate_posterior(bp_measured, sigma_lab, intcal_df):
    """Bereken posterior P(calBP | 14C_metingen)."""
    mu = intcal_df["mu14C_smooth"].values
    sigma_curve = intcal_df["sigma_curve_smooth"].values
    calBP = intcal_df["calBP"].values
    var = sigma_curve**2 + sigma_lab**2
    L = np.exp(-0.5 * (bp_measured - mu)**2 / var) / np.sqrt(2 * np.pi * var)
    posterior = L / np.trapz(L, calBP)
    return calBP, posterior

def bp_to_ad(bp_values):
    """Zet calBP om naar AD-jaar (negatief = BC)."""
    return 1950 - np.array(bp_values)

def format_bc_ad(ad_year):
    """Formateer jaartal voor as-label."""
    if ad_year < 0:
        return f"{abs(int(ad_year))} BC"
    elif ad_year == 0:
        return "0 AD"
    else:
        return f"{int(ad_year)} AD"

# ---------- Datasetbeheer ----------
if "data" not in st.session_state:
    st.session_state["data"] = pd.DataFrame(columns=["Sample name", "Lab. no.", "14C date (BP)", "BP", "Error"])

uploaded_file = st.file_uploader("📄 Upload Excel- of PDF-bestand", type=["xlsx", "xls", "pdf"])
if uploaded_file is not None:
    if uploaded_file.name.lower().endswith(".pdf"):
        df = parse_pdf(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    df["BP"], df["Error"] = zip(*df["14C date (BP)"].map(parse_bp_value))
    st.session_state["data"] = pd.concat([st.session_state["data"], df], ignore_index=True)

st.subheader("✏️ Handmatige invoer")
with st.form("manual_entry", clear_on_submit=True):
    c1, c2, c3 = st.columns(3)
    sample = c1.text_input("Sample name")
    lab = c2.text_input("Lab. no.")
    raw_date = c3.text_input("14C date (BP)", placeholder="bijv. 510 ± 30")
    add = st.form_submit_button("➕ Voeg toe")
    if add and sample and lab and raw_date:
        bp, err = parse_bp_value(raw_date)
        if bp is None or err is None:
            st.warning("Ongeldige invoer — gebruik bijv. 510 ± 30")
        else:
            new = {"Sample name": sample, "Lab. no.": lab,
                   "14C date (BP)": raw_date, "BP": bp, "Error": err}
            st.session_state["data"] = pd.concat(
                [st.session_state["data"], pd.DataFrame([new])], ignore_index=True
            )

data = st.session_state["data"]

st.subheader("📋 Samengestelde dataset")
st.dataframe(data, use_container_width=True)

# ---------- Visualisatie ----------
if not data.empty:
    st.subheader("📊 Gekalibreerde waarschijnlijkheidsverdelingen")

    fig = go.Figure()

    # ---- IntCal-band (2σ) ----
    calBP = intcal_df["calBP"].values
    AD_years = bp_to_ad(calBP)
    mu = intcal_df["mu14C_smooth"].values
    sigma = intcal_df["sigma_curve_smooth"].values

    upper = mu + 2 * sigma
    lower = mu - 2 * sigma

    fig.add_trace(go.Scatter(
        x=AD_years, y=upper,
        line=dict(color="rgba(0,0,255,0)"), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=AD_years, y=lower,
        fill="tonexty", fillcolor="rgba(0,100,255,0.25)",
        line=dict(color="rgba(0,0,255,0)"),
        name="IntCal20 ±2σ"
    ))

    # ---- Posterioren ----
    offset_step = (upper.max() - lower.min()) * 0.10
    ad_min, ad_max = None, None

    for idx, row in data.iterrows():
        if pd.notna(row["BP"]) and pd.notna(row["Error"]):
            cal_grid, post = calibrate_posterior(row["BP"], row["Error"], intcal_df)
            ad_grid = bp_to_ad(cal_grid)
            post_scaled = post / post.max() * offset_step
            offset = lower.min() - (idx + 1) * offset_step * 1.2
            fig.add_trace(go.Scatter(
                x=ad_grid, y=post_scaled + offset, mode="lines",
                line=dict(width=2), name=row["Sample name"]
            ))

            # Bepaal automatisch datumbereik
            if ad_min is None or ad_grid.min() < ad_min:
                ad_min = ad_grid.min()
            if ad_max is None or ad_grid.max() > ad_max:
                ad_max = ad_grid.max()

    # ---- Automatische x-as op basis van data ----
    if ad_min is not None and ad_max is not None:
        left = np.floor((ad_min - 100) / 100) * 100
        right = np.ceil((ad_max + 100) / 100) * 100
    else:
        left, right = 1950 - 6000, 1950

    tick_vals = np.arange(left, right + 100, 100)
    tick_texts = [format_bc_ad(t) for t in tick_vals]

    fig.update_layout(
        xaxis=dict(title="Kalenderjaren (BC/AD)",
                   range=[left, right],
                   tickmode="array", tickvals=tick_vals, ticktext=tick_texts),
        yaxis=dict(title="Posterior-dichtheid / 14C (BP)", showticklabels=False),
        template="simple_white",
        height=500 + len(data) * 80,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------- Download CSV ----------
if not data.empty:
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download dataset (CSV)",
                       data=csv, file_name="radiocarbon_data.csv",
                       mime="text/csv")

st.markdown("""
---
✅ *Posterioren nu correct op BC/AD-schaal.*  
📏 *X-as automatisch geschaald op relevante datering + 100 jaar buffer.*  
🌊 *IntCal20-band vereenvoudigd tot vloeiende 2σ-band (geen lijn).*
""")
