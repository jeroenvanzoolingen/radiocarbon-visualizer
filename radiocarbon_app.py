# radiocarbon_app.py
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

st.set_page_config(page_title="Radiocarbon Visualizer – calibrated ranges", layout="wide")

st.title("📆 Radiocarbon Visualizer — gebruik 'Gekalibreerd (2σ)' voor x-as")
st.markdown("""
Upload je Excel of PDF. Als je bestand een kolom **`Gekalibreerd (2σ)`** bevat,
dan gebruikt de app die waarden als (BC/AD) kalibratieranges en stelt de x-as automatisch in.
""")

# ----------------- Helpers -----------------
def parse_bp_value(value):
    if not value or (isinstance(value, float) and np.isnan(value)):
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

def parse_calibrated_range(val):
    if pd.isna(val):
        return None, None
    s = str(val).strip()
    s = s.replace("–", "-").replace("—", "-").replace(" to ", "-").replace("TO", "-")
    s = re.sub(r"\(.*?\)", "", s)
    s = s.replace("calBP", "").replace("cal BP", "").strip()
    tokens = re.findall(r"(-?\d+)\s*(BC|AD|BCE|CE)?", s, flags=re.IGNORECASE)
    if len(tokens) >= 2:
        def to_ad(tok):
            num_str, era = tok
            num = int(num_str)
            if era:
                era = era.upper()
                if era in ("BC", "BCE"):
                    return -abs(num)
                else:
                    return num
            else:
                return num
        a = to_ad(tokens[0])
        b = to_ad(tokens[1])
        start, end = (a, b) if a <= b else (b, a)
        return int(start), int(end)
    parts = s.split("-")
    if len(parts) >= 2:
        def parse_part(p):
            m = re.search(r"(-?\d+)", p)
            if not m:
                return None
            num = int(m.group(1))
            if re.search(r"BC|BCE", p, flags=re.IGNORECASE):
                return -abs(num)
            else:
                return num
        p0 = parse_part(parts[0])
        p1 = parse_part(parts[1])
        if p0 is not None and p1 is not None:
            start, end = (p0, p1) if p0 <= p1 else (p1, p0)
            return int(start), int(end)
    return None, None

@st.cache_data
def load_intcal(path="intcal20.csv"):
    df = pd.read_csv(path)
    df = df.sort_values("calBP").reset_index(drop=True)
    df["mu14C_smooth"] = df["mu14C"].rolling(window=50, min_periods=1, center=True).mean()
    df["sigma_curve_smooth"] = df["sigma_curve"].rolling(window=50, min_periods=1, center=True).mean()
    return df

intcal_df = load_intcal()

def calibrate_posterior(bp_measured, sigma_lab, intcal_df, use_smooth=True):
    if use_smooth:
        mu = intcal_df["mu14C_smooth"].values
        sigma_curve = intcal_df["sigma_curve_smooth"].values
    else:
        mu = intcal_df["mu14C"].values
        sigma_curve = intcal_df["sigma_curve"].values
    calBP = intcal_df["calBP"].values
    var = sigma_curve**2 + sigma_lab**2
    L = np.exp(-0.5 * (bp_measured - mu)**2 / var) / np.sqrt(2 * np.pi * var)
    posterior = L / np.trapz(L, calBP)
    return calBP, posterior

def bp_to_ad(vals):
    return 1950 - np.array(vals)

def format_bc_ad(y):
    if y < 0:
        return f"{abs(int(y))} BC"
    elif y == 0:
        return "0 AD"
    else:
        return f"{int(y)} AD"

# ----------------- Session dataset -----------------
if "data" not in st.session_state:
    st.session_state["data"] = pd.DataFrame(columns=[
        "Sample name", "Lab. no.", "14C date (BP)", "BP", "Error", "Gekalibreerd (2σ)"
    ])

# ----------------- Upload / append -----------------
uploaded_file = st.file_uploader("Upload Excel of PDF", type=["xlsx", "xls", "pdf"])
if uploaded_file is not None:
    if uploaded_file.name.lower().endswith(".pdf"):
        df = parse_pdf(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    if "14C date (BP)" in df.columns:
        df["BP"], df["Error"] = zip(*df["14C date (BP)"].map(parse_bp_value))
    else:
        df["BP"], df["Error"] = None, None
    if "Gekalibreerd (2σ)" not in df.columns:
        df["Gekalibreerd (2σ)"] = None
    st.session_state["data"] = pd.concat([st.session_state["data"], df], ignore_index=True)

# ----------------- Manual multiple-entry form -----------------
st.subheader("Handmatige invoer (meerdere regels mogelijk)")
with st.form("manual", clear_on_submit=True):
    c1, c2, c3, c4 = st.columns([2,2,2,3])
    sample = c1.text_input("Sample name")
    lab = c2.text_input("Lab. no.")
    raw_bp = c3.text_input("14C date (BP)", placeholder="bijv. 510 ± 30")
    calib = c4.text_input("Gekalibreerd (2σ) (optioneel)", placeholder="bijv. 393 BC - 70 AD")
    submitted = st.form_submit_button("Voeg toe")
    if submitted:
        bp, err = parse_bp_value(raw_bp)
        new = {"Sample name": sample or "", "Lab. no.": lab or "",
               "14C date (BP)": raw_bp or "", "BP": bp, "Error": err,
               "Gekalibreerd (2σ)": calib or None}
        st.session_state["data"] = pd.concat([st.session_state["data"], pd.DataFrame([new])], ignore_index=True)

data = st.session_state["data"]

st.subheader("Samengestelde dataset")
st.dataframe(data, use_container_width=True)

use_calibrated_col = "Gekalibreerd (2σ)" in data.columns and data["Gekalibreerd (2σ)"].notna().any()
show_intcal = st.checkbox("Toon IntCal ±2σ-band", value=True)
overlay_posteriors = st.checkbox("Overlay posterior wiggles (berekend uit BP en IntCal)", value=False)

cal_ranges = []
for idx, row in data.iterrows():
    rng = None
    if use_calibrated_col:
        rng = parse_calibrated_range(row.get("Gekalibreerd (2σ)"))
    cal_ranges.append(rng)

ad_min, ad_max = None, None
if use_calibrated_col:
    for rng in cal_ranges:
        if rng and rng[0] is not None:
            a, b = rng
            if ad_min is None or a < ad_min:
                ad_min = a
            if ad_max is None or b > ad_max:
                ad_max = b

if ad_min is None or ad_max is None:
    calBP = intcal_df["calBP"].values
    ad_vals = bp_to_ad(calBP)
    ad_min = ad_vals.min()
    ad_max = ad_vals.max()

left = int(np.floor((ad_min - 100) / 100.0) * 100)
right = int(np.ceil((ad_max + 100) / 100.0) * 100)
tick_vals = np.arange(left, right + 100, 100)
tick_texts = [format_bc_ad(v) for v in tick_vals]

if not data.empty:
    st.subheader("Visualisatie (gebruik 'Gekalibreerd (2σ)')")
    fig = go.Figure()
    if show_intcal:
        calBP = intcal_df["calBP"].values
        ad_vals = bp_to_ad(calBP)
        mu = intcal_df["mu14C_smooth"].values
        sigma = intcal_df["sigma_curve_smooth"].values
        upper = mu + 2 * sigma
        lower = mu - 2 * sigma
        fig.add_trace(go.Scatter(x=ad_vals, y=upper,
                                 line=dict(color='rgba(0,0,255,0)'), hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Scatter(x=ad_vals, y=lower, fill='tonexty',
                                 fillcolor='rgba(0,100,255,0.25)',
                                 line=dict(color='rgba(0,0,255,0)'), name='IntCal20 ±2σ'))

    y_positions = []
    y_labels = []
    for idx, row in data.iterrows():
        y = len(data) - idx
        y_positions.append(y)
        y_labels.append(row.get("Sample name", f"sample_{idx}"))
        rng = cal_ranges[idx] if idx < len(cal_ranges) else None
        if rng and rng[0] is not None:
            start_ad, end_ad = rng
            fig.add_trace(go.Scatter(
                x=[start_ad, end_ad], y=[y, y], mode="lines",
                line=dict(color="firebrick", width=8),
                name=row.get("Sample name", ""),
                hovertemplate=f"%{{x}}<br>{row.get('Sample name','')}"
            ))
            mid = (start_ad + end_ad) / 2.0
            fig.add_trace(go.Scatter(
                x=[mid], y=[y], mode="markers+text",
                marker=dict(color="black", size=6),
                text=[row.get("Sample name","")],
                textposition="middle right",  # ✅ fix
                showlegend=False, hoverinfo="skip"
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[None], y=[y], mode="markers+text", text=[row.get("Sample name","")],
                textposition="middle right", showlegend=False, hoverinfo="skip"
            ))

    if overlay_posteriors:
        for idx, row in data.iterrows():
            if pd.notna(row.get("BP")) and pd.notna(row.get("Error")):
                cal_grid, post = calibrate_posterior(row["BP"], row["Error"], intcal_df)
                ad_grid = bp_to_ad(cal_grid)
                y = len(data) - idx
                amp = 0.4
                post_scaled = post / post.max() * amp
                fig.add_trace(go.Scatter(
                    x=ad_grid, y=post_scaled + y - 0.3, mode="lines",
                    line=dict(width=1), name=f"{row.get('Sample name','')}_posterior",
                    showlegend=False
                ))

    fig.update_yaxes(tickmode="array", tickvals=y_positions, ticktext=y_labels, autorange="reversed")
    fig.update_layout(
        xaxis=dict(title="Kalenderjaren (BC/AD)", range=[left, right],
                   tickmode="array", tickvals=tick_vals, ticktext=tick_texts),
        yaxis=dict(title="", showgrid=False),
        height=150 + 60 * len(data), template="simple_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
