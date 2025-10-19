import streamlit as st
import pandas as pd
import numpy as np
import re
import pdfplumber
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Radiocarbon Visualizer w/ IntCal", layout="wide")

st.title("📆 Radiocarbon Dating Tool met IntCal20-kalibratie")

st.markdown("""
Upload je data (Excel of PDF) of voer handmatig waarden in.  
De app toont de gekalibreerde posteriorcurves samen met de IntCal20-curve (blauwe band).
""")

# ---------- Hulpfuncties ----------

def parse_bp_value(value):
    """Herken notaties zoals '510 ± 30' of '510+/-30'."""
    if not value:
        return None, None
    match = re.match(r"(\d+)\s*[±+\-/]*\s*(\d+)", str(value))
    if match:
        bp = int(match.group(1))
        error = int(match.group(2))
        return bp, error
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

def load_intcal_curve(path="intcal20.csv"):
    """Laad IntCal20 CSV met kolommen: calBP, mu14C, sigma_curve."""
    df = pd.read_csv(path)
    return df

def calibrate_posterior(bp, sigma_lab, intcal_df):
    """
    Bereken posterior densiteit over kalenderjaren (in calBP)
    gegeven een meting (bp, sigma_lab) en IntCal20-curve.
    """
    mu = intcal_df["mu14C"].values
    sigma_curve = intcal_df["sigma_curve"].values
    calBP = intcal_df["calBP"].values
    var = sigma_curve**2 + sigma_lab**2
    L = np.exp(-0.5 * (bp - mu)**2 / var) / np.sqrt(2 * np.pi * var)
    posterior = L / np.sum(L)
    return calBP, posterior

# ---------- Laad IntCal data ----------

@st.cache_data
def get_intcal():
    return load_intcal_curve()

intcal_df = get_intcal()

# ---------- Upload of handmatige invoer ----------

uploaded_file = st.file_uploader("📄 Upload Excel- of PDF-bestand", type=["xlsx", "xls", "pdf"])

data = pd.DataFrame(columns=["Sample name", "Lab. no.", "14C date (BP)", "BP", "Error"])

if uploaded_file is not None:
    if uploaded_file.name.lower().endswith(".pdf"):
        df = parse_pdf(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    df["BP"], df["Error"] = zip(*df["14C date (BP)"].map(parse_bp_value))
    data = pd.concat([data, df], ignore_index=True)

# ---------- Handmatige invoer ----------

st.subheader("✏️ Handmatige invoer")
with st.form("manual_entry"):
    c1, c2, c3 = st.columns(3)
    sample = c1.text_input("Sample name")
    lab = c2.text_input("Lab. no.")
    raw_date = c3.text_input("14C date (BP)", placeholder="bijv. 510 ± 30")
    add = st.form_submit_button("Voeg toe")
    if add and sample and lab and raw_date:
        bp, err = parse_bp_value(raw_date)
        if bp is None or err is None:
            st.warning("Ongeldige invoer — gebruik bijvoorbeeld: 510 ± 30")
        else:
            new = {"Sample name": sample, "Lab. no.": lab,
                   "14C date (BP)": raw_date, "BP": bp, "Error": err}
            data = pd.concat([data, pd.DataFrame([new])], ignore_index=True)

# ---------- Tabel tonen ----------

st.subheader("📋 Samengestelde dataset")
st.dataframe(data, use_container_width=True)

# ---------- Visualisatie ----------

if not data.empty:
    st.subheader("📊 Gekalibreerde posteriors en IntCal20-curve")

    fig = go.Figure()

    # ---- IntCal20 band (±1σ) ----
    mu = intcal_df["mu14C"]
    sigma = intcal_df["sigma_curve"]
    calBP = intcal_df["calBP"]

    # bovenkant van de band
    fig.add_trace(go.Scatter(
        x=calBP,
        y=mu + sigma,
        line=dict(color="rgba(0,0,255,0)"),
        showlegend=False,
        hoverinfo="skip"
    ))
    # onderkant van de band, gevuld
    fig.add_trace(go.Scatter(
        x=calBP,
        y=mu - sigma,
        fill='tonexty',
        fillcolor="rgba(0,100,255,0.2)",
        line=dict(color="rgba(0,0,255,0)"),
        name="IntCal20 ±1σ"
    ))
    # centrale lijn
    fig.add_trace(go.Scatter(
        x=calBP,
        y=mu,
        line=dict(color="blue", width=1),
        name="IntCal20 μ(¹⁴C)"
    ))

    # ---- Posterior curves per sample ----
    for idx, row in data.iterrows():
        if pd.notna(row["BP"]) and pd.notna(row["Error"]):
            calBP_arr, post = calibrate_posterior(row["BP"], row["Error"], intcal_df)
            # schaal en offset voor zichtbaarheid
            scale = 0.4 / post.max()
            y_offset = idx * 200
            fig.add_trace(go.Scatter(
                x=calBP_arr,
                y=post * scale + y_offset,
                mode="lines",
                name=row["Sample name"],
                line=dict(width=2)
            ))

    # ---- Asinstellingen ----
    fig.update_layout(
        xaxis=dict(
            title="Kalenderjaren (cal BP)",
            autorange="reversed",    # verleden links, heden rechts
            range=[6000, 0],         # compacte weergave
            tickmode="linear",
            dtick=500
        ),
        yaxis=dict(
            title="Samples",
            showticklabels=False
        ),
        height=400 + len(data) * 60,
        template="simple_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------- Download CSV ----------

if not data.empty:
    out = data.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download dataset als CSV", data=out,
                       file_name="radiocarbon_data.csv", mime="text/csv")

st.markdown("""
---
🧪 *Deze versie toont de IntCal20-band (blauw, ±1σ) en de posteriorcurves per sample.*  
📈 *Tijdas is omgekeerd (6000 → 0 cal BP) en compacter weergegeven.*
""")
