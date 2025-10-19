import streamlit as st
import pandas as pd
import numpy as np
import re
import pdfplumber
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Radiocarbon Visualizer w/ IntCal", layout="wide")

st.title("📆 Radiocarbon Dating Tool met Kalibratie (IntCal20)")

st.markdown("""
Upload je data (Excel of PDF) of voer handmatig waarden in.  
De app zal de posterior densiteitscurves tonen op basis van IntCal20.
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
    # Pas dit pattern aan afhankelijk van je PDF-format
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
    # Zorg dat de kolomnamen kloppen
    # eventueel: df = df.rename(...)
    return df

def calibrate_posterior(bp, sigma_lab, intcal_df):
    """
    Bereken posterior densiteit over kalenderjaren (in calBP) gegeven een meting (bp, sigma_lab).
    Intcal_df moet kolommen calBP, mu14C, sigma_curve hebben.
    Retourneer een array posterior en de corresponderende calBP.
    """
    # Verband: variantie = σ_curve^2 + σ_lab^2
    mu = intcal_df["mu14C"].values
    sigma_curve = intcal_df["sigma_curve"].values
    calBP = intcal_df["calBP"].values

    var = sigma_curve**2 + sigma_lab**2
    # Likelihood
    L = np.exp(-0.5 * (bp - mu)**2 / var) / np.sqrt(2 * np.pi * var)
    # Posterior (normeren)
    posterior = L / np.sum(L)
    return calBP, posterior

def highest_posterior_interval(calBP, posterior, cred_mass=0.95):
    """
    Bepaal interval waarin de posterior een cumulatieve massa van cred_mass dekt.
    Simpel: rangschik posterior in aflopende volgorde tot som >= cred_mass.
    Retourneer min, max calBP van dat interval.
    """
    # Sort indices op posterior aflopend
    idx = np.argsort(posterior)[::-1]
    cumulative = np.cumsum(posterior[idx])
    # bepaal hoeveel indices nodig
    cutoff = idx[cumulative <= cred_mass]
    # de indices inbegrepen
    chosen = idx[cumulative <= cred_mass]
    # neem min en max calBP van die gekozen
    min_bp = calBP[chosen].min()
    max_bp = calBP[chosen].max()
    return min_bp, max_bp

# ---------- Laad IntCal data ----------

@st.cache_data
def get_intcal():
    return load_intcal_curve()

intcal_df = get_intcal()

# ---------- Upload of handmatige invoer ----------

uploaded_file = st.file_uploader("Upload Excel of PDF", type=["xlsx", "xls", "pdf"])

data = pd.DataFrame(columns=[
    "Sample name", "Lab. no.", "14C date (BP)", "BP", "Error"
])

if uploaded_file is not None:
    if uploaded_file.name.lower().endswith(".pdf"):
        df = parse_pdf(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    df["BP"], df["Error"] = zip(*df["14C date (BP)"].map(parse_bp_value))
    data = pd.concat([data, df], ignore_index=True)

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
            st.warning("Ongeldige invoer, gebruik bv. ‘510 ± 30’")
        else:
            new = {"Sample name": sample, "Lab. no.": lab, "14C date (BP)": raw_date, "BP": bp, "Error": err}
            data = pd.concat([data, pd.DataFrame([new])], ignore_index=True)

# ---------- Visualisatie van posterior densiteiten ----------

st.subheader("📊 Posterior kalibratiecurves")

if not data.empty:
    fig = go.Figure()

    # optioneel: plot de IntCal-curve (µ(t)) als blauwe lijn
    fig.add_trace(go.Scatter(
        x=intcal_df["calBP"],
        y=intcal_df["mu14C"],
        mode="lines",
        name="IntCal20 μ(¹⁴C)",
        line=dict(color="blue", width=1),
        yaxis="y2"
    ))

    # Voor elke monster: bereken posterior en plot densiteit
    for idx, row in data.iterrows():
        if pd.notna(row["BP"]) and pd.notna(row["Error"]):
            calBP_arr, post = calibrate_posterior(row["BP"], row["Error"], intcal_df)
            # opschalen densiteit naar een geschikte amplitude (optioneel)
            # je kunt post * schaalfactor doen om zichtbaar te maken
            scale = 0.8  # aanpasbaar
            fig.add_trace(go.Scatter(
                x=calBP_arr,
                y=post * scale + idx,  # verschuif op y-positie idx
                mode="lines",
                name=f"{row['Sample name']}"
            ))
            # bepaal 95% interval
            lo, hi = highest_posterior_interval(calBP_arr, post, cred_mass=0.95)
            fig.add_trace(go.Scatter(
                x=[lo, hi],
                y=[idx + scale, idx + scale],
                mode="lines",
                line=dict(color="black", width=4),
                showlegend=False
            ))

    # Layout: dubbele y-as voor mu(¹⁴C)
    fig.update_layout(
        xaxis_title="Calibration year (cal BP)",
        yaxis_title="Samples (verschuiving voor densiteitscurves)",
        legend=dict(orientation="h"),
        yaxis=dict(showticklabels=False),
        yaxis2=dict(
            title="¹⁴C (BP) curve",
            overlaying="y",
            side="right"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------- Download CSV-output ----------

if not data.empty:
    out = data.to_csv(index=False).encode("utf-8")
    st.download_button("Download dataset als CSV", data=out, file_name="radiocarbon_data.csv", mime="text/csv")

st.markdown("""
---  
*Deze implementatie gebruikt IntCal20 om posterior densiteitscurves te berekenen.  
De y-positie van elke curve is verschoven (verticale offset) om overlapping te vermijden.  
Legenda: de blauwe lijn is de IntCal µ(¹⁴C) curve.*  
""")
