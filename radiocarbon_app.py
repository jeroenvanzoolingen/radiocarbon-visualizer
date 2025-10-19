import streamlit as st
import pandas as pd
import numpy as np
import re
import pdfplumber
import plotly.graph_objects as go

st.set_page_config(page_title="Radiocarbon Visualizer – OxCal-stijl", layout="wide")

st.title("📆 Radiocarbon Dating Visualizer met IntCal-kalibratie (BC/AD-tijdas)")
st.markdown("""
Upload je data (Excel of PDF) of voer handmatig waarden in.  
De app toont de **IntCal20-kalibratiecurve** (blauw) en de **gekalibreerde waarschijnlijkheidsverdelingen**
voor elk monster, met een BC/AD-tijdas in stappen van 100 jaar.
""")

# ---------- Parsing helpers ----------
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

# ---------- IntCal-data ----------
@st.cache_data
def load_intcal_curve(path="intcal20.csv"):
    df = pd.read_csv(path)
    df = df.sort_values("calBP").reset_index(drop=True)
    return df

intcal_df = load_intcal_curve()

# ---------- Posteriorberekening ----------
def calibrate_posterior(bp_measured, sigma_lab, intcal_df):
    mu = intcal_df["mu14C"].values
    sigma_curve = intcal_df["sigma_curve"].values
    calBP = intcal_df["calBP"].values
    var = sigma_curve**2 + sigma_lab**2
    L = np.exp(-0.5 * (bp_measured - mu)**2 / var) / np.sqrt(2 * np.pi * var)
    posterior = L / np.trapz(L, calBP)
    return calBP, posterior

# ---------- Helper: BP→AD (BC/AD) ----------
def bp_to_ad(bp_values):
    return 1950 - np.array(bp_values)

def format_bc_ad(ad_year):
    if ad_year < 0:
        return f"{abs(int(ad_year))} BC"
    elif ad_year == 0:
        return "0 AD"
    else:
        return f"{int(ad_year)} AD"

# ---------- Upload & invoer ----------
uploaded_file = st.file_uploader("📄 Upload Excel- of PDF-bestand", type=["xlsx", "xls", "pdf"])
data = pd.DataFrame(columns=["Sample name", "Lab. no.", "14C date (BP)", "BP", "Error"])

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
            st.warning("Ongeldige invoer — gebruik bijvoorbeeld: 510 ± 30")
        else:
            new = {"Sample name": sample, "Lab. no.": lab,
                   "14C date (BP)": raw_date, "BP": bp, "Error": err}
            data = pd.concat([data, pd.DataFrame([new])], ignore_index=True)

# ---------- Tabel ----------
st.subheader("📋 Samengestelde dataset")
st.dataframe(data, use_container_width=True)

# ---------- Visualisatie ----------
if not data.empty:
    st.subheader("📊 Gekalibreerde waarschijnlijkheidsverdelingen (OxCal-stijl)")

    fig = go.Figure()

    # ---- IntCal-curve ----
    calBP = intcal_df["calBP"].values
    AD_years = bp_to_ad(calBP)
    mu = intcal_df["mu14C"].values
    sigma = intcal_df["sigma_curve"].values

    fig.add_trace(go.Scatter(
        x=AD_years, y=mu + sigma,
        line=dict(color="rgba(0,0,255,0)"), hoverinfo="skip", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=AD_years, y=mu - sigma,
        fill="tonexty",
        fillcolor="rgba(0,100,255,0.25)",
        line=dict(color="rgba(0,0,255,0)"),
        name="IntCal20 ±1σ"
    ))
    fig.add_trace(go.Scatter(
        x=AD_years, y=mu,
        line=dict(color="blue", width=1),
        name="IntCal20 μ(¹⁴C)"
    ))

    # ---- Posterior-wiggles per sample ----
    offset_step = (mu.max() - mu.min()) * 0.15
    for idx, row in data.iterrows():
        if pd.notna(row["BP"]) and pd.notna(row["Error"]):
            cal_grid, post = calibrate_posterior(row["BP"], row["Error"], intcal_df)
            ad_grid = bp_to_ad(cal_grid)
            post_scaled = post / post.max() * offset_step
            offset = mu.min() - (idx + 1) * offset_step * 1.4
            fig.add_trace(go.Scatter(
                x=ad_grid,
                y=post_scaled + offset,
                mode="lines",
                line=dict(width=2),
                name=row["Sample name"]
            ))

    # ---- Bereken as-limieten en ticks ----
    ad_min = 1950 - intcal_df["calBP"].max() - 50
    ad_max = 1950 + 50
    tick_vals = np.arange(ad_min // 100 * 100, ad_max + 100, 100)
    tick_texts = [format_bc_ad(y) for y in tick_vals]

    # ---- Layout ----
    fig.update_layout(
        xaxis=dict(
            title="Kalenderjaren (BC/AD)",
            range=[ad_min, ad_max],
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_texts
        ),
        yaxis=dict(
            title="14C-waarde (BP) en posterior-dichtheid",
            showticklabels=False
        ),
        template="simple_white",
        height=500 + len(data)*80,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------- Download ----------
if not data.empty:
    out = data.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download dataset als CSV", data=out,
                       file_name="radiocarbon_data.csv", mime="text/csv")

st.markdown("""
---
🧪 *De tijdas toont nu kalenderjaren (BC/AD) in stappen van 100 jaar,  
met ±50 jaar marge voor een prettigere layout.*
""")
