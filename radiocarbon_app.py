import streamlit as st
import pandas as pd
import re
import pdfplumber
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Radiocarbon Visualizer", layout="wide")

st.title("📆 Radiocarbon Dating Data Tool")

st.markdown("""
Upload laboratoriumdata (Excel of PDF) of voer handmatig waarden in.  
De app combineert alles, toont de data, en maakt een visualisatie van de dateringen.
""")

# ---------- Hulpfuncties ----------
def parse_bp_value(value):
    """Herken diverse notaties zoals ±, +-, en +/-"""
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

def simulate_calibration(bp, error):
    """Placeholder: 68% = ±1σ, 95% = ±2σ"""
    if pd.isna(bp) or pd.isna(error):
        return None
    return {
        "Cal68_from": bp - error,
        "Cal68_to": bp + error,
        "Cal95_from": bp - 2*error,
        "Cal95_to": bp + 2*error
    }

# ---------- Upload ----------
uploaded_file = st.file_uploader("📄 Upload Excel- of PDF-bestand", type=["xlsx", "xls", "pdf"])

data = pd.DataFrame(columns=[
    "Sample name", "Lab. no.", "14C date (BP)", "BP", "Error",
    "Cal68_from", "Cal68_to", "Cal95_from", "Cal95_to", "Calibrated (cal BP or BC/AD)"
])

if uploaded_file is not None:
    if uploaded_file.name.lower().endswith(".pdf"):
        df = parse_pdf(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df["BP"], df["Error"] = zip(*df["14C date (BP)"].map(parse_bp_value))
    df["Calibrated (cal BP or BC/AD)"] = ""

    cals = df.apply(lambda r: simulate_calibration(r["BP"], r["Error"]), axis=1)
    for c in ["Cal68_from", "Cal68_to", "Cal95_from", "Cal95_to"]:
        df[c] = [d[c] if d else None for d in cals]

    data = pd.concat([data, df], ignore_index=True)

# ---------- Handmatige invoer ----------
st.subheader("✏️ Handmatige invoer")

with st.form("manual_entry"):
    c1, c2, c3 = st.columns(3)
    sample = c1.text_input("Sample name")
    lab = c2.text_input("Lab. no.")
    raw_date = c3.text_input("14C date (BP)", placeholder="bijv. 510 ± 30 BP")
    add = st.form_submit_button("Toevoegen aan dataset")

    if add and sample and lab and raw_date:
        bp, err = parse_bp_value(raw_date)
        if bp is None or err is None:
            st.warning("❗ Ongeldige invoer — gebruik bijvoorbeeld: 510 ± 30 BP")
        else:
            cal = simulate_calibration(bp, err)
            new_row = pd.DataFrame([{
                "Sample name": sample,
                "Lab. no.": lab,
                "14C date (BP)": raw_date,
                "BP": bp,
                "Error": err,
                "Cal68_from": cal["Cal68_from"],
                "Cal68_to": cal["Cal68_to"],
                "Cal95_from": cal["Cal95_from"],
                "Cal95_to": cal["Cal95_to"],
                "Calibrated (cal BP or BC/AD)": ""
            }])
            data = pd.concat([data, new_row], ignore_index=True)

# ---------- Dataset tonen ----------
st.subheader("📋 Samengestelde dataset")
st.dataframe(data, use_container_width=True)

# ---------- Visualisatie ----------
if not data.empty and "BP" in data:
    st.subheader("📊 Visualisatie (gesimuleerde intervallen)")

    fig = go.Figure()

    for idx, row in data.iterrows():
        if pd.notna(row["BP"]) and pd.notna(row["Error"]):
            y = row["Sample name"]
            fig.add_trace(go.Scatter(
                x=[row["Cal95_from"], row["Cal95_to"]],
                y=[y, y],
                mode="lines",
                line=dict(color="rgba(150,150,150,0.6)", width=6),
                name="95%"
            ))
            fig.add_trace(go.Scatter(
                x=[row["Cal68_from"], row["Cal68_to"]],
                y=[y, y],
                mode="lines",
                line=dict(color="rgba(50,50,200,0.9)", width=10),
                name="68%"
            ))
            fig.add_trace(go.Scatter(
                x=[row["BP"]],
                y=[y],
                mode="markers",
                marker=dict(color="red", size=8),
                name="BP"
            ))

    fig.update_layout(
        xaxis=dict(title="Years BP", autorange="reversed"),
        yaxis=dict(title="Sample name"),
        showlegend=False,
        height=400 + len(data)*20,
        template="simple_white"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------- CSV-export ----------
if not data.empty:
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 Download CSV",
        data=csv,
        file_name="radiocarbon_data.csv",
        mime="text/csv"
    )

st.info("💡 Grafiek-export (PNG/SVG) werkt alleen bij lokaal gebruik; op Streamlit Cloud is dit uitgeschakeld.")

st.markdown("""
---
🧪 *Toekomstige uitbreiding:* echte OxCal-kalibratie (vervangt de gesimuleerde intervallen)*  
📤 *Opmerking:* grafiek-export is alleen beschikbaar bij lokaal gebruik (niet op Streamlit Cloud).*
""")
