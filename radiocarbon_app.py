import streamlit as st
import pandas as pd
import numpy as np
import re
import pdfplumber
import plotly.graph_objects as go
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
import tempfile
import os

st.set_page_config(page_title="Radiocarbon Visualizer – PDF export", layout="wide")

st.title("📆 Radiocarbon Visualizer (BC/AD – IntCal + samples + PDF export)")

st.markdown("""
Upload een Excel of PDF met radiokoolstofdata.  
De app toont de **IntCal20-kalibratiecurve** (2σ-band + gemiddelde lijn)  
en de **gekalibreerde dateringen (2σ)** op een correcte BC/AD-schaal.  
Je kunt daarna een PDF-rapport downloaden met grafiek + tabel.
""")

# ---------- Helpers ----------
def parse_bp_value(value):
    if not value or (isinstance(value, float) and np.isnan(value)):
        return None, None
    match = re.match(r"(\d+)\s*[±+\-/]*\s*(\d+)", str(value))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def parse_calibrated_range(val):
    if pd.isna(val):
        return None, None
    s = str(val).strip()
    s = s.replace("–", "-").replace("—", "-").replace(" to ", "-").replace("TO", "-")
    s = re.sub(r"\(.*?\)", "", s)
    tokens = re.findall(r"(-?\d+)\s*(BC|AD)?", s, flags=re.IGNORECASE)
    if len(tokens) >= 2:
        def to_ad(tok):
            num_str, era = tok
            num = int(num_str)
            if era and era.upper() == "BC":
                return -abs(num)
            return num
        a, b = to_ad(tokens[0]), to_ad(tokens[1])
        return (a, b) if a <= b else (b, a)
    return None, None

def format_bc_ad(y):
    if y < 0:
        return f"{abs(int(y))} BC"
    elif y == 0:
        return "0 AD"
    else:
        return f"{int(y)} AD"

@st.cache_data
def load_intcal_curve():
    df = pd.read_csv("intcal20.csv")  # <-- lokaal bestand in repo
    df["mu14C_smooth"] = df["mu14C"].rolling(window=50, center=True, min_periods=1).mean()
    df["sigma_smooth"] = df["sigma_curve"].rolling(window=50, center=True, min_periods=1).mean()
    df["calAD"] = 1950 - df["calBP"]
    return df

intcal_df = load_intcal_curve()

# ---------- Data upload ----------
if "data" not in st.session_state:
    st.session_state["data"] = pd.DataFrame(columns=[
        "Sample name", "Lab. no.", "14C date (BP)", "BP", "Error", "Gekalibreerd (2σ)"
    ])

uploaded_file = st.file_uploader("📄 Upload Excel- of PDF-bestand", type=["xlsx", "xls", "pdf"])
if uploaded_file is not None:
    if uploaded_file.name.lower().endswith(".pdf"):
        st.warning("PDF-parse nog niet volledig geïmplementeerd, gebruik voorlopig Excel.")
    else:
        df = pd.read_excel(uploaded_file)
        if "14C date (BP)" in df.columns:
            df["BP"], df["Error"] = zip(*df["14C date (BP)"].map(parse_bp_value))
        st.session_state["data"] = pd.concat([st.session_state["data"], df], ignore_index=True)

st.subheader("✏️ Handmatige invoer (meerdere regels)")
with st.form("manual", clear_on_submit=True):
    c1, c2, c3, c4 = st.columns([2,2,2,3])
    sample = c1.text_input("Sample name")
    lab = c2.text_input("Lab. no.")
    bp_raw = c3.text_input("14C date (BP)", placeholder="bijv. 510 ± 30")
    cal = c4.text_input("Gekalibreerd (2σ)", placeholder="bijv. 393 BC - 70 AD")
    add = st.form_submit_button("Voeg toe")
    if add:
        bp, err = parse_bp_value(bp_raw)
        new = {"Sample name": sample, "Lab. no.": lab,
               "14C date (BP)": bp_raw, "BP": bp, "Error": err,
               "Gekalibreerd (2σ)": cal}
        st.session_state["data"] = pd.concat([st.session_state["data"], pd.DataFrame([new])], ignore_index=True)

data = st.session_state["data"]
st.dataframe(data, use_container_width=True)

# ---------- Plot ----------
if not data.empty:
    st.subheader("📊 Visualisatie")

    fig = go.Figure()

    # IntCal 2σ-band + lijn
    upper = intcal_df["mu14C_smooth"] + 2 * intcal_df["sigma_smooth"]
    lower = intcal_df["mu14C_smooth"] - 2 * intcal_df["sigma_smooth"]

    fig.add_trace(go.Scatter(
        x=intcal_df["calAD"], y=upper,
        line=dict(color='rgba(0,0,255,0)'), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=intcal_df["calAD"], y=lower,
        fill='tonexty', fillcolor='rgba(0,100,255,0.25)',
        line=dict(color='rgba(0,0,255,0)'), name='IntCal20 ±2σ'
    ))
    fig.add_trace(go.Scatter(
        x=intcal_df["calAD"], y=intcal_df["mu14C_smooth"],
        line=dict(color='blue', width=1.5),
        name='IntCal20 gemiddelde'
    ))

    # Sample-lijnen
    y_positions = list(range(1, len(data)+1))
    for idx, row in enumerate(data.itertuples(), start=1):
        r = parse_calibrated_range(getattr(row, "Gekalibreerd_2σ", None))
        if not r or r == (None, None):
            continue
        start, end = r
        fig.add_trace(go.Scatter(
            x=[start, end], y=[idx, idx],
            mode='lines', line=dict(color='firebrick', width=8),
            name=row.Sample_name
        ))
        fig.add_trace(go.Scatter(
            x=[end + 30], y=[idx], mode='text',
            text=[row.Sample_name],
            textposition='middle right', showlegend=False
        ))

    # Automatische schaal
    cal_min = min([r[0] for r in data["Gekalibreerd (2σ)"].map(parse_calibrated_range) if r != (None, None)], default=-500)
    cal_max = max([r[1] for r in data["Gekalibreerd (2σ)"].map(parse_calibrated_range) if r != (None, None)], default=200)
    left = int(np.floor((cal_min - 100)/100)*100)
    right = int(np.ceil((cal_max + 100)/100)*100)
    ticks = np.arange(left, right+100, 100)
    ticklabels = [format_bc_ad(t) for t in ticks]

    fig.update_layout(
        xaxis=dict(title="Kalenderjaren (BC/AD)", range=[left, right],
                   tickmode="array", tickvals=ticks, ticktext=ticklabels),
        yaxis=dict(title="", showgrid=False, tickvals=y_positions, ticktext=[""]*len(y_positions)),
        height=400 + len(data)*60,
        template="simple_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------- PDF Export ----------
    pdf_buffer = io.BytesIO()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
        fig.write_image(tmp_img.name, format="png", scale=2)
        img_path = tmp_img.name

    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, height-50, "Radiocarbon Report")

    # Plot als afbeelding
    c.drawImage(ImageReader(img_path), 40, 250, width-80, 300, preserveAspectRatio=True)

    # Tabel (alleen de belangrijkste kolommen)
    c.setFont("Helvetica", 10)
    y = 230
    for i, row in data.iterrows():
        line = f"{row.get('Sample name','')} | {row.get('Lab. no.','')} | {row.get('14C date (BP)','')} | {row.get('Gekalibreerd (2σ)','')}"
        c.drawString(40, y, line[:100])  # trunc naar max breedte
        y -= 14
        if y < 40:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height-60
    c.save()
    pdf_buffer.seek(0)
    os.unlink(img_path)

    st.download_button(
        "📄 Download rapport (PDF)",
        data=pdf_buffer,
        file_name="radiocarbon_report.pdf",
        mime="application/pdf"
    )

st.markdown("""
---
✅ IntCal20 wordt nu lokaal ingelezen (geen internet nodig).  
📄 Nieuw: download een PDF-rapport met grafiek + tabel.  
""")
