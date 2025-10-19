# radiocarbon_app.py
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
from datetime import datetime

st.set_page_config(page_title="Radiocarbon Visualizer (stabiel)", layout="wide")

st.title("📆 Radiocarbon Visualizer — gebruik 'Gekalibreerd (2σ)' voor x-as")
st.markdown("Upload Excel/PDF met kolom `Gekalibreerd (2σ)` of vul handmatig in. De app tekent de kalibratieranges en de IntCal ±2σ band.")

# ---------------- helpers ----------------
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
        rows.append({"Sample name": m.group(1), "Lab. no.": m.group(2), "14C date (BP)": m.group(3) + " BP"})
    return pd.DataFrame(rows)

def parse_calibrated_range(val):
    """Return (start_AD, end_AD) integers or (None,None). Supports multiple notations."""
    if pd.isna(val):
        return None, None
    s = str(val).strip()
    s = s.replace("–", "-").replace("—", "-").replace(" to ", "-").replace("TO", "-")
    s = re.sub(r"\(.*?\)", "", s)
    # Find tokens like "393 BC" or "70 AD" or just numbers
    tokens = re.findall(r"(-?\d+)\s*(BC|BCE|AD|CE)?", s, flags=re.IGNORECASE)
    def to_ad(token):
        num_str, era = token
        num = int(num_str)
        if era and era.upper() in ("BC", "BCE"):
            return -abs(num)
        return num
    if len(tokens) >= 2:
        a = to_ad(tokens[0]); b = to_ad(tokens[1])
        return (a, b) if a <= b else (b, a)
    # fallback: split on '-' and attempt to parse
    if "-" in s:
        parts = s.split("-")
        def parse_part(p):
            m = re.search(r"(-?\d+)", p)
            if not m:
                return None
            num = int(m.group(1))
            if re.search(r"BC|BCE", p, flags=re.IGNORECASE):
                return -abs(num)
            return num
        p0 = parse_part(parts[0]); p1 = parse_part(parts[1])
        if p0 is not None and p1 is not None:
            return (p0, p1) if p0 <= p1 else (p1, p0)
    return None, None

def format_bc_ad(y):
    if y < 0:
        return f"{abs(int(y))} BC"
    elif y == 0:
        return "0 AD"
    else:
        return f"{int(y)} AD"

# ---------------- IntCal (local file) ----------------
@st.cache_data
def load_intcal_local(path="intcal20.csv"):
    # Expect that intcal20.csv is in repo root
    df = pd.read_csv(path)
    df = df.sort_values("calBP").reset_index(drop=True)
    # smooth to avoid visual gaps
    df["mu14C_smooth"] = df["mu14C"].rolling(window=50, center=True, min_periods=1).mean()
    df["sigma_smooth"] = df["sigma_curve"].rolling(window=50, center=True, min_periods=1).mean()
    df["calAD"] = 1950 - df["calBP"]
    return df

try:
    intcal_df = load_intcal_local()
except FileNotFoundError:
    st.error("intcal20.csv niet gevonden in de repo root. Upload het bestand 'intcal20.csv' in de projectmap.")
    st.stop()

# ---------------- session dataset ----------------
if "data" not in st.session_state:
    st.session_state["data"] = pd.DataFrame(columns=["Sample name","Lab. no.","14C date (BP)","BP","Error","Gekalibreerd (2σ)"])

# ---------------- upload / append ----------------
uploaded_file = st.file_uploader("Upload Excel of PDF", type=["xlsx","xls","pdf"])
if uploaded_file is not None:
    if uploaded_file.name.lower().endswith(".pdf"):
        df = parse_pdf(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    if "14C date (BP)" in df.columns:
        df["BP"], df["Error"] = zip(*df["14C date (BP)"].map(parse_bp_value))
    if "Gekalibreerd (2σ)" not in df.columns:
        df["Gekalibreerd (2σ)"] = None
    st.session_state["data"] = pd.concat([st.session_state["data"], df], ignore_index=True)

# ---------------- manual multiple input ----------------
st.subheader("Handmatige invoer (meerdere regels mogelijk)")
with st.form("manual", clear_on_submit=True):
    c1,c2,c3,c4 = st.columns([2,2,2,3])
    sample = c1.text_input("Sample name")
    lab = c2.text_input("Lab. no.")
    raw_bp = c3.text_input("14C date (BP)", placeholder="bijv. 510 ± 30")
    calib = c4.text_input("Gekalibreerd (2σ) (optioneel)", placeholder="bijv. 393 BC - 70 AD")
    submit = st.form_submit_button("Voeg toe")
    if submit:
        bp, err = parse_bp_value(raw_bp)
        new = {"Sample name": sample or "", "Lab. no.": lab or "", "14C date (BP)": raw_bp or "",
               "BP": bp, "Error": err, "Gekalibreerd (2σ)": calib or None}
        st.session_state["data"] = pd.concat([st.session_state["data"], pd.DataFrame([new])], ignore_index=True)

data = st.session_state["data"]
st.subheader("Samengestelde dataset")
st.dataframe(data, use_container_width=True)

# ---------------- prepare plotting ranges ----------------
show_intcal = st.checkbox("Toon IntCal ±2σ-band", value=True)

# parse calibrated ranges
cal_ranges = []
for i,row in data.iterrows():
    rng = None
    if "Gekalibreerd (2σ)" in data.columns:
        rng = parse_calibrated_range(row.get("Gekalibreerd (2σ)"))
    cal_ranges.append(rng)

# determine x limits: prefer calibrated ranges if any valid, else IntCal full extent
valid_ranges = [r for r in cal_ranges if r and r[0] is not None]
if valid_ranges:
    min_ad = min(r[0] for r in valid_ranges)
    max_ad = max(r[1] for r in valid_ranges)
else:
    # fallback to IntCal extents
    min_ad = intcal_df["calAD"].min()
    max_ad = intcal_df["calAD"].max()

# add 100 year buffer and round to 100
left = int(np.floor((min_ad - 100) / 100.0) * 100)
right = int(np.ceil((max_ad + 100) / 100.0) * 100)
if left == right:
    left -= 100
    right += 100
tick_vals = np.arange(left, right+100, 100)
tick_texts = [format_bc_ad(t) for t in tick_vals]

# ---------------- plotting ----------------
if not data.empty:
    st.subheader("Visualisatie")
    fig = go.Figure()

    # IntCal band (smoothed)
    if show_intcal:
        ad_vals = intcal_df["calAD"].values
        upper = intcal_df["mu14C_smooth"].values + 2 * intcal_df["sigma_smooth"].values
        lower = intcal_df["mu14C_smooth"].values - 2 * intcal_df["sigma_smooth"].values
        # band
        fig.add_trace(go.Scatter(x=ad_vals, y=upper, line=dict(color='rgba(0,0,255,0)'), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=ad_vals, y=lower, fill='tonexty', fillcolor='rgba(0,100,255,0.25)',
                                 line=dict(color='rgba(0,0,255,0)'), name='IntCal20 ±2σ'))
        # remove central line as requested

    # sample horizontal bars (top-down)
    n = len(data)
    y_ticks = []
    y_labels = []
    for idx,row in enumerate(data.itertuples(), start=1):
        y = n - idx + 1  # top-down numbering
        y_ticks.append(y)
        samp_label = getattr(row, "Sample_name", f"sample_{idx}")
        y_labels.append(samp_label)
        rng = cal_ranges[idx-1] if idx-1 < len(cal_ranges) else None
        if rng and rng[0] is not None:
            start_ad, end_ad = rng
            fig.add_trace(go.Scatter(
                x=[start_ad, end_ad], y=[y,y], mode='lines', line=dict(color='firebrick', width=10),
                hoverinfo='text', hovertext=f"{samp_label}: {start_ad} — {end_ad}"
            ))
            # sample name as text slightly right
            fig.add_trace(go.Scatter(x=[end_ad + (right-left)*0.02], y=[y], mode='text', text=[samp_label],
                                     textposition='middle right', showlegend=False))
        else:
            # if no calibrated range, indicate with small marker and label
            fig.add_trace(go.Scatter(x=[None], y=[y], mode='markers+text', text=[samp_label],
                                     textposition='middle right', showlegend=False))

    # axes settings: disable range slider and fix y-range to prevent weird interactive widget
    fig.update_layout(
        xaxis=dict(title="Kalenderjaren (BC/AD)", range=[left,right], tickmode='array', tickvals=tick_vals,
                   ticktext=tick_texts, rangeslider=dict(visible=False)),
        yaxis=dict(title="", tickmode='array', tickvals=y_ticks, ticktext=y_labels, autorange='reversed',
                   fixedrange=True),
        height=180 + 60*n,
        template="simple_white",
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------- PDF export (try to include chart image) ----------------
    st.subheader("Exporteer PDF")
    warn_box = st.empty()

    try:
        # Try to export chart image (may fail on cloud due to missing Chromium)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
            try:
                fig.write_image(tmp_img.name, format="png", scale=2)
                tmp_img.flush()
                img_path = tmp_img.name
                can_include_image = True
            except Exception as e_img:
                can_include_image = False
                img_path = None
        # Build PDF (with or without image)
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=A4)
        W, H = A4
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, H-50, "Radiocarbon Report")
        c.setFont("Helvetica", 10)
        c.drawString(40, H-70, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        if can_include_image and img_path:
            # draw image
            img_reader = ImageReader(img_path)
            # keep margins
            c.drawImage(img_reader, 40, H-380, width=W-80, height=300, preserveAspectRatio=True)
            # remove temp file
            try:
                os.unlink(img_path)
            except Exception:
                pass
            table_y_start = H-400
        else:
            warn_box.warning("Grafiekbeeld kon niet gerenderd worden in deze (online) omgeving. PDF bevat alleen de tabel.")
            table_y_start = H-80

        # write table starting at table_y_start
        c.setFont("Helvetica-Bold", 10)
        cols = ["Sample name", "Lab. no.", "14C date (BP)", "Gekalibreerd (2σ)"]
        x_positions = [40, 220, 360, 480]
        y = table_y_start
        for i, col in enumerate(cols):
            c.drawString(x_positions[i], y, col)
        c.setFont("Helvetica", 10)
        y -= 16
        for _, r in data.iterrows():
            if y < 40:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = H-60
            c.drawString(x_positions[0], y, str(r.get("Sample name","")))
            c.drawString(x_positions[1], y, str(r.get("Lab. no.","")))
            c.drawString(x_positions[2], y, str(r.get("14C date (BP)","")))
            c.drawString(x_positions[3], y, str(r.get("Gekalibreerd (2σ)","")))
            y -= 14
        c.showPage()
        c.save()
        pdf_buffer.seek(0)

        st.download_button("Download PDF-rapport", data=pdf_buffer, file_name="radiocarbon_report.pdf", mime="application/pdf")
    except Exception as e:
        # total fallback: only table as CSV
        warn_box.warning("Kon geen PDF maken (omgeving beperkt). Download de CSV als alternatief.")
        csvbuf = data.to_csv(index=False).encode("utf-8")
        st.download_button("Download dataset (CSV)", data=csvbuf, file_name="radiocarbon_data.csv", mime="text/csv")

# ---------------- always allow CSV download ----------------
if not data.empty:
    csvbuf = data.to_csv(index=False).encode("utf-8")
    st.download_button("Download dataset (CSV)", data=csvbuf, file_name="radiocarbon_data.csv", mime="text/csv")

st.markdown("""
---
**Notities**
- Zorg dat `intcal20.csv` in de repo root staat.  
- PDF bevat grafiek + tabel **lokaal**; op Streamlit Cloud wordt grafiekbeeld vaak niet gegenereerd (Kaleido/Chromium ontbreekt) — in dat geval produceert de PDF een tabel-only fallback en is er een waarschuwing.  
""")
