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
    """Parse '510 ± 30' of '510+/-30' — return (BP, error) of (None,None)."""
    if not value or (isinstance(value, float) and np.isnan(value)):
        return None, None
    match = re.match(r"(\d+)\s*[±+\-/]*\s*(\d+)", str(value))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def parse_pdf(file):
    """Eenvoudige PDF-parser — pas aan bij afwijkend PDF-format."""
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

# --- Parser voor 'Gekalibreerd (2σ)' kolom ---
def parse_calibrated_range(val):
    """
    Parse strings like:
      '393 BC - 70 AD'
      '393 BC – 70 AD'
      '500-170 AD'
      '500–170'
      '-393 to 70'
      '393BC-70AD'
    Return (start_AD, end_AD) as integers (AD years; BC -> negative).
    If parsing fails, return (None, None).
    """
    if pd.isna(val):
        return None, None
    s = str(val).strip()
    # normalize dashes
    s = s.replace("–", "-").replace("—", "-").replace(" to ", "-").replace("TO", "-")
    # remove extraneous text like 'cal', '2σ' etc.
    s = re.sub(r"\(.*?\)", "", s)
    s = s.replace("calBP", "").replace("cal BP", "").strip()

    # find all tokens that look like number + optional era
    tokens = re.findall(r"(-?\d+)\s*(BC|AD|BCE|CE)?", s, flags=re.IGNORECASE)
    if len(tokens) >= 2:
        def to_ad(tok):
            num_str, era = tok
            num = int(num_str)
            if era:
                era = era.upper()
                if era in ("BC", "BCE"):
                    return -abs(num)
                else:  # AD or CE
                    return num
            else:
                # geen era opgegeven — heuristiek: als num > 1000 assume BP? but here treat as AD
                # better to assume: if number > 1000 -> likely AD negative? we keep simple: AD
                return num
        a = to_ad(tokens[0])
        b = to_ad(tokens[1])
        start, end = (a, b) if a <= b else (b, a)
        return int(start), int(end)

    # fallback: look for numbers (maybe with BC/AD suffixes separately)
    parts = s.split("-")
    if len(parts) >= 2:
        def parse_part(p):
            m = re.search(r"(-?\d+)", p)
            if not m:
                return None
            num = int(m.group(1))
            # detect BC/AD
            if re.search(r"BC|BCE", p, flags=re.IGNORECASE):
                return -abs(num)
            else:
                # assume AD if 'AD' present or if small number; otherwise keep as AD
                return num
        p0 = parse_part(parts[0])
        p1 = parse_part(parts[1])
        if p0 is not None and p1 is not None:
            start, end = (p0, p1) if p0 <= p1 else (p1, p0)
            return int(start), int(end)

    return None, None

# ----------------- IntCal loader (smoothing optional) -----------------
@st.cache_data
def load_intcal(path="intcal20.csv"):
    df = pd.read_csv(path)
    df = df.sort_values("calBP").reset_index(drop=True)
    # smoothing for band (window 50)
    df["mu14C_smooth"] = df["mu14C"].rolling(window=50, min_periods=1, center=True).mean()
    df["sigma_curve_smooth"] = df["sigma_curve"].rolling(window=50, min_periods=1, center=True).mean()
    return df

intcal_df = load_intcal()

# Posterior calculation (kept for optional overlay)
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
    # ensure expected columns exist; preserve 'Gekalibreerd (2σ)' if present
    if "14C date (BP)" in df.columns:
        df["BP"], df["Error"] = zip(*df["14C date (BP)"].map(parse_bp_value))
    else:
        df["BP"], df["Error"] = None, None
    # keep calibrated column if provided; if not present, create empty
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

# ----------------- Show table -----------------
st.subheader("Samengestelde dataset")
st.dataframe(data, use_container_width=True)

# ----------------- Determine plotting strategy -----------------
use_calibrated_col = "Gekalibreerd (2σ)" in data.columns and data["Gekalibreerd (2σ)"].notna().any()
show_intcal = st.checkbox("Toon IntCal ±2σ-band", value=True)
overlay_posteriors = st.checkbox("Overlay posterior wiggles (berekend uit BP en IntCal)", value=False)

# parse calibrated ranges into AD-years
cal_ranges = []
for idx, row in data.iterrows():
    rng = None
    if use_calibrated_col:
        rng = parse_calibrated_range(row.get("Gekalibreerd (2σ)"))
    cal_ranges.append(rng)  # may be (None,None)

# compute overall x-axis limits from calibrated ranges if available, else fallback to IntCal extents or posterior extents
ad_min, ad_max = None, None
if use_calibrated_col:
    for rng in cal_ranges:
        if rng and rng[0] is not None:
            a, b = rng
            # a and b are AD years (BC negative)
            if ad_min is None or a < ad_min:
                ad_min = a
            if ad_max is None or b > ad_max:
                ad_max = b

# if no calibrated col ranges or ad_min/ad_max still none, try to infer from BP/posterior if overlay_posteriors True, else use IntCal
if ad_min is None or ad_max is None:
    # fallback to IntCal coverage (convert calBP to AD)
    calBP = intcal_df["calBP"].values
    ad_vals = bp_to_ad(calBP)
    ad_min = ad_vals.min()
    ad_max = ad_vals.max()

# add 100-year buffer and round to nearest 100
left = int(np.floor((ad_min - 100) / 100.0) * 100)
right = int(np.ceil((ad_max + 100) / 100.0) * 100)

# build ticks every 100 years
tick_vals = np.arange(left, right + 100, 100)
tick_texts = [format_bc_ad(v) for v in tick_vals]

# ----------------- Plot -----------------
if not data.empty:
    st.subheader("Visualisatie (gebruik 'Gekalibreerd (2σ)')")

    fig = go.Figure()

    # IntCal band (smoothed) if requested
    if show_intcal:
        calBP = intcal_df["calBP"].values
        ad_vals = bp_to_ad(calBP)
        mu = intcal_df["mu14C_smooth"].values
        sigma = intcal_df["sigma_curve_smooth"].values
        upper = mu + 2 * sigma
        lower = mu - 2 * sigma
        # plot band as filled polygon (upper then lower)
        fig.add_trace(go.Scatter(x=ad_vals, y=upper,
                                 line=dict(color='rgba(0,0,255,0)'),
                                 hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Scatter(x=ad_vals, y=lower,
                                 fill='tonexty', fillcolor='rgba(0,100,255,0.25)',
                                 line=dict(color='rgba(0,0,255,0)'),
                                 name='IntCal20 ±2σ'))

    # If using calibrated column -> draw horizontal ranges per sample
    y_positions = []
    y_labels = []
    for idx, row in data.iterrows():
        y = len(data) - idx  # so first row at top (larger y)
        y_positions.append(y)
        y_labels.append(row.get("Sample name", f"sample_{idx}"))

        rng = cal_ranges[idx] if idx < len(cal_ranges) else None
        if rng and rng[0] is not None:
            start_ad, end_ad = rng
            # draw horizontal line for the calibrated 2σ range
            fig.add_trace(go.Scatter(
                x=[start_ad, end_ad],
                y=[y, y],
                mode="lines",
                line=dict(color="firebrick", width=8),
                name=row.get("Sample name", ""),
                hovertemplate=f"%{{x}}<br>{row.get('Sample name','')}"
            ))
            # optional marker at midpoint
            mid = (start_ad + end_ad) / 2.0
            fig.add_trace(go.Scatter(
                x=[mid], y=[y], mode="markers+text",
                marker=dict(color="black", size=6),
                text=[row.get("Sample name","")], textposition="right",
                showlegend=False, hoverinfo="skip"
            ))
        else:
            # no calibrated range: optionally show a placeholder or skip
            # show sample name at y position
            fig.add_trace(go.Scatter(
                x=[None], y=[y], mode="markers+text", text=[row.get("Sample name","")],
                textposition="right", showlegend=False, hoverinfo="skip"
            ))

    # overlay posterior wiggles if requested and if BP available
    if overlay_posteriors:
        for idx, row in data.iterrows():
            if pd.notna(row.get("BP")) and pd.notna(row.get("Error")):
                cal_grid, post = calibrate_posterior(row["BP"], row["Error"], intcal_df)
                ad_grid = bp_to_ad(cal_grid)
                # scale post to small vertical size relative to y spacing
                max_gap = 1.0  # small amplitude
                # compute vertical offset consistent with y coordinate system
                y = len(data) - idx
                amp = 0.4  # amplitude in y units
                post_scaled = post / post.max() * amp
                fig.add_trace(go.Scatter(
                    x=ad_grid, y=post_scaled + y - 0.3, mode="lines",
                    line=dict(width=1), name=f"{row.get('Sample name','')}_posterior",
                    showlegend=False
                ))

    # set layout: y ticks = sample names
    fig.update_yaxes(
        tickmode="array",
        tickvals=y_positions,
        ticktext=y_labels,
        autorange="reversed"
    )

    fig.update_layout(
        xaxis=dict(title="Kalenderjaren (BC/AD)", range=[left, right],
                   tickmode="array", tickvals=tick_vals, ticktext=tick_texts),
        yaxis=dict(title="", showgrid=False),
        height=150 + 60 * len(data),
        template="simple_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ----------------- PDF export (tries to write image; handles cloud) -----------------
    st.subheader("Exporteer PDF (grafiek + tabel)")
    try:
        # attempt to write image (may fail on Streamlit Cloud if kaleido/chrome missing)
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp_img:
            fig.write_image(tmp_img.name, format="png", scale=2)
            tmp_img.seek(0)
            pdf_buffer = io.BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=A4)
            c.setTitle("Radiocarbon Report")
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, 810, "Radiocarbon Visual Report")
            c.setFont("Helvetica", 10)
            c.drawString(50, 795, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            c.drawImage(ImageReader(tmp_img.name), 40, 350, width=520, height=420, preserveAspectRatio=True)

            # write table
            text_y = 330
            c.setFont("Helvetica-Bold", 10)
            cols = ["Sample name", "Lab. no.", "14C date (BP)", "Gekalibreerd (2σ)"]
            x_positions = [50, 200, 320, 420]
            for i, col in enumerate(cols):
                c.drawString(x_positions[i], text_y, col)
            c.setFont("Helvetica", 10)
            for _, r in data.iterrows():
                text_y -= 14
                if text_y < 40:
                    c.showPage()
                    text_y = 800
                c.drawString(x_positions[0], text_y, str(r.get("Sample name", "")))
                c.drawString(x_positions[1], text_y, str(r.get("Lab. no.", "")))
                c.drawString(x_positions[2], text_y, str(r.get("14C date (BP)", "")))
                c.drawString(x_positions[3], text_y, str(r.get("Gekalibreerd (2σ)", "")))
            c.showPage()
            c.save()
            pdf_buffer.seek(0)
        st.download_button("Download PDF-rapport", data=pdf_buffer,
                           file_name="radiocarbon_report.pdf", mime="application/pdf")
    except Exception as e:
        # if kaleido missing or other image issue, still produce PDF without chart image
        st.warning("Kan grafiek niet renderen in PDF (omgeving mist kaleido/Chromium). PDF bevat dan alleen de tabel.")
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=A4)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, 800, "Radiocarbon Visual Report (table only)")
        text_y = 780
        c.setFont("Helvetica-Bold", 10)
        cols = ["Sample name", "Lab. no.", "14C date (BP)", "Gekalibreerd (2σ)"]
        x_positions = [50, 200, 320, 420]
        for i, col in enumerate(cols):
            c.drawString(x_positions[i], text_y, col)
        c.setFont("Helvetica", 10)
        for _, r in data.iterrows():
            text_y -= 14
            if text_y < 40:
                c.showPage()
                text_y = 800
            c.drawString(x_positions[0], text_y, str(r.get("Sample name", "")))
            c.drawString(x_positions[1], text_y, str(r.get("Lab. no.", "")))
            c.drawString(x_positions[2], text_y, str(r.get("14C date (BP)", "")))
            c.drawString(x_positions[3], text_y, str(r.get("Gekalibreerd (2σ)", "")))
        c.showPage()
        c.save()
        pdf_buffer.seek(0)
        st.download_button("Download PDF-rapport (table only)", data=pdf_buffer,
                           file_name="radiocarbon_report_table_only.pdf", mime="application/pdf")

# ----------------- CSV download -----------------
if not data.empty:
    csv_buf = data.to_csv(index=False).encode("utf-8")
    st.download_button("Download dataset (CSV)", data=csv_buf, file_name="radiocarbon_data.csv", mime="text/csv")

st.markdown("""
---  
**Opmerkingen:**  
- Zorg dat de kolom exact `Gekalibreerd (2σ)` heet in je Excel, of vul die handmatig bij invoer.  
- De parser ondersteunt hoofdzakelijk notaties zoals `393 BC - 70 AD`, `500-170`, `-393 to 70` etc.  
- Als je de posterior-wiggles (uit BP + IntCal) toch wilt zien, vink dan 'Overlay posterior wiggles' in.  
""")
