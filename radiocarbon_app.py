# app.py
import io
import re
from math import ceil
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# ---------- Helpers: parse calibrated range strings ----------

def parse_calibrated_range(text: str) -> Tuple[int, int]:
    """
    Parse a variety of human-written calibrated ranges into (startYear, endYear).
    We return years in astronomical CE: CE positive (e.g. 2025), BCE negative (e.g. -600 for 600 BCE).
    Accepts forms like:
      - "600 BC - 500 BC", "600 BCE - 500 BCE"
      - "400-350 BCE"
      - "250 CE - 300 CE", "250 AD - 300 AD"
      - "100 BC – 1 AD"
      - "400 BC - 200 CE"
      - "200 BCE - 150 BCE"
      - "1500 - 1400 BC"
    If parsing fails raise ValueError.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty range")

    # Normalize punctuation
    s = text.replace("–", "-").replace("—", "-").replace("—", "-")
    # Try to capture two numbers and optional era indicators
    # Patterns: number (era?) [sep] number (era?)
    m = re.search(r'(-?\d+)\s*(BCE|BC|CE|AD|B\.C\.E\.|B\.C\.|A\.D\.)?\s*[-–]\s*(-?\d+)\s*(BCE|BC|CE|AD|B\.C\.E\.|B\.C\.|A\.D\.)?', s, flags=re.IGNORECASE)
    if not m:
        # If only single year provided treat as tiny range of ±0
        m2 = re.search(r'(-?\d+)\s*(BCE|BC|CE|AD)?', s, flags=re.IGNORECASE)
        if m2:
            val = int(m2.group(1))
            era = (m2.group(2) or "").upper()
            num = val * (-1 if "B" in era else 1) if era else int(val)
            return num, num
        raise ValueError(f"Kan range niet parsen: '{text}'")

    a_num, a_era, b_num, b_era = m.group(1), m.group(2), m.group(3), m.group(4)
    def to_year(n_str, era):
        n = int(n_str)
        era = (era or "").upper() if era else ""
        # If era mentions BC/BCE -> negative year. AD/CE -> positive. If era absent assume CE if value > 50? but safer: assume positive (CE)
        if "B" in era:
            return -abs(n)
        return n  # CE/AD or none -> positive
    y1 = to_year(a_num, a_era)
    y2 = to_year(b_num, b_era)
    # normalize order: start <= end (older -> younger). For BCE negative numbers smaller means older; e.g., -600 < -500
    start = min(y1, y2)
    end = max(y1, y2)
    return start, end

# ---------- UI / App ----------

st.set_page_config(page_title="Calibrated ranges → OxCal-like plot", layout="wide")

st.title("Gekalibreerde dateringen (2σ) — grafiek en export")
st.markdown(
    "Voer handmatig records in of upload een Excel met kolommen: "
    "`Sample name`, `Lab.no.`, `14C date (BP)`, `1σ`, `Calibrated date (2σ)`, `Context`."
)

# --- Left pane: input table and upload ---
with st.sidebar:
    st.header("Invoer")
    uploaded = st.file_uploader("Upload Excel (.xlsx/.xls) met kolommen zoals hierboven", type=["xlsx", "xls"])
    if uploaded is not None:
        try:
            df_in = pd.read_excel(uploaded, engine="openpyxl")
            # Standardize column names
            # Accept various column name variants
            col_map = {}
            for c in df_in.columns:
                cc = c.strip().lower()
                if "sample" in cc and "name" in cc:
                    col_map[c] = "Sample name"
                elif "lab" in cc:
                    col_map[c] = "Lab.no."
                elif "14c" in cc or "14c date" in cc or "radiocarbon" in cc:
                    col_map[c] = "14C date (BP)"
                elif "1σ" in cc or "1s" in cc or "sigma" in cc:
                    col_map[c] = "1σ"
                elif "calibr" in cc and "2" in cc:
                    col_map[c] = "Calibrated date (2σ)"
                elif "context" in cc:
                    col_map[c] = "Context"
            df_in = df_in.rename(columns=col_map)
            # Keep only requested columns and add missing
            cols = ["Sample name", "Lab.no.", "14C date (BP)", "1σ", "Calibrated date (2σ)", "Context"]
            for c in cols:
                if c not in df_in.columns:
                    df_in[c] = ""
            df = df_in[cols].copy()
            st.success(f"{len(df)} rijen ingelezen uit Excel")
        except Exception as e:
            st.error(f"Kon Excel niet inlezen: {e}")
            df = pd.DataFrame(columns=["Sample name", "Lab.no.", "14C date (BP)", "1σ", "Calibrated date (2σ)", "Context"])
    else:
        # start empty table with one example row
        df = pd.DataFrame([{
            "Sample name": "",
            "Lab.no.": "",
            "14C date (BP)": "",
            "1σ": "",
            "Calibrated date (2σ)": "",
            "Context": ""
        }])

    st.write("Bewerk de rijen direct (klik op cel). Voeg rijen toe met de knop.")
    edited = st.experimental_data_editor(df, num_rows="dynamic")
    if edited is None:
        edited = df.copy()

    if st.button("Voeg lege rij toe"):
        edited = pd.concat([edited, pd.DataFrame([{
            "Sample name": "",
            "Lab.no.": "",
            "14C date (BP)": "",
            "1σ": "",
            "Calibrated date (2σ)": "",
            "Context": ""
        }])], ignore_index=True)
        st.experimental_rerun()

    st.markdown("---")
    st.write("Plot opties")
    max_margin_years = st.number_input("Max marge rondom uiterste dateringen (jaar)", value=50, min_value=0, step=1)
    bin_step = st.number_input("Tick stapel op x-as (jaar)", value=100, min_value=1, step=1)
    st.markdown("---")
    generate = st.button("Genereer grafiek en APD")

# Main area: show table + generated plot
st.subheader("Ingevoerde records")
st.dataframe(edited, use_container_width=True)

# Prepare parsed intervals if generate clicked or automatic
if 'generate' not in locals():
    generate = False

if generate:
    df2 = edited.copy()
    parsed = []
    errors = []
    for idx, row in df2.iterrows():
        rng_text = str(row.get("Calibrated date (2σ)", "")).strip()
        if not rng_text:
            errors.append(f"Rij {idx+1} ('{row.get('Sample name','')}'): geen calibrated range opgegeven.")
            continue
        try:
            start, end = parse_calibrated_range(rng_text)
            parsed.append({
                "Sample name": str(row.get("Sample name", "")),
                "Lab.no.": str(row.get("Lab.no.", "")),
                "14C date (BP)": row.get("14C date (BP)", ""),
                "1σ": row.get("1σ", ""),
                "cal_start": start,
                "cal_end": end,
                "Context": row.get("Context", ""),
                "raw_range": rng_text
            })
        except Exception as e:
            errors.append(f"Rij {idx+1} ('{row.get('Sample name','')}'): kon range niet parsen: {e}")

    if len(parsed) == 0:
        st.error("Geen geldige gekalibreerde ranges gevonden. Zie fouten hieronder.")
        for e in errors:
            st.write("- " + e)
    else:
        pdf_buf = None
        # Create dataframe of parsed intervals
        parsed_df = pd.DataFrame(parsed)
        # sort by mean age descending = oldest first on top
        parsed_df["mean"] = (parsed_df["cal_start"] + parsed_df["cal_end"]) / 2
        parsed_df = parsed_df.sort_values("mean")  # ascending: more negative (older) first
        parsed_df = parsed_df.reset_index(drop=True)

        # Build Plotly horizontal bars: we want older at top -> so y categories in order of parsed_df
        y_names = parsed_df["Sample name"].fillna("").astype(str).tolist()

        # Convert starts/ends to numeric timeline for plotting (CE positive, BCE negative)
        starts = parsed_df["cal_start"].tolist()
        ends = parsed_df["cal_end"].tolist()
        # x-domain margin: compute min start and max end and add margins (in years)
        min_start = min(starts)
        max_end = max(ends)
        x_min = min_start - max_margin_years
        x_max = max_end + max_margin_years

        # Generate bars: plotly requires numeric widths; we'll use base=starts and x=widths
        widths = [end - start for start, end in zip(starts, ends)]
        # Build figure
        fig = go.Figure()
        # Each sample as separate barra to preserve ordering and hover
        for i, name in enumerate(y_names):
            fig.add_trace(go.Bar(
                x=[widths[i]],
                y=[name],
                base=[starts[i]],
                orientation='h',
                marker=dict(color='royalblue', opacity=0.6),
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    f"Range: {parsed_df.loc[i,'raw_range']}<br>"
                    f"Start: {format_year_label(starts[i])}<br>"
                    f"End: {format_year_label(ends[i])}<br>"
                    "<extra></extra>"
                ),
                showlegend=False
            ))

        # X ticks: choose step bin_step (from sidebar)
        def yr_labels(minv, maxv, step):
            # generate tick positions as integers covering the domain
            # ensure nice round ticks e.g., multiples of step
            lo = int(ceil(minv / step) * step)
            ticks = list(range(lo, int(maxv) + step, step))
            return ticks

        ticks = yr_labels(x_min, x_max, int(bin_step))
        ticktext = [format_year_label(t) for t in ticks]

        fig.update_layout(
            barmode='stack',
            xaxis=dict(range=[x_min, x_max], tickvals=ticks, ticktext=ticktext, title='Calibrated calendar years (BCE/CE)'),
            yaxis=dict(autorange='reversed'),  # keep oldest at top
            margin=dict(l=200, r=40, t=40, b=80),
            height=600
        )

        # Display figure
        st.plotly_chart(fig, use_container_width=True)

        # Build export PDF: figure -> PNG via kaleido, then compose PDF with table
        try:
            # require kaleido installed
            png_bytes = fig.to_image(format="png", engine="kaleido")
            # Build table for PDF from parsed_df with selected columns (show original calibrated range)
            pdf_buffer = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=landscape(A4))
            elements = []
            styles = getSampleStyleSheet()
            elements.append(Paragraph("Calibrated ranges — export", styles['Title']))
            elements.append(Spacer(1, 12))
            # add image
            img = Image(io.BytesIO(png_bytes))
            img._restrictSize(1000, 400)
            elements.append(img)
            elements.append(Spacer(1, 12))
            # prepare table data
            table_data = [["Sample name", "Lab.no.", "14C date (BP)", "1σ", "Calibrated date (2σ)", "Context"]]
            for _, r in parsed_df.iterrows():
                table_data.append([
                    r["Sample name"],
                    r["Lab.no."],
                    r["14C date (BP)"],
                    r["1σ"],
                    r["raw_range"],
                    r["Context"]
                ])
            tbl = Table(table_data, repeatRows=1, hAlign='LEFT')
            tbl.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
            ]))
            elements.append(tbl)
            doc.build(elements)
            pdf_bytes = pdf_buffer.getvalue()
            st.success("PDF gereed — klik download hieronder.")
            st.download_button("Download PDF (fig + tabel)", data=pdf_bytes, file_name="calibrated_ranges_export.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"PDF-export faalde: {e}")
            # fallback: allow downloading plot as PNG
            try:
                png_bytes = fig.to_image(format="png", engine="kaleido")
                st.download_button("Download plot PNG", data=png_bytes, file_name="plot.png", mime="image/png")
            except Exception:
                st.warning("Kaleido niet beschikbaar — installeer 'kaleido' om PNG/PDF export te kunnen gebruiken.")

# Utility to format tick labels nicely (BCE/CE)
def format_year_label(y: int) -> str:
    y = int(y)
    if y > 0:
        return f"{y} CE"
    if y == 0:
        return "1 BCE/CE"
    return f"{abs(y)} BCE"

# We used format_year_label inside plotly hover -> put function available
def format_year_label_public(y):
    return format_year_label(y)
