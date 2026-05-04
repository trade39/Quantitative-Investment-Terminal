import streamlit as st
import yfinance as yf
import pandas as pd
import time
import os
import plotly.graph_objects as go
import re
from io import BytesIO
from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

def get_api_key(key_name):
    """Securely retrieve API keys from secrets or environment variables."""
    if "api_keys" in st.secrets and key_name in st.secrets["api_keys"]:
        return st.secrets["api_keys"][key_name]
    if key_name in st.secrets:
        return st.secrets[key_name]
    if key_name == "gemini_api_key":
        if "GOOGLE_API_KEY" in st.secrets: return st.secrets["GOOGLE_API_KEY"]
    if key_name in os.environ:
        return os.environ[key_name]
    return None

def flatten_dataframe(df):
    """Prevents MultiIndex crashes in yfinance."""
    if df.empty: return df
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.get_level_values(0):
            df.columns = df.columns.get_level_values(0)
        elif df.columns.nlevels > 1 and 'Close' in df.columns.get_level_values(1):
            df.columns = df.columns.get_level_values(1)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def safe_yf_download(tickers, period, interval, retries=3):
    """Robust yfinance download with retries and flattening."""
    for i in range(retries):
        try:
            time.sleep(0.1) 
            df = yf.download(tickers, period=period, interval=interval, progress=False)
            if not df.empty:
                return flatten_dataframe(df)
        except Exception as e:
            if i == retries - 1: return pd.DataFrame()
            time.sleep(2 ** i)
    return pd.DataFrame()

def terminal_chart_layout(fig, title="", height=350):
    """
    Standardized Chart Layout - Monochromatic Navy/Cyan Theme
    Background: #12161F (Matches Cards)
    Grid: #333333
    Text: #CCCCCC
    """
    fig.update_layout(
        title=dict(text=title, font=dict(color="#FFFFFF", family="Arial")),
        template="plotly_dark",
        paper_bgcolor="#12161F", # Card background
        plot_bgcolor="#12161F",  # Chart area background
        height=height,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(
            showgrid=True, 
            gridcolor="#333333", 
            zerolinecolor="#444444",
            tickfont=dict(color="#AAAAAA")
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor="#333333", 
            zerolinecolor="#444444",
            tickfont=dict(color="#AAAAAA")
        ),
        font=dict(family="Courier New", color="#CCCCCC"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CCCCCC")
        )
    )
    return fig

def md_to_rl(text, body_style, header_style, bullet_style=None):
    """Converts AI thesis text to ReportLab flowables with proper section/bullet handling."""
    import re
    elements = []
    if not text: return elements
    if bullet_style is None: bullet_style = body_style
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            elements.append(Spacer(1, 5))
            continue
        # Markdown heading markers → header style
        if re.match(r'^#{1,3}\s+', line):
            line = re.sub(r'^#{1,3}\s+', '', line)
            elements.append(Spacer(1, 8))
            elements.append(Paragraph(f'<b>{line.upper()}</b>', header_style))
            continue
        # Numbered section headers: "1. PRICE ACTION..." etc.
        if re.match(r'^\d+\.\s+[A-Z]', line):
            elements.append(Spacer(1, 10))
            elements.append(Paragraph(f'<b>{line}</b>', header_style))
            continue
        # Inline bold/italic
        line = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', line)
        line = re.sub(r'\*(.+?)\*', r'<i>\1</i>', line)
        # Horizontal rules
        if line.strip('- ') == '' and len(line) >= 3:
            elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey, spaceAfter=4))
            continue
        # Bullet points
        if line.startswith(('• ', '- ', '* ')):
            elements.append(Paragraph('&bull; ' + line[2:], bullet_style))
        else:
            elements.append(Paragraph(line, body_style))
    return elements

def clean_markdown(text):
    """
    Cleans markdown characters and converts common ones to ReportLab-friendly XML tags.
    """
    if not text: return ""
    import re
    # 1. Convert bold **text** to <b>text</b>
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # 2. Convert italic *text* to <i>text</i>
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    # 3. Remove header markers ###, ##, #
    text = re.sub(r'#+\s*(.*?)\n', r'<b>\1</b><br/>', text)
    text = re.sub(r'#+\s*(.*)', r'<b>\1</b>', text)
    # 4. Clean up any remaining stray characters (like single * for bullets)
    text = text.replace('* ', '• ')
    return text

def generate_pdf_report(data):
    """
    Generates a professional Institutional Brief PDF.
    data: dict with keys: ticker, price, pct, narrative, thesis, levels, ml_signal, regime, ms_trend (optional)
    """
    import os
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=LETTER, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()

    # --- Style Definitions ---
    title_style   = ParagraphStyle('T',  parent=styles['Normal'], fontName='Helvetica-Bold',   fontSize=18, textColor=colors.HexColor("#003366"), alignment=1, spaceAfter=4)
    subtitle_style= ParagraphStyle('St', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=10, textColor=colors.HexColor("#555555"), alignment=1, spaceAfter=4)
    tagline_style = ParagraphStyle('Tg', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=11, textColor=colors.HexColor("#003366"), alignment=1, spaceBefore=6, spaceAfter=2, leading=16)
    header_style  = ParagraphStyle('H',  parent=styles['Normal'], fontName='Helvetica-Bold',   fontSize=11, textColor=colors.HexColor("#003366"), spaceBefore=12, spaceAfter=6)
    body_style    = ParagraphStyle('B',  parent=styles['Normal'], fontName='Helvetica',         fontSize=9,  leading=14, alignment=4)
    bullet_style  = ParagraphStyle('Bu', parent=styles['Normal'], fontName='Helvetica',         fontSize=9,  leading=14, leftIndent=14, spaceAfter=3)

    elements = []

    # --- Logo ---
    logo_path = "static/logo.png"
    if os.path.exists(logo_path):
        try:
            img = Image(logo_path, width=1.5*inch, height=0.5*inch)
            img.hAlign = 'CENTER'
            elements.append(img)
            elements.append(Spacer(1, 8))
        except: pass

    # --- Title & Date ---
    elements.append(Paragraph("JD CAPITAL QUANTITATIVE INVESTMENT", title_style))
    elements.append(Paragraph(f"INSTITUTIONAL BRIEF: {data['ticker']} | {pd.Timestamp.now().strftime('%B %d, %Y')}", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#003366"), spaceAfter=10))

    # --- Tagline Block (derived from regime / market structure) ---
    regime_str = str(data.get('regime', '')).upper()
    ms_str     = str(data.get('ms_trend', '')).upper()
    if 'BULLISH' in regime_str or 'TREND' in regime_str:
        bias_line = 'Bullish Trend → Momentum-driven → Breakout Confirmed'
        sub_line  = 'The market has established directional conviction above key structure.'
    elif 'BEARISH' in regime_str:
        bias_line = 'Bearish Trend → Momentum-driven → Breakdown Confirmed'
        sub_line  = 'The market has established directional conviction below key structure.'
    else:
        bias_line = 'Neutral → Event-driven → Breakout-dependent'
        sub_line  = 'Not because the market is unclear— but because it is structurally waiting to move.'

    elements.append(Spacer(1, 6))
    elements.append(Paragraph(f'<i>{bias_line}</i>', tagline_style))
    elements.append(Paragraph(f'<i>{sub_line}</i>', ParagraphStyle('Tg2', parent=tagline_style, fontSize=9, textColor=colors.HexColor("#777777"))))
    elements.append(Spacer(1, 12))

    # --- Executive Summary Table ---
    elements.append(Paragraph("EXECUTIVE PERFORMANCE SUMMARY", header_style))
    hud_data = [
        [Paragraph('<b>METRIC</b>',        body_style), Paragraph('<b>VALUE / STATUS</b>', body_style)],
        [Paragraph('ASSET TICKER',         body_style), Paragraph(str(data['ticker']), body_style)],
        [Paragraph('LAST PRICE',           body_style), Paragraph(f"{data['price']:,.2f} ({data['pct']:+.2f}%)", body_style)],
        [Paragraph('AI ML SIGNAL',         body_style), Paragraph(str(data['ml_signal']), body_style)],
        [Paragraph('QUANT REGIME',         body_style), Paragraph(str(data['regime']), body_style)],
    ]
    t = Table(hud_data, colWidths=[2.5*inch, 3.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0), colors.HexColor("#003366")),
        ('TEXTCOLOR',     (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN',         (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.HexColor("#F2F2F2"), colors.white]),
        ('GRID',          (0, 0), (-1, -1), 0.5, colors.HexColor("#D5D5D5")),
        ('TOPPADDING',    (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('LEFTPADDING',   (0, 0), (-1, -1), 12),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 18))

    # --- Optional Chart ---
    if data.get('chart_image'):
        try:
            from reportlab.platypus import Image as RLImage
            img = RLImage(BytesIO(data['chart_image']), width=6*inch, height=3*inch)
            img.hAlign = 'CENTER'
            elements.append(Paragraph("MARKET STRUCTURE & LIQUIDITY MAP", header_style))
            elements.append(img)
            elements.append(Spacer(1, 10))
        except Exception as e:
            elements.append(Paragraph(f"Chart unavailable: {e}", body_style))

    # --- Investment Thesis (numbered sections) ---
    if data.get('thesis'):
        elements.append(Paragraph("INVESTMENT THESIS", header_style))
        elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#003366"), spaceAfter=6))
        elements.extend(md_to_rl(data['thesis'], body_style, header_style, bullet_style))
        elements.append(Spacer(1, 10))

    # --- Technical Narrative (if separate) ---
    if data.get('narrative'):
        elements.append(Paragraph("TECHNICAL NARRATIVE & ANALYSIS", header_style))
        elements.extend(md_to_rl(data['narrative'], body_style, header_style, bullet_style))
        elements.append(Spacer(1, 10))

    # --- Algorithmic Execution Levels ---
    if data.get('levels'):
        elements.append(Paragraph("ALGORITHMIC EXECUTION LEVELS", header_style))
        lvl = data['levels']
        lvl_data = [[Paragraph('<b>LEVEL IDENTIFIER</b>', body_style), Paragraph('<b>PRICE LEVEL</b>', body_style)]]
        for k, v in lvl.items():
            lvl_data.append([Paragraph(str(k), body_style), Paragraph(f"{v:,.2f}", body_style)])
        lt = Table(lvl_data, colWidths=[2.5*inch, 2.5*inch])
        lt.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, 0), colors.HexColor("#1A1A2E")),
            ('TEXTCOLOR',     (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN',         (0, 0), (-1, -1), 'LEFT'),
            ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.HexColor("#F9F9F9"), colors.white]),
            ('GRID',          (0, 0), (-1, -1), 0.5, colors.HexColor("#D5D5D5")),
            ('TOPPADDING',    (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('LEFTPADDING',   (0, 0), (-1, -1), 12),
        ]))
        elements.append(lt)

    # --- Disclaimer & Footer ---
    elements.append(Spacer(1, 30))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    elements.append(Paragraph(
        '<i>DISCLAIMER: This report is for institutional information purposes only and does not constitute financial advice. '
        'JD Capital Quantitative Investment assumes no liability for trading decisions based on this AI-generated synthesis.</i>',
        ParagraphStyle('Disc', parent=styles['Normal'], fontSize=7, textColor=colors.grey, alignment=1, spaceBefore=6)
    ))

    def page_decorator(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.grey)
        canvas.line(0.5*inch, 0.75*inch, 7.5*inch, 0.75*inch)
        canvas.drawString(0.5*inch, 0.5*inch, f"JD Capital Quantitative Investment | {data['ticker']} | Internal Brief")
        canvas.drawRightString(7.5*inch, 0.5*inch, f"Page {canvas.getPageNumber()}")
        if canvas.getPageNumber() > 1:
            canvas.setFont('Helvetica-Bold', 10)
            canvas.setFillColor(colors.HexColor("#003366"))
            canvas.drawString(0.5*inch, 10.5*inch, f"INSTITUTIONAL BRIEF: {data['ticker']}")
            canvas.line(0.5*inch, 10.4*inch, 7.5*inch, 10.4*inch)
        canvas.restoreState()

    doc.build(elements, onFirstPage=page_decorator, onLaterPages=page_decorator)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

# --- UI HELPERS (Moved from app.py) ---
def parse_eco_value(val_str):
    if not isinstance(val_str, str) or val_str == '': return None
    clean = val_str.replace('%', '').replace(',', '')
    multiplier = 1.0
    if 'K' in clean.upper(): multiplier = 1000.0; clean = clean.upper().replace('K', '')
    elif 'M' in clean.upper(): multiplier = 1000000.0; clean = clean.upper().replace('M', '')
    elif 'B' in clean.upper(): multiplier = 1000000000.0; clean = clean.upper().replace('B', '')
    try: return float(clean) * multiplier
    except: return None

def analyze_eco_context(actual_str, forecast_str, previous_str):
    is_happened = actual_str is not None and actual_str != ""
    val_actual = parse_eco_value(actual_str)
    val_forecast = parse_eco_value(forecast_str)
    val_prev = parse_eco_value(previous_str)
    context_str = ""
    bias = "Neutral"
    if is_happened:
        if val_actual is not None and val_forecast is not None:
            context_str = f"Act {actual_str} / Est {forecast_str}"
            delta = val_actual - val_forecast
            if delta > 0: bias = "Bullish"
            else: bias = "Bearish"
        else: context_str = f"Actual: {actual_str}"
    else:
         context_str = f"Est {forecast_str}" if forecast_str else "Waiting..."
    return context_str, bias

def generate_cot_analysis(spec_net, hedge_net, spec_label, hedge_label):
    spec_sent = "🟢 BULLISH" if spec_net > 0 else "🔴 BEARISH"
    hedge_sent = "🟢 BULLISH" if hedge_net > 0 else "🔴 BEARISH"
    if (spec_net > 0 and hedge_net < 0) or (spec_net < 0 and hedge_net > 0):
        structure = "✅ **Healthy Structure:** Risk Transfer active."
    else:
        structure = "⚠️ **Anomaly:** Groups positioned same side."
    return f"* **{spec_label}:** {spec_sent} (Net: {int(spec_net):,})\n* **{hedge_label}:** {hedge_sent}\n{structure}"
