import streamlit as st
import yfinance as yf
import pandas as pd
import time
import os
import plotly.graph_objects as go

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

def generate_pdf_report(data):
    """
    Generates a professional Institutional Brief PDF.
    data: dict containing 'ticker', 'price', 'pct', 'narrative', 'thesis', 'levels', 'ml_signal', 'regime'
    """
    from io import BytesIO
    from reportlab.lib.pagesizes import LETTER
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=LETTER, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    
    # Custom Styles for Institutional Look
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], color=colors.HexColor("#003366"), alignment=1, spaceAfter=5, fontName='Helvetica-Bold', fontSize=18)
    subtitle_style = ParagraphStyle('SubtitleStyle', parent=styles['Normal'], color=colors.HexColor("#666666"), alignment=1, spaceAfter=20, fontName='Helvetica-Oblique', fontSize=10)
    header_style = ParagraphStyle('HeaderStyle', parent=styles['Heading2'], color=colors.HexColor("#003366"), spaceBefore=15, spaceAfter=10, fontName='Helvetica-Bold', fontSize=12, borderPadding=5)
    body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontName='Helvetica', fontSize=10, leading=14, alignment=4) # Justified
    body_style_bold = ParagraphStyle('BodyBold', parent=body_style, fontName='Helvetica-Bold')
    
    elements = []
    
    # 0. Header with Logo (if exists)
    logo_path = "static/logo.png"
    import os
    if os.path.exists(logo_path):
        try:
            img = Image(logo_path, width=1.5*inch, height=0.5*inch)
            img.hAlign = 'CENTER'
            elements.append(img)
            elements.append(Spacer(1, 10))
        except: pass

    # 1. Title
    elements.append(Paragraph("JD CAPITAL QUANTITATIVE INVESTMENT", title_style))
    elements.append(Paragraph(f"INSTITUTIONAL BRIEF: {data['ticker']} | {pd.Timestamp.now().strftime('%B %d, %Y')}", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#003366"), spaceAfter=20))
    
    # 2. Executive Summary / HUD
    elements.append(Paragraph("EXECUTIVE PERFORMANCE SUMMARY", header_style))
    hud_data = [
        [Paragraph("<b>METRIC</b>", body_style), Paragraph("<b>VALUE / STATUS</b>", body_style)],
        ["ASSET TICKER", data['ticker']],
        ["LAST PRICE", f"{data['price']:,.2f} ({data['pct']:+.2f}%)"],
        ["AI ML SIGNAL", data['ml_signal']],
        ["QUANT REGIME", data['regime']],
    ]
    t = Table(hud_data, colWidths=[2.5*inch, 3.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#003366")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#F2F2F2"), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#D5D5D5")),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 20))
    
    # 3. AI Narrative
    if data.get('narrative'):
        elements.append(Paragraph("TECHNICAL NARRATIVE & ANALYSIS", header_style))
        # Use simple formatting for the narrative
        narrative_text = data['narrative'].replace("\n", "<br/>")
        elements.append(Paragraph(narrative_text, body_style))
        elements.append(Spacer(1, 20))
        
    # 4. Deep Thesis
    if data.get('thesis'):
        elements.append(Paragraph("INVESTMENT THESIS", header_style))
        thesis_text = data['thesis'].replace("\n", "<br/>")
        elements.append(Paragraph(thesis_text, body_style))
        elements.append(Spacer(1, 20))
        
    # 5. Key Algo Levels
    if data.get('levels'):
        elements.append(Paragraph("ALGORITHMIC EXECUTION LEVELS", header_style))
        lvl = data['levels']
        lvl_data = [[Paragraph("<b>LEVEL IDENTIFIER</b>", body_style), Paragraph("<b>PRICE LEVEL</b>", body_style)]]
        for k, v in lvl.items():
            lvl_data.append([k, f"{v:,.2f}"])
            
        lt = Table(lvl_data, colWidths=[2.5*inch, 2.5*inch])
        lt.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#333333")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#F9F9F9"), colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#D5D5D5")),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ]))
        elements.append(lt)

    # Footer Disclaimer
    elements.append(Spacer(1, 40))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    disclaimer = "<i>DISCLAIMER: This report is for institutional information purposes only and does not constitute financial advice. JD Capital Quantitative Investment assumes no liability for trading decisions based on this AI-generated synthesis.</i>"
    elements.append(Paragraph(disclaimer, ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=8, color=colors.grey, alignment=1)))

    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf
