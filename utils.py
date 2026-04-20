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
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=LETTER, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], color=colors.hexColor("#00FFFF"), alignment=1, spaceAfter=20)
    header_style = ParagraphStyle('HeaderStyle', parent=styles['Heading2'], color=colors.hexColor("#40E0FF"), spaceBefore=15, spaceAfter=10)
    body_style = styles['Normal']
    
    elements = []
    
    # 1. Title
    elements.append(Paragraph(f"BLOOMBERG TERMINAL PRO - INSTITUTIONAL BRIEF", title_style))
    elements.append(Paragraph(f"ASSET: {data['ticker']} | DATE: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", body_style))
    elements.append(Spacer(1, 12))
    
    # 2. Executive HUD
    elements.append(Paragraph("EXECUTIVE HUD METRICS", header_style))
    hud_data = [
        ["METRIC", "VALUE"],
        ["CURRENT PRICE", f"{data['price']:,.2f} ({data['pct']:.2f}%)"],
        ["AI ML SIGNAL", data['ml_signal']],
        ["QUANT REGIME", data['regime']],
    ]
    t = Table(hud_data, colWidths=[150, 250])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.hexColor("#12161F")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.hexColor("#F5F5F5")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    elements.append(t)
    
    # 3. AI Narrative
    if data.get('narrative'):
        elements.append(Paragraph("TECHNICAL NARRATIVE", header_style))
        # Split by bullets if possible or just wrap
        elements.append(Paragraph(data['narrative'].replace("\n", "<br/>"), body_style))
        
    # 4. Deep Thesis
    if data.get('thesis'):
        elements.append(Paragraph("INVESTMENT THESIS", header_style))
        elements.append(Paragraph(data['thesis'].replace("\n", "<br/>"), body_style))
        
    # 5. Key Algo Levels
    if data.get('levels'):
        elements.append(Paragraph("KEY ALGO LEVELS", header_style))
        lvl = data['levels']
        lvl_data = [["LEVEL NAME", "PRICE"]]
        for k, v in lvl.items():
            lvl_data.append([k, f"{v:,.2f}"])
            
        lt = Table(lvl_data, colWidths=[150, 100])
        lt.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.hexColor("#12161F")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        elements.append(lt)

    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf
