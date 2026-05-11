"""
frontend/app_nuevo.py — Dashboard Financiero · Diseño Moderno Premium
Versión Multi-Activo con Tabs
Ejecutar DESPUÉS de levantar el backend:
    cd backend && uvicorn main:app --reload --port 8000
    cd frontend && streamlit run app_nuevo.py
"""
import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm2
import os

# ─── CONFIGURACIÓN PÁGINA ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Financiero · Teoría de Riesgo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS PREMIUM MODERNO ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');

/* ── BASE ── */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
}

.stApp {
    background: linear-gradient(135deg, #0F0C29 0%, #1a1a3e 50%, #0d0d2b 100%);
    background-attachment: fixed;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a0533 0%, #0d1b4f 100%);
    border-right: 1px solid rgba(139,92,246,0.3);
}
[data-testid="stSidebar"] * { color: #E2D9F3 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stTextInput label { color: #C4B5FD !important; font-weight: 600 !important; font-size: 13px !important; }
[data-testid="stSidebar"] .stButton>button {
    background: linear-gradient(135deg, #7C3AED, #4F46E5) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 12px !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.4) !important;
    transition: all 0.3s ease !important;
}
[data-testid="stSidebar"] .stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.6) !important;
}

/* ── MÉTRICAS ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
    border: 1px solid rgba(139,92,246,0.35);
    border-radius: 16px;
    padding: 18px 22px;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.1);
    transition: all 0.3s ease;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(139,92,246,0.7);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(124,58,237,0.25);
}
[data-testid="stMetricLabel"] {
    color: #A78BFA !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: #F8FAFC !important;
    font-size: 28px !important;
    font-weight: 800 !important;
}
[data-testid="stMetricDelta"] { font-size: 13px !important; font-weight: 600 !important; }

/* ── TEXTO GENERAL ── */
h1, h2, h3, h4, h5, h6 { color: #F1F5F9 !important; font-family: 'Outfit', sans-serif !important; font-weight: 800 !important; }
p, li, span, label { color: #CBD5E1 !important; font-family: 'Outfit', sans-serif !important; }

/* ── CARDS ── */
.card {
    background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
    border: 1px solid rgba(139,92,246,0.25);
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    transition: all 0.3s ease;
}
.card:hover { border-color: rgba(139,92,246,0.55); transform: translateY(-3px); }

/* ── PAGE HEADER ── */
.page-header {
    background: linear-gradient(135deg, #7C3AED 0%, #4F46E5 50%, #0EA5E9 100%);
    padding: 28px 36px;
    border-radius: 20px;
    margin-bottom: 28px;
    box-shadow: 0 10px 40px rgba(124,58,237,0.4);
    position: relative;
    overflow: hidden;
}
.page-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.page-header h2 { color: white !important; margin: 0; font-size: 26px !important; font-weight: 800 !important; text-shadow: 0 2px 10px rgba(0,0,0,0.3); }
.page-header p  { color: rgba(255,255,255,0.85) !important; margin: 6px 0 0; font-size: 14px !important; font-weight: 400 !important; }

/* ── MÓDULO CARDS (portada) ── */
.mod-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02));
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 18px;
    padding: 22px 16px;
    text-align: center;
    transition: all 0.35s cubic-bezier(0.4,0,0.2,1);
    cursor: default;
    height: 150px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    margin-bottom: 12px;
}
.mod-card:hover {
    background: linear-gradient(135deg, rgba(124,58,237,0.2), rgba(79,70,229,0.15));
    border-color: rgba(139,92,246,0.7);
    transform: translateY(-5px) scale(1.02);
    box-shadow: 0 12px 40px rgba(124,58,237,0.3);
}
.mod-icon { font-size: 32px; margin-bottom: 8px; }
.mod-num  { font-size: 10px; font-weight: 700; color: #A78BFA !important; letter-spacing: 1.5px; text-transform: uppercase; }
.mod-name { font-size: 15px; font-weight: 700; color: #F1F5F9 !important; margin: 4px 0; }
.mod-desc { font-size: 11px; color: #94A3B8 !important; }

/* ── BADGES ── */
.badge-success { display:block; background:rgba(34,197,94,0.15); color:#4ADE80; border:1px solid rgba(74,222,128,0.4); border-radius:12px; padding:12px 18px; font-weight:700; font-size:14px; border-left: 4px solid #22C55E; }
.badge-error   { display:block; background:rgba(239,68,68,0.15);  color:#F87171; border:1px solid rgba(248,113,113,0.4); border-radius:12px; padding:12px 18px; font-weight:700; font-size:14px; border-left: 4px solid #EF4444; }
.badge-info    { display:block; background:rgba(59,130,246,0.15); color:#60A5FA; border:1px solid rgba(96,165,250,0.4); border-radius:12px; padding:12px 18px; font-weight:700; font-size:14px; border-left: 4px solid #3B82F6; }
.badge-warning { display:block; background:rgba(245,158,11,0.15); color:#FCD34D; border:1px solid rgba(252,211,77,0.4); border-radius:12px; padding:12px 18px; font-weight:700; font-size:14px; border-left: 4px solid #F59E0B; }

/* ── SIGNAL CARDS ── */
.signal-buy  { background:linear-gradient(135deg,rgba(34,197,94,0.2),rgba(16,185,129,0.1)); color:#4ADE80; border:2px solid rgba(74,222,128,0.5); border-radius:16px; padding:22px; text-align:center; font-size:24px; font-weight:800; box-shadow:0 8px 25px rgba(34,197,94,0.2); }
.signal-sell { background:linear-gradient(135deg,rgba(239,68,68,0.2),rgba(220,38,38,0.1)); color:#F87171; border:2px solid rgba(248,113,113,0.5); border-radius:16px; padding:22px; text-align:center; font-size:24px; font-weight:800; box-shadow:0 8px 25px rgba(239,68,68,0.2); }
.signal-neut { background:linear-gradient(135deg,rgba(59,130,246,0.2),rgba(37,99,235,0.1)); color:#60A5FA; border:2px solid rgba(96,165,250,0.5); border-radius:16px; padding:22px; text-align:center; font-size:24px; font-weight:800; box-shadow:0 8px 25px rgba(59,130,246,0.2); }

.sig-card   { background:rgba(255,255,255,0.05); border:1px solid rgba(139,92,246,0.3); border-radius:16px; padding:20px 14px; text-align:center; transition:all 0.3s ease; }
.sig-card:hover { border-color:rgba(139,92,246,0.7); transform:translateY(-3px); }
.sig-label  { font-size:10px; font-weight:700; color:#A78BFA !important; letter-spacing:1.5px; margin-bottom:8px; text-transform:uppercase; }
.sig-estado { font-weight:700; font-size:16px; color:#F1F5F9 !important; margin-top:8px; }
.sig-desc   { font-size:12px; color:#94A3B8 !important; margin-top:5px; }

/* ── SIDEBAR HEADER ── */
.sidebar-logo {
    text-align: center;
    padding: 20px 10px 10px;
}
.sidebar-logo h1 { font-size: 20px !important; font-weight: 900 !important; color: white !important; margin: 0; }
.sidebar-logo p  { font-size: 11px !important; color: #A78BFA !important; margin: 4px 0 0; }

/* ── RADIO MENU ── */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 10px;
    padding: 10px 14px !important;
    margin-bottom: 4px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #C4B5FD !important;
    transition: all 0.2s ease;
    cursor: pointer;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(124,58,237,0.2);
    border-color: rgba(139,92,246,0.5);
}

/* ── DATAFRAME ── */
div[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid rgba(139,92,246,0.3) !important; }

/* ── EXPANDER ── */
details { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(139,92,246,0.3) !important; border-radius: 14px !important; }
summary { color: #C4B5FD !important; font-weight: 600 !important; }

/* ── INFO / WARNING BOXES ── */
.stAlert { border-radius: 14px !important; border-left: 4px solid !important; background: rgba(255,255,255,0.05) !important; }

/* ── SPINNER ── */
.stSpinner > div { border-top-color: #7C3AED !important; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.04); border-radius: 12px; padding: 4px; border: 1px solid rgba(139,92,246,0.2); }
.stTabs [data-baseweb="tab"] { border-radius: 10px; color: #A78BFA !important; font-weight: 600 !important; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #7C3AED, #4F46E5) !important; color: white !important; }

/* ── DIVIDER ── */
hr { border-color: rgba(139,92,246,0.3) !important; }

/* ── STATUS INDICATOR ── */
.status-dot {
    display: inline-block; width: 10px; height: 10px;
    border-radius: 50%; margin-right: 8px;
    animation: pulse 2s infinite;
}
.status-dot.online  { background: #22C55E; box-shadow: 0 0 8px #22C55E; }
.status-dot.offline { background: #EF4444; box-shadow: 0 0 8px #EF4444; }
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* ── AUTO LOAD BANNER ── */
.autoload-banner {
    background: linear-gradient(90deg, rgba(124,58,237,0.15), rgba(14,165,233,0.1));
    border: 1px solid rgba(139,92,246,0.4);
    border-radius: 12px;
    padding: 14px 20px;
    margin-bottom: 20px;
    font-size: 13px;
    color: #C4B5FD !important;
    font-weight: 600;
}

/* ── TICKER GRID ── */
.ticker-row {
    display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px;
}
.ticker-chip {
    background: rgba(124,58,237,0.15);
    border: 1px solid rgba(139,92,246,0.4);
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 700;
    color: #C4B5FD !important;
    display: inline-block;
}

/* ── SELECT BOXES ── */
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(139,92,246,0.4) !important;
    border-radius: 10px !important;
    color: #F1F5F9 !important;
}
[data-testid="stSidebar"] .stMultiSelect > div > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(139,92,246,0.4) !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] .stTextInput > div > div > input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(139,92,246,0.4) !important;
    border-radius: 10px !important;
    color: #F1F5F9 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTES ───────────────────────────────────────────────────────────────
PRIMARY  = "#7C3AED"
ACCENT   = "#4F46E5"
CYAN     = "#0EA5E9"
SUCCESS  = "#22C55E"
DANGER   = "#EF4444"
WARNING  = "#F59E0B"
INFO     = "#3B82F6"
MUTED    = "#64748B"
GOLD     = "#F59E0B"

PLOT_TPL = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(15,12,41,0.0)",
    plot_bgcolor="rgba(255,255,255,0.03)",
    font=dict(family="Outfit, sans-serif", color="#CBD5E1", size=12),
    margin=dict(t=50, b=30, l=10, r=10),
    xaxis=dict(gridcolor="rgba(139,92,246,0.12)", linecolor="rgba(139,92,246,0.3)"),
    yaxis=dict(gridcolor="rgba(139,92,246,0.12)", linecolor="rgba(139,92,246,0.3)"),
)

def page_header(title, subtitle=""):
    st.markdown(
        f'<div class="page-header"><h2>{title}</h2>'
        + (f'<p>{subtitle}</p>' if subtitle else '') +
        '</div>', unsafe_allow_html=True
    )

def badge_html(text, kind="info"):
    st.markdown(f'<div class="badge-{kind}">{text}</div>', unsafe_allow_html=True)

def fmt_pval(p: float) -> str:
    if p is None: return "N/A"
    return f"{p:.2e}" if p < 0.0001 else f"{p:.4f}"

def section_title(icon, title):
    st.markdown(
        f'<h3 style="color:#C4B5FD;font-size:18px;font-weight:700;margin:24px 0 12px;">'
        f'{icon} {title}</h3>',
        unsafe_allow_html=True
    )

# ─── API ──────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

def api_post(endpoint, payload):
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=120)
        if r.status_code == 200:
            return r.json()
        st.error(f"Error API ({r.status_code}): {r.json().get('detail', r.text)}")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"❌ No se puede conectar al backend: {API_BASE}")
        return None

def api_get(endpoint):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

@st.cache_data(ttl=3600)
def get_tickers():
    data = api_get("/api/utils/tickers")
    return data["tickers"] if data else {
        "AAPL":"Apple","MSFT":"Microsoft","GOOGL":"Alphabet","AMZN":"Amazon",
        "TSLA":"Tesla","NVDA":"NVIDIA","JPM":"JPMorgan","BAC":"Bank of America",
        "GLD":"Gold ETF","BTC-USD":"Bitcoin"
    }

TICKERS = get_tickers()
TICKER_LIST = list(TICKERS.keys())

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div style="font-size:36px;margin-bottom:6px;">📊</div>
        <h1>Dashboard<br>Financiero</h1>
        <p>Teoría de Riesgo · 2025</p>
    </div>
    <hr style="margin: 12px 0;">
    """, unsafe_allow_html=True)

    opcion = st.radio("Módulo", [
        "🏠  Portada",
        "📈  Módulo 1 · Técnico",
        "📉  Módulo 2 · Rendimientos",
        "🌊  Módulo 3 · ARCH/GARCH",
        "⚖️  Módulo 4 · CAPM y Beta",
        "🛡️  Módulo 5 · VaR y CVaR",
        "🎯  Módulo 6 · Markowitz",
        "🚦  Módulo 7 · Señales ★",
        "🌍  Módulo 8 · Macro ★",
    ], label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)
    health = api_get("/api/utils/health")
    if health:
        st.markdown('<div class="badge-success"><span class="status-dot online"></span>Backend conectado</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge-error"><span class="status-dot offline"></span>Backend desconectado</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;font-size:11px;color:#64748B;">Sergio D. Huertas · Sergio A. Prieto<br>Javier Sierra</p>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PORTADA
# ══════════════════════════════════════════════════════════════════════════════
if opcion == "🏠  Portada":
    st.markdown("""
    <div style="background:linear-gradient(135deg,#7C3AED 0%,#4F46E5 40%,#0EA5E9 100%);
                border-radius:24px;padding:48px 44px;margin-bottom:32px;
                box-shadow:0 20px 60px rgba(124,58,237,0.45);position:relative;overflow:hidden;">
        <div style="position:absolute;top:-60px;right:-40px;width:260px;height:260px;
                    background:rgba(255,255,255,0.04);border-radius:50%;"></div>
        <div style="position:absolute;bottom:-80px;right:80px;width:180px;height:180px;
                    background:rgba(255,255,255,0.03);border-radius:50%;"></div>
        <h1 style="color:white;margin:0;font-size:38px;font-weight:900;line-height:1.2;">
            📊 Dashboard de<br>Análisis Financiero
        </h1>
        <p style="color:rgba(255,255,255,0.85);margin-top:12px;font-size:16px;font-weight:400;">
            Proyecto Final · Teoría de Riesgo · Análisis cuantitativo de riesgos y activos financieros
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="card">
            <div style="font-size:24px;margin-bottom:10px;">👥</div>
            <div style="font-size:16px;font-weight:800;color:#C4B5FD;margin-bottom:12px;">Integrantes</div>
            <div style="font-size:14px;color:#E2D9F3;margin-bottom:6px;">• Sergio David Huertas Ramírez</div>
            <div style="font-size:14px;color:#E2D9F3;margin-bottom:14px;">• Sergio Andrés Prieto Orjuela</div>
            <hr style="border-color:rgba(139,92,246,0.3);margin:12px 0;">
            <div style="font-size:13px;color:#94A3B8;">📚 <b style="color:#C4B5FD;">Materia:</b> Teoría de Riesgo</div>
            <div style="font-size:13px;color:#94A3B8;margin-top:6px;">👨‍🏫 <b style="color:#C4B5FD;">Profesor:</b> Javier Sierra</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card">
            <div style="font-size:24px;margin-bottom:10px;">🏗️</div>
            <div style="font-size:16px;font-weight:800;color:#C4B5FD;margin-bottom:12px;">Arquitectura</div>
            <div style="font-size:13px;color:#94A3B8;margin:6px 0;"><span style="background:rgba(124,58,237,0.25);border-radius:6px;padding:3px 8px;color:#C4B5FD;font-weight:700;">Frontend</span> Streamlit</div>
            <div style="font-size:13px;color:#94A3B8;margin:6px 0;"><span style="background:rgba(14,165,233,0.2);border-radius:6px;padding:3px 8px;color:#7DD3FC;font-weight:700;">Backend</span> FastAPI REST</div>
            <div style="font-size:13px;color:#94A3B8;margin:6px 0;"><span style="background:rgba(34,197,94,0.2);border-radius:6px;padding:3px 8px;color:#86EFAC;font-weight:700;">HTTP</span> requests</div>
            <div style="font-size:13px;color:#94A3B8;margin:6px 0;"><span style="background:rgba(245,158,11,0.2);border-radius:6px;padding:3px 8px;color:#FCD34D;font-weight:700;">Docs API</span> /docs</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card"><div style="font-size:24px;margin-bottom:10px;">📡</div><div style="font-size:16px;font-weight:800;color:#C4B5FD;margin-bottom:14px;">Estado del Sistema</div>', unsafe_allow_html=True)
        if health:
            badge_html("✅  API Operativa — Backend OK", "success")
        else:
            badge_html("❌  API Desconectada", "error")
        st.markdown(f"""
        <div style="margin-top:14px;font-size:12px;color:#64748B;">
            Activos disponibles: <b style="color:#C4B5FD;">{len(TICKERS)}</b><br>
            Módulos activos: <b style="color:#C4B5FD;">8</b>
        </div>
        </div>
        """, unsafe_allow_html=True)

    section_title("🔢", "Módulos del Proyecto")
    mods = [
        ("1","📈","Técnico","SMA · RSI · Bollinger","#7C3AED"),
        ("2","📉","Rendimientos","Estadísticas · Normalidad","#4F46E5"),
        ("3","🌊","ARCH/GARCH","Volatilidad condicional","#0EA5E9"),
        ("4","⚖️","CAPM y Beta","Línea característica","#06B6D4"),
        ("5","🛡️","VaR y CVaR","Riesgo de pérdida","#EF4444"),
        ("6","🎯","Markowitz","Frontera eficiente","#F59E0B"),
        ("7","🚦","Señales ★","Panel semáforo","#22C55E"),
        ("8","🌍","Macro ★","Alpha · Benchmark","#8B5CF6"),
    ]
    cols = st.columns(4)
    for i, (num, icon, name, desc, color) in enumerate(mods):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="mod-card">
                <div class="mod-icon">{icon}</div>
                <div class="mod-num">Módulo {num}</div>
                <div class="mod-name">{name}</div>
                <div class="mod-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    section_title("💼", "Activos Disponibles en el Sistema")
    chips_html = '<div class="ticker-row">'
    for t, name in TICKERS.items():
        chips_html += f'<span class="ticker-chip">{t}</span>'
    chips_html += '</div>'
    st.markdown(chips_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO 1 — ANÁLISIS TÉCNICO (MULTI-ACTIVO CON TABS)
# ══════════════════════════════════════════════════════════════════════════════
elif opcion == "📈  Módulo 1 · Técnico":
    page_header("📈 Módulo 1 · Análisis Técnico",
                "SMA, EMA, RSI, MACD y Bandas de Bollinger — múltiples activos en pestañas")

    with st.sidebar:
        st.markdown("### ⚙️ Parámetros")
        
        # Selección múltiple de activos con opción "Todos"
        all_opt = "✅ Seleccionar todos"
        selected = st.multiselect(
            "Activos", [all_opt] + TICKER_LIST, default=[all_opt], key="m1_tickers"
        )
        tickers = TICKER_LIST if (all_opt in selected or not selected) else selected
        
        periodo  = st.selectbox("Horizonte", ["1y","2y","5y"], index=1, key="m1_p")
        sma_c    = st.slider("SMA Corto", 5, 50, 20, key="m1_smac")
        sma_l    = st.slider("SMA Largo", 21, 200, 50, key="m1_smal")
        rsi_p    = st.slider("Período RSI", 5, 30, 14, key="m1_rsi")
        calcular = st.button("🔄 Calcular", type="primary", use_container_width=True)

    st.markdown(f"""
    <div class="autoload-banner">
        ⚡ Analizando <b>{len(tickers)} activos</b> · Horizonte: {periodo} · 
        Resultados organizados por pestañas
    </div>
    """, unsafe_allow_html=True)

    # Procesar múltiples activos
    results = {}
    if calcular:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, t in enumerate(tickers):
            status_text.text(f"🔄 Procesando {t} ({i+1}/{len(tickers)})...")
            payload = {
                "ticker": t, "periodo": periodo,
                "sma_corto": sma_c, "sma_largo": sma_l,
                "rsi_periodo": rsi_p, "bb_periodo": 20, "bb_std": 2.0
            }
            data = api_post("/api/tecnico/indicadores", payload)
            if data:
                results[t] = data
                st.session_state[f"m1_{t}"] = data
            progress_bar.progress((i + 1) / len(tickers))
        
        progress_bar.empty()
        status_text.empty()
        if results:
            st.success(f"✅ Análisis completado para {len(results)} activos")

    # Cargar datos desde sesión si no hay cálculo reciente
    if not results:
        for t in tickers:
            cached = st.session_state.get(f"m1_{t}")
            if cached:
                results[t] = cached

    # Mostrar resultados en tabs
    if results:
        tabs = st.tabs([f"{t} · {TICKERS[t]}" for t in results])
        
        for (ticker, data), tab in zip(results.items(), tabs):
            with tab:
                st.markdown(f"#### 📈 {TICKERS[ticker]} (`{ticker}`)")
                
                # Métricas
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("💵 Precio Actual", f"${data['ultimo_precio']:,.2f}")
                m2.metric("📈 Retorno Período", f"{data['retorno_periodo_pct']:.2f}%",
                          delta=f"{data['retorno_periodo_pct']:.2f}%")
                m3.metric("⚡ RSI Actual", f"{data['rsi_actual']:.1f}")
                m4.metric("📊 Volatilidad", f"{data['volatilidad_diaria_pct']:.2f}%")

                df = pd.DataFrame(data["datos"])
                df["fecha"] = pd.to_datetime(df["fecha"])

                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.72, 0.28], vertical_spacing=0.04,
                    subplot_titles=("Precio e Indicadores", "RSI (Relative Strength Index)"))
                fig.add_trace(go.Scatter(x=df["fecha"], y=df["close"],
                    name="Precio", line=dict(color=PRIMARY, width=2.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df["fecha"], y=df[f"sma_{sma_c}"],
                    name=f"SMA {sma_c}", line=dict(color=WARNING, width=1.5, dash="dot")), row=1, col=1)
                fig.add_trace(go.Scatter(x=df["fecha"], y=df[f"sma_{sma_l}"],
                    name=f"SMA {sma_l}", line=dict(color=DANGER, width=1.5, dash="dot")), row=1, col=1)
                fig.add_trace(go.Scatter(x=df["fecha"], y=df["bb_upper"],
                    name="BB+", line=dict(color="#8B5CF6", width=1, dash="dot")), row=1, col=1)
                fig.add_trace(go.Scatter(x=df["fecha"], y=df["bb_lower"],
                    name="BB−", line=dict(color="#8B5CF6", width=1, dash="dot"),
                    fill="tonexty", fillcolor="rgba(139,92,246,0.06)"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df["fecha"], y=df["rsi"],
                    name="RSI", line=dict(color=CYAN, width=2)), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color=DANGER, row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color=SUCCESS, row=2, col=1)
                fig.update_layout(height=600, hovermode="x unified", **PLOT_TPL)
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("📘 Interpretación de Indicadores"):
                    cc = st.columns(3)
                    rsi_v = data['rsi_actual']
                    estado_rsi = "sobrecompra ⚠️" if rsi_v > 70 else "sobreventa 🔻" if rsi_v < 30 else "zona neutral ✅"
                    with cc[0]:
                        st.info(f"**SMA:** Cruce SMA{sma_c} sobre SMA{sma_l} → señal de tendencia alcista.")
                    with cc[1]:
                        st.info(f"**RSI = {rsi_v:.1f}:** Activo en {estado_rsi}.")
                    with cc[2]:
                        st.info("**Bollinger:** Expansión de bandas indica mayor volatilidad implícita.")
    else:
        st.markdown('<div class="badge-info">👆 Selecciona activos en el panel izquierdo y pulsa <b>Calcular</b> para visualizar los indicadores.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO 2 — RENDIMIENTOS (MULTI-ACTIVO CON TABS)
# ══════════════════════════════════════════════════════════════════════════════
elif opcion == "📉  Módulo 2 · Rendimientos":
    page_header("📉 Módulo 2 · Rendimientos",
                "Caracterización estadística, pruebas de normalidad y hechos estilizados — múltiples activos")

    with st.sidebar:
        st.markdown("### ⚙️ Parámetros")
        
        all_opt = "✅ Seleccionar todos"
        selected = st.multiselect("Activos", [all_opt] + TICKER_LIST, default=[all_opt], key="m2_tickers")
        tickers = TICKER_LIST if (all_opt in selected or not selected) else selected
        
        periodo  = st.selectbox("Horizonte", ["1y","2y","5y"], index=1, key="m2_p")
        tipo     = st.radio("Tipo rendimiento", ["Simple","Logarítmico"], key="m2_tipo")
        calcular = st.button("🔄 Calcular", type="primary", use_container_width=True)

    st.markdown(f'<div class="autoload-banner">⚡ Análisis de rendimientos para <b>{len(tickers)} activos</b> · Tipo: {tipo}</div>', unsafe_allow_html=True)

    results = {}
    if calcular:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, t in enumerate(tickers):
            status_text.text(f"🔄 Procesando {t} ({i+1}/{len(tickers)})...")
            data = api_post("/api/rendimientos/estadisticas",
                            {"ticker": t, "periodo": periodo, "tipo": tipo})
            if data:
                results[t] = data
                st.session_state[f"m2_{t}"] = data
            progress_bar.progress((i + 1) / len(tickers))
        
        progress_bar.empty()
        status_text.empty()
        if results:
            st.success(f"✅ Análisis completado para {len(results)} activos")

    if not results:
        for t in tickers:
            cached = st.session_state.get(f"m2_{t}")
            if cached:
                results[t] = cached

    if results:
        tabs = st.tabs([f"{t} · {TICKERS[t]}" for t in results])
        
        for (ticker, data), tab in zip(results.items(), tabs):
            with tab:
                st.markdown(f"#### 📊 {TICKERS[ticker]} (`{ticker}`)")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("μ Media", f"{data['media']:.5f}")
                m2.metric("σ Desv. Estándar", f"{data['desviacion']:.5f}")
                m3.metric("Asimetría (Skew)", f"{data['asimetria']:.3f}")
                m4.metric("Curtosis (Exceso)", f"{data['curtosis']:.3f}")

                rend = [r["rendimiento"] for r in data["datos_rendimientos"]]
                fechas = [r["fecha"] for r in data["datos_rendimientos"]]

                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("Histograma + Curva Normal Teórica",
                                    "Gráfico Q-Q vs Normal",
                                    "Boxplot de Rendimientos",
                                    "Serie Temporal (Volatility Clustering)"),
                    vertical_spacing=0.14, horizontal_spacing=0.1
                )
                fig.add_trace(go.Histogram(x=rend, histnorm="probability density",
                    name="Rendimientos", marker_color=PRIMARY, opacity=0.7), row=1, col=1)
                x_r = np.linspace(min(rend), max(rend), 200)
                y_n = norm.pdf(x_r, data["media"], data["desviacion"])
                fig.add_trace(go.Scatter(x=x_r, y=y_n, mode="lines",
                    name="Normal Teórica", line=dict(color=DANGER, width=2.5)), row=1, col=1)
                qq = sm2.ProbPlot(np.array(rend), dist=norm, fit=True)
                tq, sq = qq.theoretical_quantiles, np.sort(rend)
                fig.add_trace(go.Scatter(x=tq, y=sq, mode="markers",
                    name="Q-Q", marker=dict(size=3, color=CYAN, opacity=0.6)), row=1, col=2)
                lr = [float(min(tq)), float(max(tq))]
                fig.add_trace(go.Scatter(x=lr, y=lr, mode="lines",
                    name="45°", line=dict(color=DANGER, dash="dash", width=2)), row=1, col=2)
                fig.add_trace(go.Box(y=rend, name="Rend.",
                    boxpoints="outliers", marker_color=PRIMARY, line_color=CYAN), row=2, col=1)
                fig.add_trace(go.Scatter(x=fechas, y=rend, mode="lines",
                    name="Retornos", line=dict(width=1, color=PRIMARY)), row=2, col=2)
                fig.update_layout(height=680, showlegend=False, **PLOT_TPL)
                st.plotly_chart(fig, use_container_width=True)

                section_title("🔬", "Pruebas de Normalidad")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown(f"**Jarque-Bera** · p-valor: `{fmt_pval(data['jarque_bera_pvalor'])}`")
                    st.markdown("H₀: La distribución es Normal")
                    if data["es_normal_jb"]:
                        badge_html("✅ No rechazamos H₀ · Distribución Normal", "success")
                    else:
                        badge_html("❌ Rechazamos H₀ · No sigue distribución Normal", "error")
                    st.markdown('</div>', unsafe_allow_html=True)
                with c2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown(f"**Shapiro-Wilk** · p-valor: `{fmt_pval(data['shapiro_pvalor'])}`")
                    st.markdown("H₀: La distribución es Normal")
                    if data["es_normal_sw"]:
                        badge_html("✅ No rechazamos H₀ · Distribución Normal", "success")
                    else:
                        badge_html("❌ Rechazamos H₀ · No sigue distribución Normal", "error")
                    st.markdown('</div>', unsafe_allow_html=True)

                with st.expander("📖 Discusión de Hechos Estilizados"):
                    kurt = data['curtosis']; asim = data['asimetria']
                    st.markdown(f"""
**Colas Pesadas (Leptocurtosis):** Curtosis de exceso = **{kurt:.2f}**.
{"Las colas son significativamente más anchas que la normal — mayor probabilidad de eventos extremos." if abs(kurt) > 1 else "Distribución cercana a la normal en curtosis."}

**Agrupamiento de Volatilidad:** La serie temporal muestra bloques de alta y baja volatilidad, motivando el uso de modelos GARCH.

**Efecto Apalancamiento:** Asimetría = **{asim:.2f}** → {"sesgo negativo: retornos negativos más extremos." if asim < -0.2 else "sesgo positivo." if asim > 0.2 else "distribución aproximadamente simétrica."}
                    """)
    else:
        st.markdown('<div class="badge-info">👆 Selecciona activos y pulsa <b>Calcular</b>.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO 3 — ARCH/GARCH (MULTI-ACTIVO CON TABS)
# ══════════════════════════════════════════════════════════════════════════════
elif opcion == "🌊  Módulo 3 · ARCH/GARCH":
    page_header("🌊 Módulo 3 · ARCH/GARCH",
                "Modelado de volatilidad condicional y pronóstico de riesgo — múltiples activos")

    with st.sidebar:
        st.markdown("### ⚙️ Parámetros")
        
        all_opt = "✅ Seleccionar todos"
        selected = st.multiselect("Activos", [all_opt] + TICKER_LIST, default=[all_opt], key="m3_tickers")
        tickers = TICKER_LIST if (all_opt in selected or not selected) else selected
        
        dist = st.selectbox("Distribución", ["Normal","t-Student","Skewed t-Student"], index=1, key="m3_dist")
        horizonte = st.slider("Días a pronosticar", 5, 30, 10, key="m3_hor")
        calcular = st.button("🔄 Ajustar Modelos", type="primary", use_container_width=True)

    st.markdown(f'<div class="autoload-banner">⚡ Modelos ARCH/GARCH para <b>{len(tickers)} activos</b> · Dist: {dist} · Horizonte: {horizonte} días</div>', unsafe_allow_html=True)

    with st.expander("💡 Justificación: ¿Por qué volatilidad condicional?", expanded=False):
        st.markdown("""
Los modelos **ARCH/GARCH** capturan el *agrupamiento de volatilidad* en series financieras.
- **ARCH(1):** La varianza depende del error cuadrado del período anterior.
- **GARCH(1,1):** Añade la varianza condicional rezagada — más parsimónico.
- **EGARCH(1,1):** Captura asimetría (efecto apalancamiento) sin restricción de positividad.
        """)

    results = {}
    if calcular:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, t in enumerate(tickers):
            status_text.text(f"🔄 Ajustando {t} ({i+1}/{len(tickers)})...")
            data = api_post("/api/garch/volatilidad",
                            {"ticker": t, "horizonte": horizonte, "distribucion": dist})
            if data:
                results[t] = data
                st.session_state[f"m3_{t}"] = data
            progress_bar.progress((i + 1) / len(tickers))
        
        progress_bar.empty()
        status_text.empty()
        if results:
            st.success(f"✅ Modelos ajustados para {len(results)} activos")

    if not results:
        for t in tickers:
            cached = st.session_state.get(f"m3_{t}")
            if cached:
                results[t] = cached

    if results:
        tabs = st.tabs([f"{t} · {TICKERS[t]}" for t in results])
        
        for (ticker, data), tab in zip(results.items(), tabs):
            with tab:
                st.markdown(f"#### 🌊 {TICKERS[ticker]} (`{ticker}`)")
                
                section_title("📊", "Comparativa de Modelos")
                df_mod = pd.DataFrame(data["comparativa_modelos"])
                cols_num = [c for c in ["log_likelihood","aic","bic"] if c in df_mod.columns]
                if cols_num:
                    styled = df_mod.set_index("modelo").style
                    if "aic" in cols_num and "bic" in cols_num:
                        styled = styled.highlight_min(subset=["aic","bic"], color="#1a3a1a")
                    if "log_likelihood" in cols_num:
                        styled = styled.highlight_max(subset=["log_likelihood"], color="#1a3a1a")
                    st.dataframe(styled.format({c: "{:.2f}" for c in cols_num}), use_container_width=True)
                    st.caption("✅ Verde oscuro = mejor valor (AIC/BIC menores; Log-Likelihood mayor)")

                section_title("🔍", "Diagnóstico GARCH(1,1)")
                col1, col2 = st.columns([3, 1])
                with col1:
                    fig_r = go.Figure()
                    fig_r.add_trace(go.Scatter(y=data["residuos_std"], mode="lines",
                        name="Residuos Estandarizados", line=dict(color=CYAN, width=1.2)))
                    fig_r.add_hline(y= 2, line_dash="dash", line_color=DANGER, opacity=0.5)
                    fig_r.add_hline(y=-2, line_dash="dash", line_color=DANGER, opacity=0.5)
                    fig_r.update_layout(title="Residuos Estandarizados · GARCH(1,1)",
                                        yaxis_title="Residuo Std", height=320, **PLOT_TPL)
                    st.plotly_chart(fig_r, use_container_width=True)
                with col2:
                    pval = data["jb_residuos_pvalor"]
                    st.metric("JB p-valor (residuos)", f"{pval:.4f}")
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    if pval < 0.05:
                        badge_html("⚠️ Residuos no normales", "warning")
                        st.markdown("<br><small style='color:#64748B;'>Frecuente en finanzas — usar distribución t-Student.</small>", unsafe_allow_html=True)
                    else:
                        badge_html("✅ Residuos normales", "success")
                    st.markdown('</div>', unsafe_allow_html=True)

                section_title("🔮", f"Pronóstico de Volatilidad · {horizonte} días")
                vol_fc = data["pronostico_volatilidad"]
                dias = list(range(1, horizonte + 1))
                fig_f = go.Figure()
                fig_f.add_trace(go.Scatter(x=dias, y=vol_fc, mode="lines+markers",
                    name="Vol. Pronosticada (%)",
                    line=dict(color=PRIMARY, width=2.5),
                    marker=dict(size=8, color=CYAN, line=dict(color=PRIMARY, width=2)),
                    fill="tozeroy", fillcolor="rgba(124,58,237,0.1)"))
                fig_f.update_layout(xaxis_title="Días hacia adelante", yaxis_title="Volatilidad (%)", height=360, **PLOT_TPL)
                st.plotly_chart(fig_f, use_container_width=True)
                badge_html(f"📊 Vol. promedio pronosticada: {np.mean(vol_fc):.3f}% diario — {np.mean(vol_fc)*np.sqrt(252):.2f}% anualizado.", "info")
    else:
        st.markdown('<div class="badge-info">👆 Selecciona activos y pulsa <b>Ajustar Modelos</b>.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO 4 — CAPM (MULTI-ACTIVO CON TABS)
# ══════════════════════════════════════════════════════════════════════════════
elif opcion == "⚖️  Módulo 4 · CAPM y Beta":
    page_header("⚖️ Módulo 4 · CAPM y Beta",
                "Línea característica del activo, Beta y retorno esperado CAPM — múltiples activos")

    with st.sidebar:
        st.markdown("### ⚙️ Parámetros")
        
        all_opt = "✅ Seleccionar todos"
        selected = st.multiselect("Activos", [all_opt] + TICKER_LIST, default=[all_opt], key="m4_tickers")
        tickers = TICKER_LIST if (all_opt in selected or not selected) else selected
        
        periodo = st.selectbox("Horizonte", ["1y","2y"], index=1, key="m4_p")
        calcular = st.button("🔄 Calcular", type="primary", use_container_width=True)

    st.markdown(f'<div class="autoload-banner">⚡ CAPM y Beta para <b>{len(tickers)} activos</b> vs S&P 500 · Horizonte: {periodo}</div>', unsafe_allow_html=True)

    results = {}
    if calcular:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, t in enumerate(tickers):
            status_text.text(f"🔄 Calculando {t} ({i+1}/{len(tickers)})...")
            data = api_post("/api/capm/beta",
                            {"ticker": t, "benchmark": "^GSPC", "periodo": periodo})
            if data:
                results[t] = data
                st.session_state[f"m4_{t}"] = data
            progress_bar.progress((i + 1) / len(tickers))
        
        progress_bar.empty()
        status_text.empty()
        if results:
            st.success(f"✅ Cálculo completado para {len(results)} activos")

    if not results:
        for t in tickers:
            cached = st.session_state.get(f"m4_{t}")
            if cached:
                results[t] = cached

    if results:
        tabs = st.tabs([f"{t} · {TICKERS[t]}" for t in results])
        
        for (ticker, data), tab in zip(results.items(), tabs):
            with tab:
                st.markdown(f"#### ⚖️ {TICKERS[ticker]} (`{ticker}`)")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    df_r = pd.DataFrame(data["datos_regresion"])
                    x_line = np.array([df_r["market"].min(), df_r["market"].max()])
                    alpha_r = data["retorno_esperado_pct"] - data["beta"] * data["rm_anual_pct"]
                    y_line = alpha_r + data["beta"] * x_line
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_r["market"], y=df_r["asset"], mode="markers",
                        name="Observaciones", marker=dict(color=PRIMARY, size=5, opacity=0.5)))
                    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines",
                        name=f"Regresión (β={data['beta']:.2f})",
                        line=dict(color=CYAN, width=3)))
                    fig.update_layout(title=f"Línea Característica {ticker} vs S&P 500",
                        xaxis_title="Rendimiento Mercado", yaxis_title=f"Rendimiento {ticker}",
                        height=430, **PLOT_TPL)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.metric("Beta (β)", f"{data['beta']:.4f}")
                    st.metric("Retorno CAPM", f"{data['retorno_esperado_pct']:.2%}")
                    st.metric("R²", f"{data['r_squared']:.4f}")
                    st.metric("Rf (10Y)", f"{data['rf_anual_pct']:.2%}")
                    st.metric("Rm Anual", f"{data['rm_anual_pct']:.2%}")
                    st.markdown("---")
                    cls = data["clasificacion"]
                    if cls == "Agresivo":
                        badge_html("🔴 Agresivo (β > 1.1)", "error")
                    elif cls == "Defensivo":
                        badge_html("🟢 Defensivo (β < 0.9)", "success")
                    else:
                        badge_html("🔵 Neutro (β ≈ 1)", "info")
                    st.markdown('</div>', unsafe_allow_html=True)

                beta = data["beta"]
                st.info(
                    f"📌 Con β = {beta:.2f}, el activo **{'amplifica' if beta > 1 else 'amortigua'}** "
                    f"los movimientos del mercado un **{abs((beta-1)*100):.1f}%**. "
                    f"Retorno esperado CAPM: **{data['retorno_esperado_pct']:.2%}** anual."
                )
    else:
        st.markdown('<div class="badge-info">👆 Selecciona activos y pulsa <b>Calcular</b>.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO 5 — VaR y CVaR (MULTI-ACTIVO CON TABS)
# ══════════════════════════════════════════════════════════════════════════════
elif opcion == "🛡️  Módulo 5 · VaR y CVaR":
    page_header("🛡️ Módulo 5 · VaR y CVaR",
                "Pérdida potencial máxima: Paramétrico, Histórico y Montecarlo — múltiples activos")

    with st.sidebar:
        st.markdown("### ⚙️ Parámetros")
        
        all_opt = "✅ Seleccionar todos"
        selected = st.multiselect("Activos", [all_opt] + TICKER_LIST, default=[all_opt], key="m5_tickers")
        tickers = TICKER_LIST if (all_opt in selected or not selected) else selected
        
        confianza = st.select_slider("Nivel de Confianza", [0.95, 0.99], value=0.95, key="m5_conf")
        inversion = st.number_input("Inversión (USD)", value=10000, step=1000, key="m5_inv")
        calcular = st.button("🔄 Calcular", type="primary", use_container_width=True)

    st.markdown(f'<div class="autoload-banner">⚡ VaR/CVaR para <b>{len(tickers)} activos</b> · Confianza: {confianza:.0%} · Inversión: ${inversion:,.0f}</div>', unsafe_allow_html=True)

    results = {}
    if calcular:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, t in enumerate(tickers):
            status_text.text(f"🔄 Calculando {t} ({i+1}/{len(tickers)})...")
            data = api_post("/api/var/calcular", {
                "ticker": t, "confianza": confianza,
                "inversion": inversion, "n_sims": 10000
            })
            if data:
                results[t] = data
                st.session_state[f"m5_{t}"] = data
            progress_bar.progress((i + 1) / len(tickers))
        
        progress_bar.empty()
        status_text.empty()
        if results:
            st.success(f"✅ Cálculo completado para {len(results)} activos")

    if not results:
        for t in tickers:
            cached = st.session_state.get(f"m5_{t}")
            if cached:
                results[t] = cached

    if results:
        tabs = st.tabs([f"{t} · {TICKERS[t]}" for t in results])
        
        for (ticker, data), tab in zip(results.items(), tabs):
            with tab:
                st.markdown(f"#### 🛡️ {TICKERS[ticker]} (`{ticker}`)")
                
                rend = data["datos_rendimientos"]
                vhist = data["var_historico_diario_pct"]
                cvar = data["cvar_diario_pct"]

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("VaR Param. Diario", f"{data['var_parametrico_diario_pct']:.2%}", delta_color="inverse")
                m2.metric("VaR Histórico", f"{vhist:.2%}", delta_color="inverse")
                m3.metric("VaR Montecarlo", f"{data['var_montecarlo_diario_pct']:.2%}", delta_color="inverse")
                m4.metric("CVaR (Expected Shortfall)", f"{cvar:.2%}", delta_color="inverse")

                fig = go.Figure()
                fig.add_trace(go.Histogram(x=rend, histnorm="probability density",
                    name="Distribución", marker_color=PRIMARY, opacity=0.65, nbinsx=80))
                x_loss = [v for v in rend if v <= vhist]
                fig.add_trace(go.Histogram(x=x_loss, histnorm="probability density",
                    name="Cola de pérdida", marker_color=DANGER, opacity=0.6, nbinsx=30))
                fig.add_vline(x=vhist, line_dash="dash", line_color=DANGER, line_width=2.5,
                              annotation_text=f"VaR {confianza:.0%}", annotation_font_color=DANGER)
                fig.add_vline(x=cvar, line_dash="dot", line_color=WARNING, line_width=2.5,
                              annotation_text="CVaR", annotation_font_color=WARNING)
                fig.update_layout(title="Distribución de Rendimientos y Zonas de Riesgo",
                    xaxis_title="Rendimiento Diario", yaxis_title="Densidad",
                    height=400, **PLOT_TPL)
                st.plotly_chart(fig, use_container_width=True)

                section_title("📋", "Tabla Comparativa de Metodologías")
                tabla = pd.DataFrame({
                    "Metodología": ["Paramétrico (Diario)","Paramétrico (Anual)",
                                    "Simulación Histórica","Montecarlo (10k)","CVaR / Expected Shortfall"],
                    "VaR / CVaR (%)": [
                        f"{data['var_parametrico_diario_pct']:.2%}",
                        f"{data['var_parametrico_anual_pct']:.2%}",
                        f"{data['var_historico_diario_pct']:.2%}",
                        f"{data['var_montecarlo_diario_pct']:.2%}",
                        f"{data['cvar_diario_pct']:.2%}",
                    ],
                    "Pérdida USD": [
                        f"${data['perdida_param_usd']:,.2f}", "—",
                        f"${data['perdida_hist_usd']:,.2f}",
                        f"${data['perdida_mc_usd']:,.2f}",
                        f"${data['perdida_cvar_usd']:,.2f}",
                    ]
                })
                st.dataframe(tabla.set_index("Metodología"), use_container_width=True)
                st.info(
                    f"📌 Con **{confianza:.0%}** de confianza, la pérdida máxima diaria para "
                    f"**${inversion:,.0f}** es **${data['perdida_hist_usd']:,.2f}** (histórico). "
                    f"CVaR (peor escenario promedio): **${data['perdida_cvar_usd']:,.2f}**."
                )
    else:
        st.markdown('<div class="badge-info">👆 Selecciona activos y pulsa <b>Calcular</b>.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO 6 — MARKOWITZ (Ya es multi-activo, se mantiene con ajuste visual)
# ══════════════════════════════════════════════════════════════════════════════
elif opcion == "🎯  Módulo 6 · Markowitz":
    page_header("🎯 Módulo 6 · Frontera Eficiente de Markowitz",
                "Optimización Media-Varianza: Máximo Sharpe y Mínima Varianza")

    with st.sidebar:
        st.markdown("### ⚙️ Parámetros")
        tickers_sel = st.multiselect("Activos", TICKER_LIST, default=TICKER_LIST[:5], key="m6_tickers")
        num_port = st.select_slider("Simulaciones", [1000, 5000, 10000], value=5000, key="m6_nport")
        calcular = st.button("🔄 Calcular Frontera", type="primary", use_container_width=True)

    n_sel = len(tickers_sel)
    st.markdown(f'<div class="autoload-banner">⚡ Frontera eficiente para <b>{n_sel} activos</b>: {", ".join(tickers_sel[:6])}{"..." if n_sel > 6 else ""}</div>', unsafe_allow_html=True)

    if calcular:
        if len(tickers_sel) < 2:
            st.warning("Selecciona al menos 2 activos.")
        else:
            with st.spinner("Simulando portafolios Montecarlo..."):
                data = api_post("/api/markowitz/frontera", {
                    "tickers": tickers_sel, "num_portafolios": num_port, "periodo": "2y"
                })
            if data:
                st.session_state["m6_data"] = data

    data = st.session_state.get("m6_data")
    if data:
        tab1, tab2, tab3 = st.tabs(["📈 Frontera Eficiente", "🔗 Correlaciones", "📋 Portafolios Óptimos"])

        with tab1:
            fe = pd.DataFrame(data["frontera_eficiente"])
            ms = data["portafolio_max_sharpe"]
            mv = data["portafolio_min_varianza"]
            fig_fe = px.scatter(fe, x="volatilidad", y="retorno", color="sharpe",
                color_continuous_scale="Plasma", opacity=0.5,
                labels={"volatilidad":"Riesgo (σ anual)","retorno":"Retorno anual","sharpe":"Sharpe"})
            fig_fe.add_trace(go.Scatter(
                x=[ms["volatilidad_anual_pct"]], y=[ms["retorno_anual_pct"]],
                mode="markers", name="⭐ Máx. Sharpe",
                marker=dict(color=GOLD, size=18, symbol="star",
                            line=dict(color="white", width=1.5))))
            fig_fe.add_trace(go.Scatter(
                x=[mv["volatilidad_anual_pct"]], y=[mv["retorno_anual_pct"]],
                mode="markers", name="💎 Mín. Varianza",
                marker=dict(color=SUCCESS, size=15, symbol="diamond",
                            line=dict(color="white", width=1.5))))
            fig_fe.update_layout(height=480, **PLOT_TPL)
            st.plotly_chart(fig_fe, use_container_width=True)

        with tab2:
            corr_df = pd.DataFrame(data["matriz_correlacion"])
            fig_c = px.imshow(corr_df, text_auto=".2f",
                              color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            fig_c.update_layout(height=400, **PLOT_TPL)
            st.plotly_chart(fig_c, use_container_width=True)

        with tab3:
            c1, c2 = st.columns(2)
            for col, port, label, color in [
                (c1, ms, "⭐ Máximo Sharpe", "rgba(245,158,11,0.2)"),
                (c2, mv, "💎 Mínima Varianza", "rgba(34,197,94,0.2)")
            ]:
                with col:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown(f"**{label}**")
                    st.metric("Sharpe", f"{port['sharpe_ratio']:.4f}")
                    st.metric("Retorno", f"{port['retorno_anual_pct']:.2%}")
                    st.metric("Volatilidad", f"{port['volatilidad_anual_pct']:.2%}")
                    pesos_df = pd.DataFrame(list(port["pesos"].items()),
                                            columns=["Activo","Peso"]).set_index("Activo")
                    st.dataframe(pesos_df.style.format("{:.2%}").bar(color=color),
                                 use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge-info">👆 Selecciona los activos y pulsa <b>Calcular Frontera</b>.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO 7 — SEÑALES (MULTI-ACTIVO CON TABS)
# ══════════════════════════════════════════════════════════════════════════════
elif opcion == "🚦  Módulo 7 · Señales ★":
    page_header("🚦 Módulo 7 · Panel de Señales Algorítmicas",
                "Sistema automático de alertas basado en indicadores técnicos — múltiples activos")

    with st.sidebar:
        st.markdown("### ⚙️ Parámetros")
        
        all_opt = "✅ Seleccionar todos"
        selected = st.multiselect("Activos", [all_opt] + TICKER_LIST, default=[all_opt], key="m7_tickers")
        tickers = TICKER_LIST if (all_opt in selected or not selected) else selected
        
        rsi_up = st.slider("Sobrecompra RSI", 60, 80, 70, key="m7_rsiup")
        rsi_down = st.slider("Sobreventa RSI", 20, 40, 30, key="m7_rsido")
        calcular = st.button("🔄 Actualizar", type="primary", use_container_width=True)

    st.markdown(f'<div class="autoload-banner">⚡ Panel de señales para <b>{len(tickers)} activos</b> — RSI [{rsi_down}, {rsi_up}]</div>', unsafe_allow_html=True)

    results = {}
    if calcular:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, t in enumerate(tickers):
            status_text.text(f"🔄 Generando señales para {t} ({i+1}/{len(tickers)})...")
            data = api_post("/api/senales/panel",
                            {"ticker": t, "rsi_up": rsi_up, "rsi_down": rsi_down})
            if data:
                results[t] = data
                st.session_state[f"m7_{t}"] = data
            progress_bar.progress((i + 1) / len(tickers))
        
        progress_bar.empty()
        status_text.empty()
        if results:
            st.success(f"✅ Señales generadas para {len(results)} activos")

    if not results:
        for t in tickers:
            cached = st.session_state.get(f"m7_{t}")
            if cached:
                results[t] = cached

    if results:
        tabs = st.tabs([f"{t} · {TICKERS[t]}" for t in results])
        
        for (ticker, data), tab in zip(results.items(), tabs):
            with tab:
                st.markdown(f"#### 🚦 {TICKERS[ticker]} (`{ticker}`)")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("💵 Precio Actual", f"${data['precio_actual']:,.2f}")
                m2.metric("⚡ RSI Actual", f"{data['rsi_actual']:.2f}")
                m3.metric("📡 Señales", str(len(data["senales"])))

                gs = data["señal_global"]
                if gs == "COMPRA":
                    st.markdown(f'<div class="signal-buy">🟢 SEÑAL GLOBAL: {gs}</div>', unsafe_allow_html=True)
                elif gs == "VENTA":
                    st.markdown(f'<div class="signal-sell">🔴 SEÑAL GLOBAL: {gs}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="signal-neut">🔵 SEÑAL GLOBAL: {gs}</div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                emoji_map = {"green":"🟢","red":"🔴","blue":"🔵"}
                cols = st.columns(min(len(data["senales"]), 4))
                for i, sig in enumerate(data["senales"]):
                    with cols[i % len(cols)]:
                        st.markdown(f"""
                        <div class="sig-card">
                            <div class="sig-label">{sig["indicador"].upper()}</div>
                            <div style="font-size:28px;margin:4px 0;">{emoji_map.get(sig["color"],"🔵")}</div>
                            <div class="sig-estado">{sig["estado"]}</div>
                            <div class="sig-desc">{sig["descripcion"]}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                badge_html("⚠️ Señales automáticas — no constituyen recomendación de inversión.", "warning")
    else:
        st.markdown('<div class="badge-info">👆 Selecciona activos y pulsa <b>Actualizar</b>.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO 8 — MACRO (Ya es multi-activo, se mantiene con ajuste visual)
# ══════════════════════════════════════════════════════════════════════════════
elif opcion == "🌍  Módulo 8 · Macro ★":
    page_header("🌍 Módulo 8 · Macro y Benchmark",
                "Contexto macroeconómico y comparativa portafolio vs S&P 500")

    with st.sidebar:
        st.markdown("### ⚙️ Parámetros")
        tickers_sel = st.multiselect("Activos del portafolio", TICKER_LIST,
                                     default=TICKER_LIST[:5], key="m8_tickers")
        calcular = st.button("🔄 Calcular", type="primary", use_container_width=True)

    st.markdown(f'<div class="autoload-banner">⚡ Comparativa portafolio ({len(tickers_sel)} activos) vs S&P 500</div>', unsafe_allow_html=True)

    section_title("🌐", "Indicadores Macroeconómicos")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tasa Libre de Riesgo (10Y)", "4.32%", "+0.05 bps")
    m2.metric("Inflación (CPI)", "3.10%", "-0.10%", delta_color="inverse")
    m3.metric("TRM COP/USD", "$4,000", "+15")
    m4.metric("Fed Funds Rate", "5.25%", "Sin cambio")

    if calcular:
        if not tickers_sel:
            st.warning("Selecciona al menos un activo.")
        else:
            with st.spinner("Calculando comparativa..."):
                data = api_post("/api/macro/benchmark", {
                    "tickers": tickers_sel, "benchmark": "^GSPC", "periodo": "1y"
                })
            if data:
                st.session_state["m8_data"] = data

    data = st.session_state.get("m8_data")
    if data:
        port = pd.DataFrame(data["portafolio_acumulado"])
        bench = pd.DataFrame(data["benchmark_acumulado"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=port["fecha"], y=port["valor"],
            name="Mi Portafolio",
            line=dict(color=PRIMARY, width=3),
            fill="tozeroy", fillcolor="rgba(124,58,237,0.08)"))
        fig.add_trace(go.Scatter(x=bench["fecha"], y=bench["valor"],
            name="Benchmark (S&P 500)",
            line=dict(color=MUTED, width=2, dash="dash")))
        fig.add_hline(y=100, line_dash="dot", line_color=MUTED, opacity=0.4)
        fig.update_layout(
            title="Rendimiento Acumulado Base 100 · Portafolio vs S&P 500",
            xaxis_title="Fecha", yaxis_title="Valor (Base 100)",
            height=440, **PLOT_TPL)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Alpha", f"{data['alpha_pct']:.2%}", delta=f"{data['alpha_pct']:.2%}")
        c2.metric("Tracking Error", f"{data['tracking_error_pct']:.2%}")
        c3.metric("Info. Ratio", f"{data['information_ratio']:.2f}")
        c4.metric("Máx. Drawdown", f"{data['max_drawdown_pct']:.2%}", delta_color="inverse")

        cb1, cb2 = st.columns(2)
        with cb1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**📐 Métricas de Selección**")
            st.table(pd.DataFrame({
                "Métrica": ["Alpha de Jensen","Tracking Error","Information Ratio"],
                "Valor": [f"{data['alpha_pct']:.2%}",
                          f"{data['tracking_error_pct']:.2%}",
                          f"{data['information_ratio']:.2f}"]
            }).set_index("Métrica"))
            st.markdown('</div>', unsafe_allow_html=True)
        with cb2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**📊 Resumen de Desempeño**")
            st.table(pd.DataFrame({
                "Métrica": ["Retorno Anualizado","Volatilidad Anual","Máx. Drawdown"],
                "Valor": [f"{data['rendimiento_portafolio_pct']:.2%}",
                          f"{data['volatilidad_anual_pct']:.2%}",
                          f"{data['max_drawdown_pct']:.2%}"]
            }).set_index("Métrica"))
            st.markdown('</div>', unsafe_allow_html=True)

        if data["alpha_pct"] > 0:
            badge_html(
                f"🚀 El portafolio superó al benchmark con Alpha = {data['alpha_pct']:.2%}. "
                f"La selección activa generó valor respecto al S&P 500.",
                "success"
            )
        else:
            badge_html(
                f"📉 El portafolio sub-performó al benchmark (Alpha = {data['alpha_pct']:.2%}). "
                f"Considera ajustar los pesos en Módulo 6 (Markowitz).",
                "warning"
            )
    else:
        st.markdown('<div class="badge-info">👆 Selecciona los activos del portafolio y pulsa <b>Calcular</b>.</div>', unsafe_allow_html=True)